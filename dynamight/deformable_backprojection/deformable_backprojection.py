from pathlib import Path
from typing import Optional
import mrcfile
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy.spatial import KDTree
from dynamight.utils.utils_new import field2bild, FSC
from dynamight.deformable_backprojection.backprojection_utils import \
    get_ess_grid, DeformationInterpolator, RotateVolume, \
    generate_smooth_mask_and_grids, generate_smooth_mask_from_consensus, get_latent_space_and_indices, get_latent_space_tiling
from dynamight.data.handlers.particle_image_preprocessor import \
    ParticleImagePreprocessor
from tqdm import tqdm
from ..data.dataloaders.relion import RelionDataset
from typer import Option
import matplotlib.pyplot as plt
from .._cli import cli


@cli.command()
def deformable_backprojection(
    output_directory: Path,
    mask_file: Optional[Path] = None,
    refinement_star_file: Optional[Path] = None,
    vae_directory: Optional[Path] = None,
    inverse_deformation_directory: Optional[Path] = None,
    use_ctf: bool = Option(True),
    gpu_id: Optional[int] = 0,
    batch_size: int = Option(24),
    preload_images: bool = Option(False),
    #pooling_fraction: Optional[float] = Option(0.05),
    #pooling_multiplier: Optional[float] = Option(3),
    ignore_number: int = Option(0),
    latent_sampling: int = Option(100),
    particle_diameter: Optional[float] = Option(None),
    mask_soft_edge_width: int = Option(20),
    data_loader_threads: int = Option(4),
    downsample: int = Option(2),
):

    # To do:
    # Auto masking from the gaussian model (Generate 2 masks, one for deformation and one for the final reconstruction)

    backprojection_directory = output_directory / 'backprojection'
    backprojection_directory.mkdir(exist_ok=True, parents=True)

    device = 'cuda:' + str(gpu_id)
    if inverse_deformation_directory == None:
        inverse_deformation_directory = output_directory / 'inverse_deformations'
    if vae_directory == None:
        vae_directory = output_directory / \
            'forward_deformations/checkpoints/checkpoint_final.pth'

    cp = torch.load(inverse_deformation_directory /
                    'inv_chkpt.pth', map_location=device)
    cp_vae = torch.load(vae_directory, map_location=device)

    if refinement_star_file == None:
        refinement_star_file = cp_vae['refinement_directory']

    encoder_half1 = cp_vae['encoder_half1']
    encoder_half2 = cp_vae['encoder_half2']
    decoder_half1 = cp_vae['decoder_half1']
    decoder_half2 = cp_vae['decoder_half2']

    encoder_half1.load_state_dict(cp_vae['encoder_half1_state_dict'])
    encoder_half2.load_state_dict(cp_vae['encoder_half2_state_dict'])
    decoder_half1.load_state_dict(cp_vae['decoder_half1_state_dict'])
    decoder_half2.load_state_dict(cp_vae['decoder_half2_state_dict'])

    decoder_half1.p2i.device = device
    decoder_half2.p2i.device = device
    decoder_half1.projector.device = device
    decoder_half2.projector.device = device
    decoder_half1.image_smoother.device = device
    decoder_half2.image_smoother.device = device
    decoder_half1.p2v.device = device
    decoder_half2.p2v.device = device
    decoder_half1.device = device
    decoder_half2.device = device
    decoder_half1.to(device)
    decoder_half2.to(device)

    poses = cp_vae['poses']

    relion_dataset = RelionDataset(
        path=refinement_star_file,
        circular_mask_thickness=mask_soft_edge_width,
        particle_diameter=particle_diameter,
    )
    dataset = relion_dataset.make_particle_dataset()
    diameter_ang = relion_dataset.particle_diameter
    box_size = relion_dataset.box_size
    ang_pix = relion_dataset.pixel_spacing_angstroms

    if preload_images:
        dataset.preload_images()

    inds_half1 = cp_vae['indices_half1'].cpu().numpy()
    inds_half2 = list(set(range(len(dataset))) - set(list(inds_half1)))

    dataset_half1 = torch.utils.data.Subset(dataset, inds_half1)
    dataset_half2 = torch.utils.data.Subset(dataset, inds_half2)

    data_loader_half1 = DataLoader(
        dataset=dataset_half1,
        batch_size=batch_size,
        num_workers=data_loader_threads,
        shuffle=True,
        pin_memory=True
    )
    data_loader_half2 = DataLoader(
        dataset=dataset_half2,
        batch_size=batch_size,
        num_workers=data_loader_threads,
        shuffle=True,
        pin_memory=True
    )

    batch = next(iter(data_loader_half1))
    data_preprocessor = ParticleImagePreprocessor()
    data_preprocessor.initialize_from_stack(
        stack=batch["image"],
        circular_mask_radius=diameter_ang / (2 * ang_pix),
        circular_mask_thickness=mask_soft_edge_width / box_size
    )

    inv_half1 = cp['inv_half1']
    inv_half2 = cp['inv_half2']
    inv_half1.load_state_dict(cp['inv_half1_state_dict'])
    inv_half2.load_state_dict(cp['inv_half2_state_dict'])

    latent_dim = inv_half1.latent_dim

    rec_mask_h1 = generate_smooth_mask_from_consensus(
        decoder_half1, box_size, ang_pix, 5, soft_edge=5)
    rec_mask_h2 = generate_smooth_mask_from_consensus(
        decoder_half2, box_size, ang_pix, 5, soft_edge=5)

    def_mask_h1 = generate_smooth_mask_from_consensus(
        decoder_half1, box_size, ang_pix, 40, soft_edge=0)
    def_mask_h2 = generate_smooth_mask_from_consensus(
        decoder_half2, box_size, ang_pix, 40, soft_edge=0)

    def_mask = def_mask_h1*def_mask_h2
    rec_mask = rec_mask_h1*rec_mask_h2

    if mask_file == None:
        ess_grid, out_grid, sm_bin_mask = generate_smooth_mask_and_grids(def_mask,
                                                                         device)
    else:
        ess_grid, out_grid, sm_bin_mask = generate_smooth_mask_and_grids(str(mask_file),
                                                                         device)

    with mrcfile.new(backprojection_directory / 'mask_reconstruction.mrc', overwrite=True) as mrc:
        mrc.set_data(rec_mask.float().cpu().numpy())

    with mrcfile.new(backprojection_directory / 'mask_deformation.mrc', overwrite=True) as mrc:
        mrc.set_data(def_mask.float().cpu().numpy())

    print('Computing latent_space and indices for half 1')

    latent_space = torch.zeros(len(dataset), latent_dim)

    latent_space, latent_indices_half1 = get_latent_space_and_indices(data_loader_half1,
                                                                      encoder_half1,
                                                                      poses,
                                                                      latent_space,
                                                                      data_preprocessor)
    latent_space, latent_indices_half2 = get_latent_space_and_indices(data_loader_half2,
                                                                      encoder_half2,
                                                                      poses,
                                                                      latent_space,
                                                                      data_preprocessor)

    xx = get_latent_space_tiling(latent_space, latent_sampling)

    latent_space_half1 = latent_space[latent_indices_half1]
    latent_space_half2 = latent_space[latent_indices_half2]

    XY = torch.meshgrid(xx)
    gxy = torch.stack([X.ravel() for X in XY], 1)

    tree = KDTree(gxy.cpu().numpy())
    (dists_half1, latent_points_half1) = tree.query(
        latent_space_half1.cpu().numpy(), p=1)
    (dists_half2, latent_points_half2) = tree.query(
        latent_space_half2.cpu().numpy(), p=1)

    lam_thres = box_size ** 3 / 250 ** 3

    V = torch.zeros(box_size, box_size, box_size).to(device)

    i = 0
    gs = torch.linspace(-0.5, 0.5, box_size // downsample)
    Gs = torch.meshgrid(gs, gs, gs)
    smallgrid = torch.stack([Gs[0].ravel(), Gs[1].ravel(), Gs[2].ravel()], 1)
    smallgrid, outsmallgrid = get_ess_grid(
        smallgrid, decoder_half1.model_positions, box_size)

    gss = torch.linspace(-0.5, 0.5, box_size // 8)
    Gss = torch.meshgrid(gss, gss, gss)
    supersmallgrid = torch.stack(
        [Gss[0].ravel(), Gss[1].ravel(), Gss[2].ravel()], 1)

    fwd_int = DeformationInterpolator(device, smallgrid, smallgrid, box_size,
                                      downsample)

    rotation = RotateVolume(box_size, device)
    CTF = torch.zeros(box_size, box_size, box_size).to(device)
    om_imgs = torch.zeros(1)
    bp_imgs = torch.zeros(1)

    his = []
    rel_inds = []

    print('start deformable_backprojection of half 1')

    print('computing relevant tiles')
    for j in range(gxy.shape[0]):
        tile_indices = latent_indices_half1[latent_points_half1 == j]
        if len(tile_indices) > 0:
            his.append(len(tile_indices))
            rel_inds.append(j)
    # plt.hist(his,bins = 100)
    # plt.show()
    print(np.sum(np.array(his) == 1), 'tiles with single particles')
    print('tile with the most images has ', np.max(np.array(his)), 'particles')

    for j in tqdm(rel_inds):
        tile_indices = latent_indices_half1[latent_points_half1 == j]
        current_data = torch.utils.data.Subset(dataset, tile_indices)
        current_data_loader = DataLoader(
            dataset=current_data,
            batch_size=8,
            num_workers=8,
            shuffle=False,
            pin_memory=False
        )
        z_tile = torch.stack(2 * [gxy[j]], 0).to(device)
        r = torch.zeros(2, 3)
        t = torch.zeros(2, 2)

        if len(tile_indices) > ignore_number:
            with torch.no_grad():
                _, n_pos, deformation = decoder_half1(z_tile.to(device),
                                                      r.to(device),
                                                      t.to(device),
                                                      supersmallgrid.to(
                                                          device),
                                                      )
                tile_deformation = inv_half1(z_tile, torch.stack(
                    2 * [smallgrid.to(device)], 0))

                disp = fwd_int.compute_field(tile_deformation[0])
                disp0 = disp[0, ...]
                disp1 = disp[1, ...]
                disp2 = disp[2, ...]
                disp = torch.stack(
                    [disp0.squeeze(), disp1.squeeze(), disp2.squeeze()], 3)
                dis_grid = disp[None, :, :, :]
                # if i % 200 == 0:
                #     im_deformation = inv_half1(z_tile, n_pos)
                #     field2bild(n_pos[0], im_deformation[0],
                #                str(output_directory) + 'Ninversefield' + str(
                #                    i).zfill(3),
                #                box_size, ang_pix)
                #     field2bild(supersmallgrid.to(device), n_pos[0],
                #                str(output_directory) + 'Nforwardfield' + str(
                #                    i).zfill(3),
                #                box_size, ang_pix)
                #     field2bild(supersmallgrid.to(device), im_deformation[0],
                #                str(output_directory) + 'Nfwdbwdfield' + str(i).zfill(
                #                    3), box_size,
                #                ang_pix)

                bp_imgs += len(tile_indices)

        else:
            om_imgs += len(tile_indices)
        if len(tile_indices) > ignore_number:
            i = i + 1
            if i % 50 == 0:
                try:
                    VV = V[:, :, :] * rec_mask
                except:
                    VV = V[:, :, :]
                VV = torch.fft.fftn(torch.fft.fftshift(
                    VV, dim=[-1, -2, -3]), dim=[-1, -2, -3])

                VV2 = torch.fft.fftshift(torch.real(
                    torch.fft.ifftn(VV / torch.maximum(CTF,
                                                       lam_thres * torch.ones_like(
                                                           CTF)),
                                    dim=[-1, -2, -3])), dim=[-1, -2, -3])
                nr = i//50
                with mrcfile.new(backprojection_directory / ('reconstruction_half1_' + f'{nr:03}.mrc'), overwrite=True) as mrc:
                    mrc.set_data((VV2 / torch.max(VV2)).float().cpu().numpy())

            with torch.no_grad():
                for batch_ndx, sample in enumerate(current_data_loader):
                    r, y, ctf = sample["rotation"], sample["image"], sample[
                        "ctf"]
                    idx = sample['idx']
                    r, t = poses(idx)
                    batch_size = y.shape[0]
                    ctfs_l = torch.nn.functional.pad(ctf, (
                        box_size // 2, box_size // 2, box_size // 2,
                        box_size // 2, 0, 0)).to(device)
                    ctfs = torch.fft.fftshift(ctf, dim=[-1, -2])
                    y_in = data_preprocessor.apply_square_mask(y)
                    y_in = data_preprocessor.apply_translation(y_in, -t[:, 0],
                                                               -t[:, 1])
                    y_in = data_preprocessor.apply_circular_mask(y_in)
                    y_in, r, t, ctfs, ctf = y_in.to(device), r.to(device), t.to(
                        device), ctfs.to(
                        device), ctf.to(device)
                    if use_ctf:
                        y = torch.fft.fft2(y_in, dim=[-1, -2])
                        y = y * ctfs
                        yr = torch.real(
                            torch.fft.ifft2(y, dim=[-1, -2])).unsqueeze(1)
                    else:
                        yr = y_in.unsqueeze(1)

                    ctfr2_l = torch.fft.fftshift(torch.real(
                        torch.fft.ifft2(
                            torch.fft.fftshift(ctfs_l, dim=[-1, -2]),
                            dim=[-1, -2], )).unsqueeze(1), dim=[-1, -2])
                    ctfr2_lc = torch.nn.functional.avg_pool2d(ctfr2_l,
                                                              kernel_size=3,
                                                              stride=2,
                                                              padding=1)
                    CTFy = ctfr2_lc.expand(batch_size, box_size, box_size,
                                           box_size)
                    CTFy = torch.nn.functional.pad(CTFy,
                                                   (0, 1, 0, 1, 0, 1, 0, 0))
                    CTFy = rotation(CTFy.unsqueeze(1), r).squeeze()

                    if len(CTFy.shape) < 4:
                        CTFy = CTFy.unsqueeze(0)

                    CTFy = CTFy[:, :-1, :-1, :-1]
                    CTF += (1 / len(dataset)) * torch.real(
                        torch.sum(torch.fft.fftn(torch.fft.fftshift(CTFy.unsqueeze(1), dim=[-1, -2, -3]),
                                                 dim=[-1, -2, -3]) ** 2,
                                  0).squeeze())

                    Vy = yr.expand(batch_size, box_size, box_size, box_size)
                    Vy = torch.nn.functional.pad(Vy, (0, 1, 0, 1, 0, 1, 0, 0))
                    Vy = rotation(Vy.unsqueeze(1), r).squeeze()
                    if len(Vy.shape) < 4:
                        Vy = Vy.unsqueeze(0)
                    if len(Vy.shape) < 4:
                        Vy = Vy.unsqueeze(0)

                    Vy = torch.sum(Vy, 0)
                    Vy = F.grid_sample(input=Vy.unsqueeze(0).unsqueeze(0),
                                       grid=dis_grid.to(device),
                                       mode='bilinear', align_corners=False)
                    V += (1 / len(dataset)) * Vy.squeeze()

    try:
        V = V * rec_mask
    except:
        V = V
    V = torch.fft.fftn(torch.fft.fftshift(
        V, dim=[-1, -2, -3]), dim=[-1, -2, -3])
    V = torch.fft.fftshift(torch.real(torch.fft.ifftn(
        V / torch.maximum(CTF, lam_thres * torch.ones_like(CTF)),
        dim=[-1, -2, -3])), dim=[-1, -2, -3])

    with mrcfile.new(backprojection_directory / 'map_half1.mrc', overwrite=True) as mrc:
        mrc.set_data((V / torch.max(V)).float().detach().cpu().numpy())

    del V, CTF
    V = torch.zeros(box_size, box_size, box_size).to(device)
    CTF = torch.zeros(box_size, box_size, box_size).to(device)
    i = 0
    his = []
    rel_inds = []

    print('start deformable_backprojection of half 2')
    print('computing relevant tiles')
    for j in range(gxy.shape[0]):
        tile_indices = latent_indices_half2[latent_points_half2 == j]
        if len(tile_indices) > 0:
            his.append(len(tile_indices))
            rel_inds.append(j)
    # plt.hist(his,bins = 100)
    # plt.show()
    print(np.sum(np.array(his) == 1), 'tiles with single particles')
    print('tile with the most images has ', np.max(np.array(his)), 'particles')

    for j in tqdm(rel_inds):
        tile_indices = latent_indices_half2[latent_points_half2 == j]
        current_data = torch.utils.data.Subset(dataset, tile_indices)
        current_data_loader = DataLoader(
            dataset=current_data,
            batch_size=8,
            num_workers=8,
            shuffle=False,
            pin_memory=False
        )
        z_tile = torch.stack(2 * [gxy[j]], 0).to(device)
        r = torch.zeros(2, 3)
        t = torch.zeros(2, 2)

        if len(tile_indices) > ignore_number:
            with torch.no_grad():
                tile_deformation = inv_half2(z_tile, torch.stack(
                    2 * [smallgrid.to(device)], 0))

                disp = fwd_int.compute_field(tile_deformation[0])
                disp0 = disp[0, ...]
                disp1 = disp[1, ...]
                disp2 = disp[2, ...]
                disp = torch.stack(
                    [disp0.squeeze(), disp1.squeeze(), disp2.squeeze()], 3)
                dis_grid = disp[None, :, :, :]

                # print('Backprojected', bp_imgs.cpu().numpy(), 'particles from',
                #       len(dataset),
                #       'and omitted backprojection of', om_imgs.cpu().numpy(),
                #       'particles')
                # bp_imgs += len(tile_indices)

                del tile_deformation
        else:
            om_imgs += len(tile_indices)
        if len(tile_indices) > ignore_number:
            i = i + 1
            if i % 50 == 0:
                try:
                    VV = V[:, :, :] * rec_mask
                except:
                    VV = V[:, :, :]
                VV = torch.fft.fftn(torch.fft.fftshift(
                    VV, dim=[-1, -2, -3]), dim=[-1, -2, -3])

                VV2 = torch.fft.fftshift(torch.real(
                    torch.fft.ifftn(VV / torch.maximum(CTF,
                                                       lam_thres * torch.ones_like(
                                                           CTF)),
                                    dim=[-1, -2, -3])), dim=[-1, -2, -3])

                with mrcfile.new(backprojection_directory / ('reconstruction_half2_' + f'{i//50:03}.mrc'), overwrite=True) as mrc:
                    mrc.set_data((VV2 / torch.max(VV2)).float().cpu().numpy())

            with torch.no_grad():
                for batch_ndx, sample in enumerate(current_data_loader):
                    r, y, ctf = sample["rotation"], sample["image"], sample[
                        "ctf"]
                    idx = sample['idx']
                    r, t = poses(idx)
                    batch_size = y.shape[0]
                    ctfs_l = torch.nn.functional.pad(ctf, (
                        box_size // 2, box_size // 2, box_size // 2,
                        box_size // 2, 0, 0)).to(device)
                    ctfs = torch.fft.fftshift(ctf, dim=[-1, -2])
                    y_in = data_preprocessor.apply_square_mask(y)
                    y_in = data_preprocessor.apply_translation(y_in, -t[:, 0],
                                                               -t[:, 1])
                    y_in = data_preprocessor.apply_circular_mask(y_in)
                    y_in, r, t, ctfs, ctf = y_in.to(device), r.to(device), t.to(
                        device), ctfs.to(
                        device), ctf.to(device)
                    if use_ctf:
                        y = torch.fft.fft2(y_in, dim=[-1, -2])
                        y = y * ctfs
                        yr = torch.real(
                            torch.fft.ifft2(y, dim=[-1, -2])).unsqueeze(1)
                    else:
                        yr = y_in.unsqueeze(1)

                    ctfr2_l = torch.fft.fftshift(torch.real(
                        torch.fft.ifft2(
                            torch.fft.fftshift(ctfs_l, dim=[-1, -2]),
                            dim=[-1, -2], )).unsqueeze(1), dim=[-1, -2])
                    ctfr2_lc = torch.nn.functional.avg_pool2d(ctfr2_l,
                                                              kernel_size=3,
                                                              stride=2,
                                                              padding=1)
                    CTFy = ctfr2_lc.expand(batch_size, box_size, box_size,
                                           box_size)
                    CTFy = torch.nn.functional.pad(CTFy,
                                                   (0, 1, 0, 1, 0, 1, 0, 0))
                    CTFy = rotation(CTFy.unsqueeze(1), r).squeeze()

                    if len(CTFy.shape) < 4:
                        CTFy = CTFy.unsqueeze(0)

                    CTFy = CTFy[:, :-1, :-1, :-1]
                    CTF += (1 / len(dataset)) * torch.real(
                        torch.sum(torch.fft.fftn(torch.fft.fftshift(CTFy.unsqueeze(1), dim=[-1, -2, -3]),
                                                 dim=[-1, -2, -3]) ** 2,
                                  0).squeeze())

                    Vy = yr.expand(batch_size, box_size, box_size, box_size)
                    Vy = torch.nn.functional.pad(Vy, (0, 1, 0, 1, 0, 1, 0, 0))
                    Vy = rotation(Vy.unsqueeze(1), r).squeeze()
                    if len(Vy.shape) < 4:
                        Vy = Vy.unsqueeze(0)
                    if len(Vy.shape) < 4:
                        Vy = Vy.unsqueeze(0)

                    Vy = torch.sum(Vy, 0)
                    Vy = F.grid_sample(input=Vy.unsqueeze(0).unsqueeze(0),
                                       grid=dis_grid.to(device),
                                       mode='bilinear', align_corners=False)
                    V += (1 / len(dataset)) * Vy.squeeze()

    try:
        V = V * rec_mask
    except:
        V = V
    V = torch.fft.fftn(torch.fft.fftshift(
        V, dim=[-1, -2, -3]), dim=[-1, -2, -3])

    V = torch.fft.fftshift(torch.real(torch.fft.ifftn(
        V / torch.maximum(CTF, lam_thres * torch.ones_like(CTF)),
        dim=[-1, -2, -3])), dim=[-1, -2, -3])

    with mrcfile.new(backprojection_directory / 'map_half2.mrc', overwrite=True) as mrc:
        mrc.set_data((V / torch.max(V)).float().detach().cpu().numpy())

    del V, CTF

    with mrcfile.open(backprojection_directory / 'map_half1.mrc') as mrc:
        map_h1 = torch.tensor(mrc.data).to(device)

    with mrcfile.open(backprojection_directory / 'map_half2.mrc') as mrc:
        map_h2 = torch.tensor(mrc.data).to(device)

    fsc, res = FSC(map_h1, map_h2)
    end_ind = torch.round(torch.tensor(map_h1.shape[-1] / 2)).long()
    plt.figure(figsize=(10, 10))
    plt.rcParams['axes.xmargin'] = 0
    plt.plot(fsc[:end_ind].cpu(), c='r')
    plt.plot(torch.ones(end_ind) * 0.5, c='black', linestyle='dashed')
    plt.plot(torch.ones(end_ind) * 0.143,
             c='slategrey', linestyle='dotted')
    plt.xticks(torch.arange(start=0, end=end_ind, step=10), labels=np.round(
        res[torch.arange(start=0, end=end_ind, step=10)].numpy(), 1))
    plt.savefig(backprojection_directory / 'Fourier-Shell-Correlation.png')
