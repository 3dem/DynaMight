from pathlib import Path
from typing import Optional
import mrcfile
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy.spatial import KDTree
from dynamight.utils.utils_new import initialize_dataset, field2bild
from dynamight.deformable_backprojection.backprojection_utils import \
    get_ess_grid, DeformationInterpolator, RotateVolume, \
    generate_smooth_mask_and_grids
from dynamight.data.handlers.particle_image_preprocessor import \
    ParticleImagePreprocessor

from typer import Option

from .._cli import cli

@cli.command()
def deformable_backprojection(
    refinement_star_file: Path,
    vae_directory: Path,
    inverse_deformation_directory: Path,
    output_directory: Path,
    mask_file: Path = Option(None),
    use_ctf: bool = Option(True),
    gpu_id: Optional[int] = Option(None),
    batch_size: int = Option(24),
    preload_images: bool = Option(False),
    pooling_fraction: Optional[float] = Option(0.05),
    pooling_multiplier: Optional[float] = Option(3),
    particle_diameter: Optional[float] = Option(None),
    mask_soft_edge_width: int = Option(20),
    data_loader_threads: int = Option(4),
    downsample: int = Option(2),
):
    # setup output directories
    output_directory.mkdir(exist_ok=True, parents=True)

    device = 'cuda:' + str(gpu_id)

    cp = torch.load(output_directory + 'inv_chkpt.pth', map_location=device)
    cp_vae = torch.load(vae_directory, map_location=device)

    encoder_half1 = cp_vae['encoder_half1']
    encoder_half2 = cp_vae['encoder_half2']
    cons_model = cp_vae['consensus']
    decoder_half1 = cp_vae['decoder_half1']
    decoder_half2 = cp_vae['decoder_half2']

    encoder_half1.load_state_dict(cp_vae['encoder_half1_state_dict'])
    encoder_half2.load_state_dict(cp_vae['encoder_half2_state_dict'])
    cons_model.load_state_dict(cp_vae['consensus_state_dict'])
    decoder_half1.load_state_dict(cp_vae['decoder_half1_state_dict'])
    decoder_half2.load_state_dict(cp_vae['decoder_half2_state_dict'])

    n_classes = 2

    points = cons_model.pos.detach().cpu()
    points = torch.tensor(points)
    cons_model.pos = torch.nn.Parameter(points, requires_grad=False)
    cons_model.n_points = len(points)
    decoder_half1.n_points = cons_model.n_points
    decoder_half2.n_points = cons_model.n_points
    cons_model.ampvar = torch.nn.Parameter(
        torch.ones(n_classes, decoder_half1.n_points).to(device),
        requires_grad=False)

    cons_model.p2i.device = device
    decoder_half1.p2i.device = device
    decoder_half2.p2i.device = device
    cons_model.proj.device = device
    decoder_half1.proj.device = device
    decoder_half2.proj.device = device
    cons_model.i2F.device = device
    decoder_half1.i2F.device = device
    decoder_half2.i2F.device = device
    cons_model.p2v.device = device
    decoder_half1.p2v.device = device
    decoder_half2.p2v.device = device
    decoder_half1.device = device
    decoder_half2.device = device
    cons_model.device = device
    decoder_half1.to(device)
    decoder_half2.to(device)
    cons_model.to(device)

    encoder_half1.to(device)
    encoder_half2.to(device)

    poses = cp_vae['poses']

    circular_mask_thickness = mask_soft_edge_width
    dataset, diameter_ang, box_size, ang_pix, optics_group = initialize_dataset(
        refinement_star_file, circular_mask_thickness, preload_images,
        particle_diameter)

    max_diameter_ang = box_size * optics_group[
        'pixel_size'] - circular_mask_thickness

    if particle_diameter is None:
        diameter_ang = box_size * 1 * optics_group[
            'pixel_size'] - circular_mask_thickness
        print(f"Assigning a diameter of {round(diameter_ang)} angstrom")
    else:
        if particle_diameter > max_diameter_ang:
            print(
                f"WARNING: Specified particle diameter {round(particle_diameter)} angstrom is too large\n"
                f" Assigning a diameter of {round(max_diameter_ang)} angstrom"
            )
            diameter_ang = max_diameter_ang
        else:
            diameter_ang = particle_diameter

    if preload_images:
        dataset.preload_images()

    inds_half1 = cp['indices_half1'].cpu().numpy()
    inds_half2 = list(set(range(len(dataset))) - set(list(inds_half1)))
    print(len(inds_half1))
    print(len(inds_half2))
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
        circular_mask_radius=diameter_ang / (2 * optics_group['pixel_size']),
        circular_mask_thickness=circular_mask_thickness / optics_group[
            'pixel_size']
    )

    inv_half1 = cp['inv_half1']
    inv_half2 = cp['inv_half2']
    inv_half1.load_state_dict(cp['inv_half1_state_dict'])
    inv_half2.load_state_dict(cp['inv_half2_state_dict'])

    latent_dim = inv_half1.latent_dim

    ess_grid, out_grid, sm_bin_mask = generate_smooth_mask_and_grids(mask_file,
                                                                     device)

    print('Computing latent_space and indices for half 1')
    latent_space = torch.zeros(len(dataset), latent_dim)
    latent_indices_half1 = []

    for batch_ndx, sample in enumerate(data_loader_half1):
        with torch.no_grad():
            r, y, ctf = sample["rotation"], sample["image"], sample["ctf"]
            idx = sample['idx']
            r, t = poses(idx)
            batch_size = y.shape[0]
            ctfs_l = torch.nn.functional.pad(ctf, (
                box_size // 2, box_size // 2, box_size // 2, box_size // 2, 0,
                0)).to(device)
            ctfs = torch.fft.fftshift(ctf, dim=[-1, -2])
            y_in = data_preprocessor.apply_square_mask(y)
            y_in = data_preprocessor.apply_translation(y_in, -t[:, 0], -t[:, 1])
            y_in = data_preprocessor.apply_circular_mask(y_in)
            mu, _ = encoder_half1(y_in.to(device), ctfs.to(device))
            latent_space[sample["idx"].cpu().numpy()] = mu.detach().cpu()
            latent_indices_half1.append(sample['idx'])

    print('Computing latent_space and indices for half2')
    latent_indices_half2 = []

    for batch_ndx, sample in enumerate(data_loader_half2):
        with torch.no_grad():
            r, y, ctf = sample["rotation"], sample["image"], sample["ctf"]
            idx = sample['idx']
            r, t = poses(idx)
            batch_size = y.shape[0]
            ctfs_l = torch.nn.functional.pad(ctf, (
                box_size // 2, box_size // 2, box_size // 2, box_size // 2, 0,
                0)).to(device)
            ctfs = torch.fft.fftshift(ctf, dim=[-1, -2])
            y_in = data_preprocessor.apply_square_mask(y)
            y_in = data_preprocessor.apply_translation(y_in, -t[:, 0], -t[:, 1])
            y_in = data_preprocessor.apply_circular_mask(y_in)
            mu, _ = encoder_half2(y_in.to(device), ctfs.to(device))
            latent_space[sample["idx"].cpu().numpy()] = mu.detach().cpu()
            latent_indices_half2.append(sample['idx'])

    diam = torch.zeros(1)
    xmin = torch.zeros(latent_dim)

    for i in range(latent_dim):
        xmin[i] = torch.min(latent_space[:, i])

        xmax = torch.max(latent_space[:, i])
        diam = torch.maximum(xmax - xmin[i], diam)

    max_side = diam
    xx = []
    for i in range(latent_dim):
        xx.append(torch.linspace(xmin[i], xmin[i] + max_side[0],
                                 args.latent_sampling))

    latent_indices_half1 = torch.cat(latent_indices_half1, 0)
    latent_indices_half2 = torch.cat(latent_indices_half2, 0)

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
    cons_model.amp = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    gs = torch.linspace(-0.5, 0.5, box_size // downsample)
    Gs = torch.meshgrid(gs, gs, gs)
    smallgrid = torch.stack([Gs[0].ravel(), Gs[1].ravel(), Gs[2].ravel()], 1)
    smallgrid, outsmallgrid = get_ess_grid(smallgrid, cons_model.pos)

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

    for j in rel_inds:
        tile_indices = latent_indices_half1[latent_points_half1 == j]
        current_data = torch.utils.data.Subset(dataset, tile_indices)
        current_data_loader = DataLoader(
            dataset=current_data,
            batch_size=8,
            num_workers=8,
            shuffle=False,
            pin_memory=False
        )
        z_tile = [torch.stack(2 * [gxy[j]], 0).to(device)]
        print(j, len(tile_indices))
        r = torch.zeros(2, 3)
        t = torch.zeros(2, 2)

        if len(tile_indices) > args.ignore:
            with torch.no_grad():
                _, _, _, n_pos, deformation = decoder_half1(z_tile,
                                                            r.to(device),
                                                            supersmallgrid.to(
                                                                device),
                                                            cons_model.amp.to(
                                                                device),
                                                            torch.ones(
                                                                n_classes,
                                                                supersmallgrid.shape[
                                                                    0]).to(
                                                                device),
                                                            t.to(device))
                tile_deformation = inv_half1(z_tile, torch.stack(
                    2 * [smallgrid.to(device)], 0))

                disp = fwd_int.compute_field(tile_deformation[0])
                disp0 = disp[0, ...]
                disp1 = disp[1, ...]
                disp2 = disp[2, ...]
                disp = torch.stack(
                    [disp0.squeeze(), disp1.squeeze(), disp2.squeeze()], 3)
                dis_grid = disp[None, :, :, :]
                if i % 200 == 0:
                    im_deformation = inv_half1(z_tile, n_pos)
                    field2bild(n_pos[0], im_deformation[0],
                               output_directory + 'Ninversefield' + str(
                                   i).zfill(3),
                               box_size, ang_pix)
                    field2bild(supersmallgrid.to(device), n_pos[0],
                               output_directory + 'Nforwardfield' + str(
                                   i).zfill(3),
                               box_size, ang_pix)
                    field2bild(supersmallgrid.to(device), im_deformation[0],
                               output_directory + 'Nfwdbwdfield' + str(i).zfill(
                                   3), box_size,
                               ang_pix)
                    print('saved field')

                print('Backprojected', bp_imgs.cpu().numpy(), 'particles from',
                      len(dataset),
                      'and omitted backprojection of', om_imgs.cpu().numpy(),
                      'particles')

                bp_imgs += len(tile_indices)

        else:
            om_imgs += len(tile_indices)
        if len(tile_indices) > args.ignore:
            print('Starting backprojection of tile', j, 'from', gxy.shape[0],
                  '::',
                  len(tile_indices), 'images')
            i = i + 1
            if i % 50 == 0:
                try:
                    VV = V[:, :, :] * sm_bin_mask
                except:
                    VV = V[:, :, :]
                VV = torch.fft.fftn(VV, dim=[-1, -2, -3])

                VV2 = torch.real(
                    torch.fft.ifftn(VV / torch.maximum(CTF,
                                                       lam_thres * torch.ones_like(
                                                           CTF)),
                                    dim=[-1, -2, -3]))

                with mrcfile.new(
                    output_directory + 'reconstruction_half1_' + str(
                        i // 50).zfill(3) + '.mrc', overwrite=True) as mrc:
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
                        torch.sum(torch.fft.fftn(CTFy.unsqueeze(1),
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
        V = V * sm_bin_mask
    except:
        V = V
    V = torch.fft.fftn(V, dim=[-1, -2, -3])
    V = torch.real(torch.fft.ifftn(
        V / torch.maximum(CTF, lam_thres * torch.ones_like(CTF)),
        dim=[-1, -2, -3]))

    with mrcfile.new(output_directory + 'map_half1.mrc', overwrite=True) as mrc:
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

    for j in rel_inds:
        tile_indices = latent_indices_half2[latent_points_half2 == j]
        current_data = torch.utils.data.Subset(dataset, tile_indices)
        current_data_loader = DataLoader(
            dataset=current_data,
            batch_size=8,
            num_workers=8,
            shuffle=False,
            pin_memory=False
        )
        z_tile = [torch.stack(2 * [gxy[j]], 0).to(device)]
        print(j, len(tile_indices))
        r = torch.zeros(2, 3)
        t = torch.zeros(2, 2)

        if len(tile_indices) > args.ignore:
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

                print('Backprojected', bp_imgs.cpu().numpy(), 'particles from',
                      len(dataset),
                      'and omitted backprojection of', om_imgs.cpu().numpy(),
                      'particles')
                bp_imgs += len(tile_indices)

                del tile_deformation
        else:
            om_imgs += len(tile_indices)
        if len(tile_indices) > args.ignore:
            print('Starting backprojection of tile', j, 'from', gxy.shape[0],
                  '::',
                  len(tile_indices), 'images')
            i = i + 1
            if i % 50 == 0:
                try:
                    VV = V[:, :, :] * sm_bin_mask
                except:
                    VV = V[:, :, :]
                VV = torch.fft.fftn(VV, dim=[-1, -2, -3])

                VV2 = torch.real(
                    torch.fft.ifftn(VV / torch.maximum(CTF,
                                                       lam_thres * torch.ones_like(
                                                           CTF)),
                                    dim=[-1, -2, -3]))

                with mrcfile.new(
                    output_directory + 'reconstruction_half2_' + str(
                        i // 50).zfill(3) + '.mrc', overwrite=True) as mrc:
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
                        torch.sum(torch.fft.fftn(CTFy.unsqueeze(1),
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
        V = V * sm_bin_mask
    except:
        V = V
    V = torch.fft.fftn(V, dim=[-1, -2, -3])

    V = torch.real(torch.fft.ifftn(
        V / torch.maximum(CTF, lam_thres * torch.ones_like(CTF)),
        dim=[-1, -2, -3]))

    with mrcfile.new(output_directory + 'map_half2.mrc', overwrite=True) as mrc:
        mrc.set_data((V / torch.max(V)).float().detach().cpu().numpy())

    del V, CTF


