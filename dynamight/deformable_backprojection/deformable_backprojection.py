#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:04:24 2023

@author: schwab
"""

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
    generate_smooth_mask_and_grids, generate_smooth_mask_from_consensus, \
    get_latent_space_and_indices, get_latent_space_tiling, backproject_images_from_tile, \
    backproject_single_image
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
    gpu_id: Optional[int] = 0,
    batch_size: int = Option(24),
    backprojection_batch_size: int = Option(1),
    preload_images: bool = Option(False),
    particle_diameter: Optional[float] = Option(None),
    mask_soft_edge_width: int = Option(20),
    data_loader_threads: int = Option(8),
    downsample: int = Option(2),
    mask_reconstruction: bool = Option(False),
    do_deformations: bool = Option(True),
    pipeline_control = None
):
    backprojection_directory = output_directory / 'backprojection'
    backprojection_directory.mkdir(exist_ok=True, parents=True)

    device = 'cuda:' + str(gpu_id)
    if inverse_deformation_directory is None:
        inverse_deformation_directory = output_directory / 'inverse_deformations'
    if vae_directory is None:
        vae_directory = output_directory / \
            'forward_deformations/checkpoints/checkpoint_final.pth'

    cp = torch.load(inverse_deformation_directory /
                    'inv_chkpt.pth', map_location=device)
    cp_vae = torch.load(vae_directory, map_location=device)

    if refinement_star_file is None:
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
    encoder_half1.eval()
    encoder_half2.eval()
    decoder_half1.eval()
    decoder_half2.eval()

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

    try:
        inds_val = cp_vae['indices_val'].cpu().numpy()
        inds_half1 = np.concatenate(
            [inds_half1, inds_val[:inds_val.shape[0]//2]])

    except:
        print('no validation set given')

    inds_half2 = list(set(range(len(dataset))) -
                      set(list(inds_half1)))

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
    inv_half1 = inv_half1.half().eval()
    inv_half2 = inv_half2.half().eval()

    latent_dim = inv_half1.latent_dim

    if mask_reconstruction is True:
        rec_mask_h1 = generate_smooth_mask_from_consensus(
            decoder_half1, box_size, ang_pix, distance=100, soft_edge=5
        )
        rec_mask_h2 = generate_smooth_mask_from_consensus(
            decoder_half2, box_size, ang_pix, distance=100, soft_edge=5
        )
    else:
        rec_mask_h1 = torch.ones(box_size, box_size, box_size).to(device)
        rec_mask_h2 = torch.ones(box_size, box_size, box_size).to(device)
    print('generate deformation mask')
    # def_mask_h1 = generate_smooth_mask_from_consensus(
    #     decoder_half1, box_size, ang_pix, distance=10, soft_edge=0
    # )
    # def_mask_h2 = generate_smooth_mask_from_consensus(
    #     decoder_half2, box_size, ang_pix, distance=10, soft_edge=0
    # )

    def_mask_h1 = torch.ones(box_size, box_size, box_size).to(device)
    def_mask_h2 = torch.ones(box_size, box_size, box_size).to(device)
    print('masks generated')
    def_mask = def_mask_h1 * def_mask_h2
    rec_mask = rec_mask_h1 * rec_mask_h2

    if mask_file is None:
        ess_grid, out_grid, sm_bin_mask = generate_smooth_mask_and_grids(
            def_mask, device
        )
    else:
        ess_grid, out_grid, sm_bin_mask = generate_smooth_mask_and_grids(
            str(mask_file), device
        )
    mrcfile.write(
        name=backprojection_directory / 'mask_reconstruction.mrc',
        data=rec_mask.float().cpu().numpy(),
        voxel_size=ang_pix,
        overwrite=True,
    )
    mrcfile.write(
        name=backprojection_directory / 'mask_deformation.mrc',
        data=def_mask.float().cpu().numpy(),
        voxel_size=ang_pix,
        overwrite=True,
    )

    print('Computing latent_space and indices for half 1')
    latent_space = torch.zeros(len(dataset), latent_dim)
    latent_space, latent_indices_half1 = get_latent_space_and_indices(
        data_loader_half1,
        encoder_half1,
        poses,
        latent_space,
        data_preprocessor,
        device
    )
    print('Computing latent_space and indices for half 2')
    latent_space, latent_indices_half2 = get_latent_space_and_indices(
        data_loader_half2,
        encoder_half2,
        poses,
        latent_space,
        data_preprocessor,
        device
    )

    print('Initialising output volume for half1')
    lam_thres = box_size ** 3 / 250 ** 3
    V = torch.zeros(box_size, box_size, box_size).to(device)
    i = 0
    gs = torch.linspace(-0.5, 0.5, box_size // downsample)
    Gs = torch.meshgrid(gs, gs, gs,indexing = 'ij')
    smallgrid = torch.stack([Gs[0].ravel(), Gs[1].ravel(), Gs[2].ravel()], 1)
    smallgrid, outsmallgrid = get_ess_grid(
        smallgrid, decoder_half1.model_positions, box_size
    )
    
    smallgrid = smallgrid.to(torch.float16)
    fwd_int = DeformationInterpolator(
        device, smallgrid, smallgrid, box_size, downsample
    )

    rotation = RotateVolume(box_size, device)
    print('initialising output volume containing filter for reconstruction')
    tot_filter = torch.zeros(box_size, box_size, box_size).to(device)

    print('start deformable_backprojection of half 1')
    current_data = torch.utils.data.Subset(dataset, latent_indices_half1)
    current_data_loader = DataLoader(
        dataset=current_data,
        batch_size=backprojection_batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=False,
        drop_last=True
    )
    nr = 0
    for batch_ndx, sample in tqdm(enumerate(current_data_loader)):
        r, y, ctf = sample["rotation"].to(torch.float16), sample["image"].to(torch.float16), sample[
            "ctf"]
        idx = sample['idx']

        z_image = latent_space[idx].to(torch.float16)

        Vol, Filter = backproject_single_image(
            z_image=z_image,
            decoder=decoder_half1,
            inverse_model=inv_half1,
            grid=smallgrid,
            interpolate_field=fwd_int,
            rotation=rotation,
            idx=idx,
            poses=poses,
            y=y,
            ctf=ctf,
            data_preprocessor=data_preprocessor,
            use_ctf=True,
            do_deformations=do_deformations)

        V += 1/len(dataset)*Vol.squeeze()
        tot_filter += 1/len(dataset)*Filter.squeeze()
        i += 1

        if i % (len(current_data_loader) // 100) == 0:

            try:
                VV = V[:, :, :] * rec_mask
            except:
                VV = V[:, :, :]
            VV = torch.fft.fftn(torch.fft.fftshift(
                VV, dim=[-1, -2, -3]), dim=[-1, -2, -3])

            VV2 = torch.fft.fftshift(torch.real(
                torch.fft.ifftn(VV / torch.maximum(tot_filter,
                                                   lam_thres * torch.ones_like(
                                                       tot_filter)),
                                dim=[-1, -2, -3])), dim=[-1, -2, -3])
            nr += 1
            mrcfile.write(
                name=backprojection_directory /
                f'reconstruction_half1_{nr:03}.mrc',
                data=(VV2).float().cpu().numpy(),
                voxel_size=decoder_half1.ang_pix,
                overwrite=True,
            )
    try:
        V = V * rec_mask
    except:
        V = V
    V = torch.fft.fftn(torch.fft.fftshift(
        V, dim=[-1, -2, -3]), dim=[-1, -2, -3])
    V = torch.fft.fftshift(torch.real(torch.fft.ifftn(
        V / torch.maximum(tot_filter, lam_thres * torch.ones_like(tot_filter)),
        dim=[-1, -2, -3])), dim=[-1, -2, -3])

    mrcfile.write(
        name=backprojection_directory / 'map_half1.mrc',
        data=(V).float().detach().cpu().numpy(),
        voxel_size=decoder_half1.ang_pix,
        overwrite=True,
    )

    del V, tot_filter

    print('Initialising output volume for half2')
    V = torch.zeros(box_size, box_size, box_size).to(device)
    tot_filter = torch.zeros(box_size, box_size, box_size).to(device)
    i = 0

    print('start deformable_backprojection of half 2')
    current_data = torch.utils.data.Subset(dataset, latent_indices_half2)
    current_data_loader = DataLoader(
        dataset=current_data,
        batch_size=backprojection_batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=False,
        drop_last=True
    )
    nr = 0
    for batch_ndx, sample in tqdm(enumerate(current_data_loader)):
        r, y, ctf = sample["rotation"].to(torch.float16), sample["image"].to(torch.float16), sample[
            "ctf"]
        idx = sample['idx']

        z_image = latent_space[idx].to(torch.float16)

        Vol, Filter = backproject_single_image(
            z_image=z_image,
            decoder=decoder_half2,
            inverse_model=inv_half2,
            grid=smallgrid,
            interpolate_field=fwd_int,
            rotation=rotation,
            idx=idx,
            poses=poses,
            y=y,
            ctf=ctf,
            data_preprocessor=data_preprocessor,
            use_ctf=True,
            do_deformations=do_deformations)

        V += 1/len(dataset)*Vol.squeeze()
        tot_filter += 1/len(dataset)*Filter.squeeze()
        i += 1

        if i % (len(current_data_loader) // 100) == 0:

            try:
                VV = V[:, :, :] * rec_mask
            except:
                VV = V[:, :, :]
            VV = torch.fft.fftn(torch.fft.fftshift(
                VV, dim=[-1, -2, -3]), dim=[-1, -2, -3])
            VV2 = torch.fft.fftshift(torch.real(
                torch.fft.ifftn(VV / torch.maximum(tot_filter,
                                                   lam_thres * torch.ones_like(
                                                       tot_filter)),
                                dim=[-1, -2, -3])), dim=[-1, -2, -3])
            nr += 1
            mrcfile.write(
                name=backprojection_directory /
                f'reconstruction_half2_{nr:03}.mrc',
                data=(VV2).float().cpu().numpy(),
                voxel_size=decoder_half2.ang_pix,
                overwrite=True,
            )

    try:
        V = V * rec_mask
    except:
        V = V
    V = torch.fft.fftn(torch.fft.fftshift(
        V, dim=[-1, -2, -3]), dim=[-1, -2, -3])
    V = torch.fft.fftshift(torch.real(torch.fft.ifftn(
        V / torch.maximum(tot_filter, lam_thres * torch.ones_like(tot_filter)),
        dim=[-1, -2, -3])), dim=[-1, -2, -3])

    with mrcfile.new(backprojection_directory / 'map_half2.mrc', overwrite=True) as mrc:
        mrc.set_data((V).float().detach().cpu().numpy())
        mrc.voxel_size = decoder_half2.ang_pix
    del V, tot_filter

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
    plt.plot(torch.ones(end_ind) * 0.143, c='slategrey', linestyle='dotted')
    plt.xticks(torch.arange(start=0, end=end_ind, step=10), labels=np.round(
        res[torch.arange(start=0, end=end_ind, step=10)].numpy(), 1)
    )
    plt.savefig(backprojection_directory / 'Fourier-Shell-Correlation.png')
