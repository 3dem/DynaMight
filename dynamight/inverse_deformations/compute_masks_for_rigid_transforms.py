#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:42:59 2023

@author: schwab
"""


from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from typer import Option
import os
import sys

from ..data.handlers.particle_image_preprocessor import \
    ParticleImagePreprocessor
from ..models.blocks import LinearBlock
from ..models.decoder import InverseDisplacementDecoder
from ..utils.utils_new import initialize_dataset, add_weight_decay_to_named_parameters, maskpoints, points2bild
from ..data.dataloaders.relion import RelionDataset, abort_if_relion_abort, write_relion_job_exit_status
from ._optimize_single_epoch import optimize_epoch
from .rigid import get_rotation_translation, initialize_masks, compute_spatial_cluster_probability, compute_transformation_probability
from tqdm import tqdm
from .._cli import cli
import numpy as np
import mrcfile
import starfile
import matplotlib.pyplot as plt


@cli.command()
def compute_masks_for_rigid_transforms(
    output_directory: Path,
    refinement_star_file: Optional[Path] = None,
    checkpoint_file: Optional[Path] = None,
    batch_size: int = Option(100),
    gpu_id: Optional[int] = Option(0),
    particle_diameter: Optional[float] = Option(None),
    mask_soft_edge_width: int = Option(20),
    data_loader_threads: int = Option(4),
    pipeline_control=None,
    number_of_masks: int = 4,
    refinement_iterations: int = 2,
    mask_resolution: float = 20
):

    masks_directory = output_directory / 'masks'
    masks_directory.mkdir(exist_ok=True, parents=True)
    forward_deformations_directory = output_directory / \
        'forward_deformations' / 'checkpoints'
    if not forward_deformations_directory.exists():
        raise NotADirectoryError(
            f'{forward_deformations_directory} does not exist. Please run dynamight optimize-deformations or use a checkpoint file')
    device = 'cuda:' + str(gpu_id)
    if checkpoint_file is None:
        checkpoint_file = forward_deformations_directory / 'checkpoint_final.pth'

    checkpoint = torch.load(checkpoint_file, map_location=device)

    if refinement_star_file == None:
        refinement_star_file = checkpoint['refinement_directory']

    encoder_half1 = checkpoint['encoder_half1']
    encoder_half2 = checkpoint['encoder_half2']
    decoder_half1 = checkpoint['decoder_half1']
    decoder_half2 = checkpoint['decoder_half2']
    poses = checkpoint['poses']

    encoder_half1.load_state_dict(checkpoint['encoder_half1_state_dict'])
    encoder_half2.load_state_dict(checkpoint['encoder_half2_state_dict'])
    decoder_half1.load_state_dict(checkpoint['decoder_half1_state_dict'])
    decoder_half2.load_state_dict(checkpoint['decoder_half2_state_dict'])

    decoder_half1.mask = None
    decoder_half2.mask = None

    n_points = decoder_half1.n_points

    points = decoder_half1.model_positions.detach().cpu()
    points = torch.tensor(points)

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

    encoder_half1.to(device)
    encoder_half2.to(device)

    latent_dim = encoder_half1.latent_dim

    star_file = starfile.read(refinement_star_file)

    relion_dataset = RelionDataset(
        path=refinement_star_file,
        circular_mask_thickness=mask_soft_edge_width,
        particle_diameter=particle_diameter,
    )
    particle_dataset = relion_dataset.make_particle_dataset()
    diameter_ang = relion_dataset.particle_diameter
    box_size = relion_dataset.box_size
    ang_pix = relion_dataset.pixel_spacing_angstroms

    inds_half1 = checkpoint['indices_half1'].cpu().numpy()
    try:
        inds_val = checkpoint['indices_val'].cpu().numpy()
        inds_half1 = np.concatenate(
            [inds_half1, inds_val[:inds_val.shape[0]//2]])
    except:
        print('no validation set given')

    inds_half2 = list(set(range(len(particle_dataset))) -
                      set(list(inds_half1)))

    dataset_half1 = torch.utils.data.Subset(particle_dataset, inds_half1)
    dataset_half2 = torch.utils.data.Subset(particle_dataset, inds_half2)

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
        circular_mask_thickness=mask_soft_edge_width / ang_pix
    )

    total_labels, current_labels = initialize_masks(
        encoder_half1, decoder_half1, data_loader_half1, poses, data_preprocessor, star_file, half=1, number_of_classes=number_of_masks)

    # points2bild(decoder_half1.model_positions, torch.ones(
    #    decoder_half1.model_positions.shape[0]), '/masks_directory/initial', decoder_half1.box_size, decoder_half1.ang_pix, 5*current_labels)

    probs = compute_spatial_cluster_probability(
        current_labels, decoder_half1.model_positions)

    N_iter = refinement_iterations
    masks = []

    for i in range(N_iter):

        spat_probs = compute_spatial_cluster_probability(
            current_labels, decoder_half1.model_positions)
        masked_points = [decoder_half1.model_positions[current_labels == i]
                         for i in range(number_of_masks)]
        rot_probs = compute_transformation_probability(
            encoder_half1, decoder_half1, data_loader_half1, poses, data_preprocessor, masked_points)
        tot_probs = rot_probs*spat_probs
        classes = torch.zeros_like(spat_probs)
        classes[current_labels, torch.arange(probs.shape[1])] = 1

        current_labels = torch.argmax(tot_probs, 0).long().cpu().numpy()
        rot_labels = torch.argmax(rot_probs, 0).long().cpu().numpy()

        # points2bild(decoder_half1.model_positions, torch.ones(
        #    decoder_half1.model_positions.shape[0]), '/ceph/scheres_grp/schwab/iteration'+str(i), decoder_half1.box_size, decoder_half1.ang_pix, 5*current_labels)

    for i in range(number_of_masks):
        mask = torch.zeros(decoder_half1.box_size,
                           decoder_half1.box_size, decoder_half1.box_size)
        indices = torch.round((masked_points[i]+0.5)*(box_size-1)).long()
        mask[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

        filter_width = int(
            np.round(3*decoder_half1.mean_neighbour_distance.cpu() /
                     decoder_half1.ang_pix))
        print(filter_width)
        if filter_width % 2 == 0:
            filter_width += 1

        mean_filter = torch.ones(
            1, 1, filter_width, filter_width, filter_width)
        x = torch.arange(-filter_width//2+1, filter_width//2+1)
        X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
        R = torch.sqrt(X**2+Y**2+Z**2) <= filter_width//2
        mean_filter = mean_filter*R[None, None, :, :, :]
        mean_filter /= torch.sum(mean_filter)
        # print(mean_filter.shape)
        expanded_mask = torch.nn.functional.conv3d(
            mask[None, None, :, :, :].to(device).float(), mean_filter.to(device), padding=filter_width//2)
        mask = expanded_mask > 0.3 * torch.max(mean_filter)
        masks.append(mask[0, 0].movedim(0, 2).movedim(0, 1))

        with mrcfile.new(masks_directory / ('mask_body' + str(i+1) + '.mrc'), overwrite=True) as mrc:
            mrc.set_data(mask[0, 0].movedim(
                0, 2).movedim(0, 1).cpu().float().numpy())
            mrc.voxel_size = decoder_half1.ang_pix

    smooth_masks = []

    for m in masks:
        fmask = torch.fft.fftn(torch.fft.fftshift(
            m, dim=[-1, -2, -3]), dim=[-1, -2, -3])
        x = torch.fft.fftfreq(fmask.shape[-1])
        X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
        F = torch.exp(-(X**2+Y**2+Z**2)/2*mask_resolution **
                      2).to(decoder_half1.device)
        plt.imshow(F[125, :, :].cpu())
        plt.show()
        s_mask = torch.real(torch.fft.fftshift(torch.fft.ifftn(
            F*fmask, dim=[-1, -2, -3]), dim=[-1, -2, -3]))

        smooth_masks.append(s_mask)

    total_smooth_mask = torch.zeros_like(smooth_masks[0])
    for sm in smooth_masks:
        total_smooth_mask += sm
    i = 0
    for sm in smooth_masks:
        sm[sm > 0.01] /= total_smooth_mask[sm > 0.01]
        with mrcfile.new(masks_directory / ('smooth_mask_body' + str(i+1) + '.mrc'), overwrite=True) as mrc:
            mrc.set_data(sm.cpu().float().numpy())
            mrc.voxel_size = decoder_half1.ang_pix
        i += 1
