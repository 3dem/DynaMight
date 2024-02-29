#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:34:51 2023

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
from ..utils.utils_new import initialize_dataset, add_weight_decay_to_named_parameters, maskpoints
from ..data.dataloaders.relion import RelionDataset, abort_if_relion_abort, write_relion_job_exit_status
from ._optimize_single_epoch import optimize_epoch
from .rigid import get_rotation_translation
from tqdm import tqdm
from .._cli import cli
import numpy as np
import mrcfile
import starfile


@cli.command()
def compute_rigid_transforms(
    output_directory: Path,
    refinement_star_file: Optional[Path] = None,
    checkpoint_file: Optional[Path] = None,
    batch_size: int = Option(100),
    gpu_id: Optional[int] = Option(0),
    particle_diameter: Optional[float] = Option(None),
    mask_soft_edge_width: int = Option(20),
    data_loader_threads: int = Option(4),
    pipeline_control=None,
    mask=None,
):
    forward_deformations_directory = output_directory / \
        'forward_deformations' / 'checkpoints'
    masks_directory = output_directory / 'masks'
    if not forward_deformations_directory.exists() and checkpoint_file is None:
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

    masked_points = []
    for file in sorted(os.listdir(masks_directory)):

        try:
            if file.startswith('mask'):
                print('reading in:', file)
                with mrcfile.open(masks_directory / file) as mrc:
                    mask = torch.tensor(mrc.data).to(device)
                    mask = mask.movedim(0, 2).movedim(0, 1)
                    m_points, inds = maskpoints(
                        decoder_half1.model_positions, decoder_half1.ampvar, mask, decoder_half1.box_size)
                    print(m_points.shape)
                    masked_points.append(m_points)
        except:
            print('No masks in this directory')
    print(len(masked_points))
    latent_dim = encoder_half1.latent_dim

    star_file = starfile.read(refinement_star_file)
    star_directory = output_directory / 'subsets'
    star_directory.mkdir(exist_ok=True, parents=True)

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
        shuffle=False,
        pin_memory=True
    )
    data_loader_half2 = DataLoader(
        dataset=dataset_half2,
        batch_size=batch_size,
        num_workers=data_loader_threads,
        shuffle=False,
        pin_memory=True
    )

    batch = next(iter(data_loader_half1))
    data_preprocessor = ParticleImagePreprocessor()
    data_preprocessor.initialize_from_stack(
        stack=batch["image"],
        circular_mask_radius=diameter_ang / (2 * ang_pix),
        circular_mask_thickness=mask_soft_edge_width / ang_pix
    )

    for i in range(len(masked_points)):
        current_star_file = star_file.copy()
        new_star = get_rotation_translation(
            encoder_half1, decoder_half1, data_loader_half1, poses, data_preprocessor, [masked_points[i]], [current_star_file], half=1)
        print(len(new_star))
        new_star = get_rotation_translation(
            encoder_half2, decoder_half2, data_loader_half2, poses, data_preprocessor, [masked_points[i]], new_star, half=2)
        starfile.write(new_star[0], star_directory /
                       ('body_' + str(i+1) + '.star'))
