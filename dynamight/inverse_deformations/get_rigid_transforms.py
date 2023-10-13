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
from ..utils.utils_new import initialize_dataset, add_weight_decay_to_named_parameters
from ..data.dataloaders.relion import RelionDataset, abort_if_relion_abort, write_relion_job_exit_status
from ._optimize_single_epoch import optimize_epoch
from .rigid import get_rotation_translation
from tqdm import tqdm
from .._cli import cli
import numpy as np


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
    pipeline_control=None
):
    try:

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

        relion_dataset = RelionDataset(
            path=refinement_star_file,
            circular_mask_thickness=mask_soft_edge_width,
            particle_diameter=particle_diameter,
        )
        particle_dataset = relion_dataset.make_particle_dataset()
        diameter_ang = relion_dataset.particle_diameter
        box_size = relion_dataset.box_size
        ang_pix = relion_dataset.pixel_spacing_angstroms

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

        C = get_rotation_translation(
            encoder_half1, decoder_half1, data_loader_half1, poses, data_preprocessor, masked_points)

    except:
        if is_relion_abort(output_directory) == False:
            write_relion_job_exit_status(
                output_directory, 'FAILURE', pipeline_control)
