#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:45:02 2023

@author: schwab
"""

import torch
from torch.utils.data import DataLoader, Subset

from ..models.encoder import HetEncoder
from ..models.decoder import DisplacementDecoder
from ..data.handlers.particle_image_preprocessor import ParticleImagePreprocessor


def update_model_positions(
    dataset: torch.utils.data.Dataset,
    data_preprocessor: ParticleImagePreprocessor,
    encoder: HetEncoder,
    decoder: DisplacementDecoder,
    particle_shifts: torch.nn.Parameter,
    particle_euler_angles: torch.nn.Parameter,
    indices,
    consensus_update_pooled_particles,
    batch_size: int = 100,
):

    sub_data = torch.utils.data.Subset(dataset, indices)

    data_loader_sub = DataLoader(
        dataset=sub_data,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=False
    )
    device = decoder.device
    if decoder.warmup == False and decoder.mask != None:
        new_positions = torch.zeros(
            decoder.unmasked_positions.shape[0], decoder.unmasked_positions.shape[1]).to(device)
    else:
        new_positions = torch.zeros(
            decoder.model_positions.shape[0], decoder.model_positions.shape[1]).to(device)


    with torch.no_grad():
        for batch_ndx, sample in enumerate(data_loader_sub):
            r, y, ctf, shift = sample["rotation"], sample["image"], \
                sample["ctf"], - \
                sample['translation']
            idx = sample['idx'].to(device)
            r = particle_euler_angles[idx]
            shift = particle_shifts[idx]
            y, r, ctf, shift = y.to(device), r.to(
                device), ctf.to(device), shift.to(device)

            data_preprocessor.set_device(device)
            y_in = data_preprocessor.apply_square_mask(y)
            y_in = data_preprocessor.apply_translation(
                y_in.detach(), -shift[:, 0].detach(),
                -shift[:, 1].detach())
            y_in = data_preprocessor.apply_circular_mask(y_in)

            ctf = torch.fft.fftshift(ctf, dim=[-1, -2])

            mu, logsigma = encoder(y_in, ctf)
            #mu, logsigma, denois1, out1, target1 = encoder(y_in, ctf)

            z = mu + torch.exp(0.5 * logsigma) * torch.randn_like(mu)
            z_in = z
            Proj, new_points, deformed_points = decoder(
                z_in,
                r,
                shift.to(
                    device))
            new_positions += torch.sum(new_points, 0).detach()
        new_positions /= consensus_update_pooled_particles
        if decoder.warmup == False and decoder.mask != None:
            combined_pos = decoder.model_positions
            combined_pos[decoder.active_indices, :] = new_positions
            new_positions = combined_pos

        return new_positions
