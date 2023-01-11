#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 11:31:48 2021

@author: schwab
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional
from enum import Enum, auto

import numpy as np
import torch
import mrcfile
import typer

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torch_geometric.nn import knn_graph, radius_graph
from ..data.handlers.particle_image_preprocessor import \
    ParticleImagePreprocessor
from ..data.dataloaders.relion import RelionDataset
from ..data.handlers.io_logger import IOLogger
from ..models.consensus import ConsensusModel
from ..models.decoder import DisplacementDecoder
from ..models.encoder import HetEncoder
from ..models.blocks import LinearBlock
from ..models.pose import PoseModule
from ..models.utils import initialize_points_from_volume
from ..utils.utils_new import compute_threshold, initialize_consensus, \
    initialize_dataset, load_models, add_weight_decay, \
    fourier_loss, power_spec2, radial_avg2, geometric_loss, frc, \
    reset_all_linear_layer_weights, graph2bild, generate_form_factor, \
    prof2radim, visualize_latent, tensor_plot, tensor_imshow, tensor_scatter, \
    tensor_hist, apply_ctf, write_xyz, my_knn_graph, my_radius_graph, \
    calculate_grid_oversampling_factor, generate_data_normalization_mask
# from coarse_grain import optimize_coarsegraining

# TODO: add coarse graining to GitHub


from typer import Option, Typer

from .._cli import cli


class ConsensusInitializationMode(Enum):
    EMPTY = auto()
    MAP = auto()
    MODEL = auto()


@cli.command(no_args_is_help=True)
def optimize_deformations(
    refinement_star_file: Path = typer.Option(
        default=..., prompt=False, prompt_required=True
    ),
    output_directory: Path = typer.Option(
        default=..., prompt=False, prompt_required=True
    ),
    initial_model: Optional[Path] = None,
    initial_threshold: Optional[float] = None,
    initial_resolution: int = 8,
    mask_file: Optional[Path] = None,
    checkpoint_file: Optional[Path] = None,
    n_gaussians: int = 20000,
    n_gaussian_widths: int = 1,
    n_latent_dimensions: int = 2,
    n_positional_encoding_dimensions: int = 10,
    n_linear_layers: int = 8,
    n_neurons_per_layer: int = 32,
    n_warmup_epochs: int = 0,
    consensus_update_rate: float = 1,
    consensus_update_decay: float = 0.9,
    consensus_update_pooled_particles: int = 100,
    regularization_factor: float = 0.2,
    apply_bfactor: float = 0,
    particle_diameter: Optional[float] = None,
    soft_edge_width: float = 20,
    batch_size: int = 100,
    gpu_id: Optional[int] = 0,
    n_epochs: int = Option(200),
    n_threads: int = 8,
    preload_images: bool = True,
    n_workers: int = 8,
):
    # create directory structure
    deformations_directory = output_directory / 'forward_deformations'
    deformations_directory.mkdir(exist_ok=True, parents=True)

    # todo: implement
    #       add argument for free gaussians and mask
    #       add argument intial_resolution
    #       implement threshold computation
    #       implement bfactor in a useful way

    add_free_gaussians = 0

    # initialise logging
    summ = SummaryWriter(output_directory)
    sys.stdout = IOLogger(os.path.join(output_directory, 'std.out'))

    torch.set_num_threads(n_threads)

    # set consensus initialisation mode
    if initial_model == None:
        initialization_mode = ConsensusInitializationMode.EMPTY
    elif str(initial_model).endswith('.mrc'):
        initialization_mode = ConsensusInitializationMode.MAP
    else:
        initialization_mode = ConsensusInitializationMode.MODEL

    if gpu_id is None:
        typer.echo("Running on CPU")
    else:
        typer.echo(f"Training on GPU {gpu_id}")

    device = 'cpu' if gpu_id is None else 'cuda:' + str(gpu_id)

    typer.echo('Initializing the particle dataset')

    relion_dataset = RelionDataset(
        path=refinement_star_file,
        circular_mask_thickness=soft_edge_width,
        particle_diameter=particle_diameter,
    )
    particle_dataset = relion_dataset.make_particle_dataset()
    diameter_ang = relion_dataset.particle_diameter
    box_size = relion_dataset.box_size
    ang_pix = relion_dataset.pixel_spacing_angstroms

    print('Number of particles:', len(particle_dataset))

    # initialise poses and pose optimiser
    original_angles = particle_dataset.part_rotation.astype(np.float32)
    original_shifts = -particle_dataset.part_translation.astype(np.float32)
    angles = original_angles
    shifts = original_shifts

    angles = torch.nn.Parameter(torch.tensor(
        angles, requires_grad=True).to(device))
    angles_op = torch.optim.Adam([angles], lr=1e-3)
    shifts = torch.nn.Parameter(torch.tensor(
        shifts, requires_grad=True).to(device))
    shifts_op = torch.optim.Adam([shifts], lr=1e-3)

    # initialise training dataloaders
    if checkpoint_file is not None:  # get subsets from checkpoint if present
        cp = torch.load(checkpoint_file, map_location=device)
        inds_half1 = cp['indices_half1'].cpu().numpy()
        inds_half2 = list(
            set(range(len(particle_dataset))) - set(list(inds_half1)))
        dataset_half1 = torch.utils.data.Subset(particle_dataset, inds_half1)
        dataset_half2 = torch.utils.data.Subset(particle_dataset, inds_half2)
    else:
        dataset_half1, dataset_half2 = torch.utils.data.dataset.random_split(
            particle_dataset, [len(particle_dataset) // 2,
                               len(particle_dataset) - len(
                                   particle_dataset) // 2])

    data_loader_half1 = DataLoader(
        dataset=dataset_half1,
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=True,
        pin_memory=True
    )

    data_loader_half2 = DataLoader(
        dataset=dataset_half2,
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=True,
        pin_memory=True
    )

    batch = next(iter(data_loader_half1))

    # initialise preprocessor for particle images (masking)
    data_preprocessor = ParticleImagePreprocessor()
    data_preprocessor.initialize_from_stack(
        stack=batch["image"],
        circular_mask_radius=diameter_ang / (2 * ang_pix),
        circular_mask_thickness=soft_edge_width / ang_pix
    )
    data_preprocessor.set_device(device)
    print('Initialized data loaders for half sets of size',
          len(dataset_half1), ' and ', len(dataset_half2))

    print('box size:', box_size, 'pixel_size:', ang_pix, 'virtual pixel_size:',
          1 / (box_size + 1), ' dimension of latent space: ',
          n_latent_dimensions)
    latent_dim = n_latent_dimensions
    n_classes = n_gaussian_widths

    grid_oversampling_factor = calculate_grid_oversampling_factor(box_size)

    # initialise the gaussian model
    if initialization_mode == ConsensusInitializationMode.MODEL:
        mode = 'model'
        consensus_model, gr = optimize_coarsegraining(
            initial_model, box_size, ang_pix, device, str(output_directory),
            n_gaussian_widths, add_free_gaussians,
            initial_resolution=initial_resolution)
        consensus_model.amp.requires_grad = True
        if consensus_update_rate == None:
            consensus_update_rate = 0

    if initialization_mode in (ConsensusInitializationMode.MAP,
                               ConsensusInitializationMode.EMPTY):
        n_points = n_gaussians

    else:
        n_points = consensus_model.pos.shape[0]

    if initialization_mode == ConsensusInitializationMode.EMPTY:
        mode = 'density'

    if initialization_mode == ConsensusInitializationMode.MAP:
        mode = 'density'
        with mrcfile.open(initial_model) as mrc:
            Ivol = torch.tensor(mrc.data)
            if Ivol.shape[0] > 360:
                Ivols = torch.nn.functional.avg_pool3d(
                    Ivol[None, None], (2, 2, 2))
                Ivols = Ivols[0, 0]
            else:
                Ivols = Ivol
            if initial_threshold == None:
                initial_threshold = compute_threshold(Ivol)
            print('Setting threshold for the initialization to:', initial_threshold)
            initial_points = initialize_points_from_volume(
                Ivols.movedim(0, 2).movedim(0, 1),
                threshold=initial_threshold,
                n_points=n_points,
            )

    # define optimisation parameters

    pos_enc_dim = n_positional_encoding_dimensions

    LR = 0.001
    posLR = 0.001

    typer.echo(f'Number of used gaussians: {n_points}')

    decoder_kwargs = {
        'box_size': box_size,
        'device': device,
        'n_latent_dims': latent_dim,
        'n_points': n_points,
        'n_classes': n_classes,
        'n_layers': n_linear_layers,
        'n_neurons_per_layer': n_neurons_per_layer,
        'block': LinearBlock,
        'pos_enc_dim': pos_enc_dim,
        'grid_oversampling_factor': grid_oversampling_factor,
        'model_positions': initial_points

    }
    deformation_half1 = DisplacementDecoder(**decoder_kwargs).to(device)
    deformation_half2 = DisplacementDecoder(**decoder_kwargs).to(device)
    for decoder in (deformation_half1, deformation_half2):
        decoder.initialize_physical_parameters(reference_volume=Ivol)
    encoder_half1 = HetEncoder(box_size, latent_dim, 1).to(device)
    encoder_half2 = HetEncoder(box_size, latent_dim, 1).to(device)

    if mask_file:
        with mrcfile.open(mask_file) as mrc:
            mask = torch.tensor(mrc.data)
        mask = mask.movedim(0, 2).movedim(0, 1)

    if checkpoint_file:
        encoder_half1, encoder_half2, deformation_half1, deformation_half2 = load_models(
            checkpoint_file, device, box_size, n_classes)
    print('consensus model  initialization finished')
    if initialization_mode != ConsensusInitializationMode.MODEL.EMPTY:
        deformation_half1.model_positions.requires_grad = False
        deformation_half2.model_positions.requires_grad = False
    dec_half1_params = deformation_half1.parameters()
    dec_half2_params = deformation_half2.parameters()
    enc_half1_params = encoder_half1.parameters()
    enc_half2_params = encoder_half2.parameters()

    dec_half1_params = add_weight_decay(deformation_half1, weight_decay=1e-1)
    dec_half2_params = add_weight_decay(deformation_half2, weight_decay=1e-1)

    dec_half1_optimizer = torch.optim.Adam(dec_half1_params, lr=LR)
    dec_half2_optimizer = torch.optim.Adam(dec_half2_params, lr=LR)
    enc_half1_optimizer = torch.optim.Adam(enc_half1_params, lr=LR)
    enc_half2_optimizer = torch.optim.Adam(enc_half2_params, lr=LR)

    cons_optimizer_half1 = torch.optim.Adam(
        [deformation_half1.model_positions, deformation_half1.amp, deformation_half1.ampvar, deformation_half1.image_smoother.A, deformation_half1.image_smoother.B], lr=posLR)
    cons_optimizer_half2 = torch.optim.Adam(
        [deformation_half2.model_positions, deformation_half2.amp, deformation_half2.ampvar, deformation_half2.image_smoother.A, deformation_half2.image_smoother.B], lr=posLR)

    mean_dist = torch.zeros(n_points)
    mean_dist_half1 = torch.zeros(n_points)
    mean_dist_half2 = torch.zeros(n_points)

    BF = generate_data_normalization_mask(
        box_size, dampening_factor=apply_bfactor, device=device)

    '--------------------------------------------------------------------------------------------------------------------'
    'Start Training'
    '--------------------------------------------------------------------------------------------------------------------'

    kld_weight = batch_size / len(particle_dataset)
    beta = kld_weight * 0.0006

    distance = 0
    epoch_t = 0
    if n_warmup_epochs == None:
        n_warmup_epochs = 0

    old_loss_half1 = 1e8
    old_loss_half2 = 1e8
    FRC_im = torch.ones(box_size, box_size)

    with torch.no_grad():
        if initialization_mode == ConsensusInitializationMode.MODEL:
            pos_h1 = deformation_half1.model_positions * box_size * ang_pix
            pos_h2 = deformation_half2.model_positions * box_size * ang_pix
            gr1 = gr
            gr2 = gr
            cons_dis = torch.pow(
                torch.sum((pos_h1[gr1[0]] - pos_h1[gr1[1]]) ** 2, 1), 0.5)
            distance = 0
            deformation_half1.image_smoother.B.requires_grad = False
            deformation_half2.image_smoother.B.requires_grad = False
            consensus_model.i2F.B.requires_grad = False
        else:
            pos_h1 = deformation_half1.model_positions * box_size * ang_pix
            pos_h2 = deformation_half2.model_positions * box_size * ang_pix
            # grn = knn_graph(pos, 2, num_workers=8)
            grn_h1 = my_knn_graph(pos_h1, 2, workers=8)
            grn_h2 = my_knn_graph(pos_h2, 2, workers=8)
            mean_neighbour_dist_h1 = torch.mean(
                torch.pow(torch.sum((pos_h1[grn_h1[0]] - pos_h1[grn_h1[1]]) ** 2, 1), 0.5))
            mean_neighbour_dist_h2 = torch.mean(
                torch.pow(torch.sum((pos_h2[grn_h2[0]] - pos_h2[grn_h2[1]]) ** 2, 1), 0.5))
            print('mean distance in graph for half 1:', mean_neighbour_dist_h1,
                  ';This distance is also used to construct the initial graph ')
            print('mean distance in graph for half 2:', mean_neighbour_dist_h2,
                  ';This distance is also used to construct the initial graph ')
            distance_h1 = mean_neighbour_dist_h1
            distance_h2 = mean_neighbour_dist_h2

            gr_h1 = my_radius_graph(
                pos_h1, distance_h1 + distance_h1 / 2, workers=8)
            gr_h2 = my_radius_graph(
                pos_h2, distance_h2 + distance_h2 / 2, workers=8)
            gr1_h1 = my_radius_graph(deformation_half1.model_positions * ang_pix * box_size,
                                     distance_h1 + distance_h1 / 2, workers=8)
            gr1_h2 = my_radius_graph(deformation_half2.model_positions * ang_pix * box_size,
                                     distance_h2 + distance_h2 / 2, workers=8)

            gr2_h1 = my_knn_graph(
                deformation_half1.model_positions, 1, workers=8)
            gr2_h2 = my_knn_graph(
                deformation_half2.model_positions, 1, workers=8)
            cons_dis_h1 = torch.pow(
                torch.sum(
                    (pos_h1[gr1_h1[0]] - pos_h1[gr1_h1[1]]) ** 2, 1),
                0.5)
            cons_dis_h2 = torch.pow(
                torch.sum(
                    (pos_h2[gr1_h2[0]] - pos_h2[gr1_h2[1]]) ** 2, 1),
                0.5)

    tot_latent_dim = encoder_half1.latent_dim

    half1_indices = []
    K = 0

    with torch.no_grad():
        print('Computing half-set indices')
        for batch_ndx, sample in enumerate(data_loader_half1):
            idx = sample['idx']
            half1_indices.append(idx)
            if batch_ndx % batch_size == 0:
                print('Computing indices', batch_ndx / batch_size, 'of',
                      int(np.ceil(len(data_loader_half1) / batch_size)))

        half1_indices = torch.tensor(
            [item for sublist in half1_indices for item in sublist])
        cols = torch.ones(len(particle_dataset))
        cols[half1_indices] = 2

    for epoch in range(n_epochs):
        mean_positions_h1 = torch.zeros_like(deformation_half1.model_positions,
                                             requires_grad=False)
        mean_positions_h2 = torch.zeros_like(deformation_half2.model_positions,
                                             requires_grad=False)

        with torch.no_grad():
            if initialization_mode in (ConsensusInitializationMode.EMPTY,
                                       ConsensusInitializationMode.MAP):
                # gr1 = radius_graph(cons_model.pos*ang_pix *
                #                   box_size, distance+distance/2, num_workers=8)
                gr1_h1 = my_radius_graph(deformation_half1.model_positions * ang_pix *
                                         box_size, distance_h1 + distance_h1 / 2,
                                         workers=8)
                gr1_h2 = my_radius_graph(deformation_half2.model_positions * ang_pix *
                                         box_size, distance_h2 + distance_h2 / 2,
                                         workers=8)
                # gr2 = knn_graph(cons_model.pos, 2, num_workers=8)
                gr2_h1 = my_knn_graph(
                    deformation_half1.model_positions, 2, workers=8)
                gr2_h1 = my_knn_graph(
                    deformation_half2.model_positions, 2, workers=8)
                pos_h1 = deformation_half1.model_positions * box_size * ang_pix
                pos_h2 = deformation_half2.model_positions * box_size * ang_pix
                cons_dis_h1 = torch.pow(
                    torch.sum((pos_h1[gr1_h1[0]] - pos_h1[gr1_h1[1]]) ** 2, 1), 0.5)
                cons_dis_h2 = torch.pow(
                    torch.sum((pos_h2[gr1_h2[0]] - pos_h2[gr1_h2[1]]) ** 2, 1), 0.5)
            else:
                gr1 = gr
                gr2 = gr
                pos = deformation_half1.model_positions * box_size * ang_pix
                cons_dis = torch.pow(
                    torch.sum((pos[gr1[0]] - pos[gr1[1]]) ** 2, 1), 0.5)

        if epoch < n_warmup_epochs:
            N_graph_h1 = gr1_h1.shape[1]
            N_graph_h2 = gr1_h2.shape[1]
        else:
            N_graph_h1 = gr2_h1.shape[1]
            N_graph_h2 = gr2_h2.shape[1]
        try:
            gr_diff_h1 = gr1_h1.shape[1] - gr_old_h1.shape[1]
            gr_diff_h2 = gr1_h2.shape[1] - gr_old_h2.shape[1]
            if gr_diff_h1 < 0:
                print(torch.abs(gr_diff_h1),
                      'Gaussians removed to the neighbour graph in half 1')
            else:
                print(torch.abs(gr_diff_h1),
                      'Gaussians added to the neighbour graph in half1')
            if gr_diff_h2 < 0:
                print(torch.abs(gr_diff_h2),
                      'Gaussians removed to the neighbour graph in half 2')
            else:
                print(torch.abs(gr_diff_h2),
                      'Gaussians added to the neighbour graph in half 2')
        except:
            pass

        angles_op.zero_grad()
        shifts_op.zero_grad()
        if epoch > 0:
            print('Epoch:', epoch, 'Epoch time:', epoch_t)

        if epoch == n_warmup_epochs and initialization_mode != ConsensusInitializationMode.EMPTY:
            dec_half1_params = deformation_half1.parameters()
            dec_half2_params = deformation_half2.parameters()
            dec_half1_optimizer = torch.optim.Adam(dec_half1_params, lr=LR)
            dec_half2_optimizer = torch.optim.Adam(dec_half2_params, lr=LR)

        elif epoch == n_warmup_epochs and initialization_mode == ConsensusInitializationMode.EMPTY:
            dec_half1_params = deformation_half1.parameters()
            dec_half2_params = deformation_half2.parameters()
            dec_half1_optimizer = torch.optim.Adam(dec_half1_params, lr=LR)
            dec_half2_optimizer = torch.optim.Adam(dec_half2_params, lr=LR)

        running_recloss_half1 = 0
        running_latloss_half1 = 0
        running_total_loss_half1 = 0
        var_total_loss_half1 = 0

        running_recloss_half2 = 0
        running_latloss_half2 = 0
        running_total_loss_half2 = 0
        var_total_loss_half2 = 0

        start_t = time.time()

        latent_space = np.zeros([len(particle_dataset), tot_latent_dim])
        diff = np.zeros([len(particle_dataset), 1])

        mean_dist_h1 = torch.zeros(deformation_half1.n_points)
        mean_dist_h2 = torch.zeros(deformation_half2.n_points)
        displacement_variance_h1 = torch.zeros_like(mean_dist_h1)
        displacement_variance_h2 = torch.zeros_like(mean_dist_h2)

        calibration_data = torch.utils.data.Subset(
            dataset_half1, torch.randint(0, len(dataset_half1), (1000,)))
        data_loader_cal = DataLoader(
            dataset=calibration_data,
            batch_size=batch_size,
            num_workers=8,
            shuffle=True,
            pin_memory=False
        )

        dec_norm_tot = 0
        print('calibrating loss parameters and data profile')
        for batch_ndx, sample in enumerate(data_loader_cal):
            deformation_half1.zero_grad()
            encoder_half1.zero_grad()
            r, y, ctf = sample["rotation"], sample["image"], sample["ctf"]
            idx = sample['idx']
            r = angles[idx]
            shift = shifts[idx]

            y, r, ctf, shift = y.to(device), r.to(
                device), ctf.to(device), shift.to(device)

            data_preprocessor.set_device(device)
            y_in = data_preprocessor.apply_square_mask(y)
            y_in = data_preprocessor.apply_translation(
                y_in.detach(), -shift[:, 0].detach(), -shift[:, 1].detach())
            y_in = data_preprocessor.apply_circular_mask(y_in)

            ctf = torch.fft.fftshift(ctf, dim=[-1, -2])

            mu, logsigma = encoder_half1(y_in, ctf)
            z = mu + torch.exp(0.5 * logsigma) * torch.randn_like(mu)
            z_in = z

            if epoch < n_warmup_epochs:
                # Set latent code for consensus reconstruction to zero
                Proj, P, PP, new_points = consensus_model(r, shift)
            else:
                Proj, new_points, deformed_points = deformation_half1(
                    z_in,
                    r,
                    shift.to(device)
                )

            y = sample["image"].to(device)
            y = data_preprocessor.apply_circular_mask(y.detach())
            rec_loss = fourier_loss(
                Proj.squeeze(), y.squeeze(), ctf.float(), W=BF[None, :, :])

            rec_loss.backward()
            with torch.no_grad():
                try:
                    total_norm = 0

                    for p in deformation_half1.parameters():
                        if p.requires_grad == True:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                    dec_norm_tot += total_norm ** 0.5
                except:
                    dec_norm_tot = 1

        data_norm = dec_norm_tot
        dec_norm_tot = 0

        with torch.no_grad():
            x = Proj
            yd = torch.fft.fft2(y, dim=[-1, -2], norm='ortho')
            x = torch.multiply(x, ctf)
            model_spec, m_s = power_spec2(x, batch_reduce='mean')
            data_spec, d_s = power_spec2(yd, batch_reduce='mean')
            model_avg, m_a = radial_avg2(x, batch_reduce='mean')
            data_avg, d_a = radial_avg2(BF[None, :, :] * yd,
                                        batch_reduce='mean')
            data_var, d_v = radial_avg2((data_avg - yd) ** 2,
                                        batch_reduce='mean')
            model_var, m_v = radial_avg2((model_avg - x) ** 2,
                                         batch_reduce='mean')
            err_avg, e_a = radial_avg2(
                (x - BF[None, :, :] * yd) ** 2, batch_reduce='mean')
            err_im = (x - yd) ** 2
            err_im_real = torch.real(torch.fft.ifft2(err_im))

        for batch_ndx, sample in enumerate(data_loader_cal):
            encoder_half1.zero_grad()
            deformation_half1.zero_grad()
            angles_op.zero_grad()
            shifts_op.zero_grad()
            r, y, ctf = sample["rotation"], sample["image"], sample["ctf"]
            idx = sample['idx']
            r = angles[idx]
            shift = shifts[idx]

            y, r, ctf, shift = y.to(device), r.to(
                device), ctf.to(device), shift.to(device)

            data_preprocessor.set_device(device)
            y_in = data_preprocessor.apply_square_mask(y)
            y_in = data_preprocessor.apply_translation(
                y_in.detach(), -shift[:, 0].detach(), -shift[:, 1].detach())
            y_in = data_preprocessor.apply_circular_mask(y_in)
            # y_in = y

            ctf = torch.fft.fftshift(ctf, dim=[-1, -2])

            mu, logsigma = encoder_half1(y_in, ctf)
            z = mu + torch.exp(0.5 * logsigma) * torch.randn_like(mu)
            z_in = z

            if epoch < n_warmup_epochs:  # Set latent code for consensus reconstruction to zero
                Proj, P, PP, new_points = consensus_model(r, shift)
                deformed_points = torch.zeros_like(new_points)
            else:
                Proj, new_points, deformed_points = deformation_half1(
                    z_in,
                    r,
                    shift.to(
                        device))

            y = sample["image"].to(device)
            y = data_preprocessor.apply_circular_mask(y.detach())
            geo_loss = geometric_loss(new_points, box_size, ang_pix, distance_h1,
                                      deformation=cons_dis_h1, graph1=gr1_h1,
                                      graph2=gr2_h1, mode=mode)
            try:
                geo_loss.backward()
            except:
                pass
            with torch.no_grad():
                try:
                    total_norm = 0

                    for p in deformation_half1.parameters():
                        if p.requires_grad == True:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                        dec_norm_tot += total_norm ** 0.5
                except:
                    dec_norm_tot = 1
            prior_norm = dec_norm_tot
        if epoch == 0:
            lam_reg = 1
        else:
            lam_reg = regularization_factor * \
                (0.1 * 0.5 * data_norm / prior_norm + 0.9 * lam_reg)
        # lam_reg = 0
        print('new regularization parameter:', lam_reg)

        for batch_ndx, sample in enumerate(data_loader_half1):
            if batch_ndx % 100 == 0:
                print('Processing batch', batch_ndx / batch_size, 'of',
                      int(np.ceil(len(data_loader_half1) / batch_size)),
                      ' from half 1')

            enc_half1_optimizer.zero_grad()
            dec_half1_optimizer.zero_grad()
            cons_optimizer_half1.zero_grad()

            r, y, ctf = sample["rotation"], sample["image"], sample["ctf"]
            idx = sample['idx']
            r = angles[idx]
            shift = shifts[idx]

            y, r, ctf, shift = y.to(device), r.to(
                device), ctf.to(device), shift.to(device)

            data_preprocessor.set_device(device)
            y_in = data_preprocessor.apply_square_mask(y)
            y_in = data_preprocessor.apply_translation(
                y_in.detach(), -shift[:, 0].detach(), -shift[:, 1].detach())
            y_in = data_preprocessor.apply_circular_mask(y_in)
            # y_in = y

            ctf = torch.fft.fftshift(ctf, dim=[-1, -2])

            mu, logsigma = encoder_half1(y_in, ctf)
            z = mu + torch.exp(0.5 * logsigma) * torch.randn_like(mu)
            z_in = z

            if epoch < n_warmup_epochs:  # Set latent code for consensus reconstruction to zero
                Proj,  new_points, deformed_points = deformation_half1(
                    z_in,
                    r,
                    shift.to(
                        device))
            else:
                Proj, new_points, deformed_points = deformation_half1(
                    z_in,
                    r,
                    shift.to(
                        device))

                with torch.no_grad():
                    try:
                        frc_half1 += frc(Proj, y, ctf)
                    except:
                        frc_half1 = frc(Proj, y, ctf)
                    mu2, logsigma2 = encoder_half2(y_in, ctf)
                    z_in2 = mu2
                    _, _, d_points2 = deformation_half2(z_in2, r,
                                                        shift.to(device))
                    mean_positions_h1 += torch.sum(new_points.detach(), 0)

                    diff[sample["idx"].cpu().numpy()] = torch.mean(
                        torch.sum((deformed_points - d_points2) ** 2, 2),
                        1).unsqueeze(
                        1).detach().cpu()

            displacement_variance_h1 += torch.sum(
                torch.linalg.norm(deformed_points.detach().cpu(), dim=2) ** 2,
                0)
            y = sample["image"].to(device)
            y = data_preprocessor.apply_circular_mask(y.detach())
            rec_loss = fourier_loss(
                Proj.squeeze(), y.squeeze(), ctf.float(), W=BF[None, :, :])
            latent_loss = -0.5 * \
                torch.mean(torch.sum(1 + logsigma - mu ** 2 -
                                     torch.exp(logsigma), dim=1),
                           dim=0)

            st = time.time()

            if epoch < n_warmup_epochs:  # and cons_model.n_points<args.n_gauss:
                geo_loss = torch.zeros(1).to(device)

            else:
                encoder_half1.requires_grad = True
                geo_loss = geometric_loss(
                    new_points, box_size, ang_pix, distance_h1,
                    deformation=cons_dis_h1,
                    graph1=gr1_h1, graph2=gr2_h1, mode=mode)

            if epoch < n_warmup_epochs:
                loss = rec_loss + beta * kld_weight * latent_loss
            else:
                loss = rec_loss + beta * kld_weight * latent_loss + lam_reg * geo_loss

            loss.backward()
            if epoch < n_warmup_epochs:
                cons_optimizer_half1.step()

            else:
                encoder_half1.requires_grad = True
                deformation_half1.requires_grad = True
                deformation_half2.requires_grad = True
                enc_half1_optimizer.step()
                dec_half1_optimizer.step()
                cons_optimizer_half1.step()

            eval_t = time.time() - st
            latent_space[sample["idx"].cpu().numpy()] = mu.detach().cpu()
            running_recloss_half1 += rec_loss.item()
            running_latloss_half1 += latent_loss.item()
            running_total_loss_half1 += loss.item()
            var_total_loss_half1 += geo_loss.item()

        for batch_ndx, sample in enumerate(data_loader_half2):
            if batch_ndx % 100 == 0:
                print('Processing batch', batch_ndx / batch_size, 'of',
                      int(np.ceil(len(data_loader_half2) / batch_size)),
                      'from half 2')
            enc_half2_optimizer.zero_grad()
            dec_half2_optimizer.zero_grad()
            cons_optimizer_half2.zero_grad()

            r, y, ctf = sample["rotation"], sample["image"], sample["ctf"]
            idx = sample['idx']
            r = angles[idx]
            shift = shifts[idx]
            y, r, ctf, shift = y.to(device), r.to(
                device), ctf.to(device), shift.to(device)

            data_preprocessor.set_device(device)
            y_in = data_preprocessor.apply_square_mask(y)
            y_in = data_preprocessor.apply_translation(
                y_in.detach(), -shift[:, 0].detach(), -shift[:, 1].detach())
            y_in = data_preprocessor.apply_circular_mask(y_in)

            ctf = torch.fft.fftshift(ctf, dim=[-1, -2])

            mu, logsigma = encoder_half2(y_in, ctf)

            z = mu + torch.exp(0.5 * logsigma) * torch.randn_like(mu)

            z_in = z

            if epoch < n_warmup_epochs:  # Set latent code for consensus reconstruction to zero
                Proj, new_points, deformed_points = deformation_half2(
                    z_in, r,
                    shift.to(
                        device))
            else:
                Proj, new_points, deformed_points = deformation_half2(
                    z_in, r,
                    shift.to(
                        device))

                with torch.no_grad():
                    try:
                        frc_half2 += frc(Proj, y, ctf)
                    except:
                        frc_half2 = frc(Proj, y, ctf)
                    mu2, logsigma2 = encoder_half1(y_in, ctf)
                    z_in2 = mu2
                    _, _, d_points2 = deformation_half1(z_in2, r,
                                                        shift.to(device))
                    mean_positions_h2 += torch.sum(new_points.detach(), 0)

                    diff[sample["idx"].cpu().numpy()] = torch.mean(
                        (deformed_points - d_points2) ** 2).detach().cpu()

            displacement_variance_h2 += torch.sum(
                torch.linalg.norm(deformed_points.detach().cpu(), dim=2) ** 2,
                0)
            y = sample["image"].to(device)
            y = data_preprocessor.apply_circular_mask(y.detach())
            rec_loss = fourier_loss(
                Proj.squeeze(), y.squeeze(), ctf.float(), W=BF[None, :, :])
            latent_loss = -0.5 * \
                torch.mean(torch.sum(1 + logsigma - mu ** 2 -
                                     torch.exp(logsigma), dim=1),
                           dim=0)

            st = time.time()

            if epoch < n_warmup_epochs:  # and cons_model.n_points<args.n_gauss:
                geo_loss = torch.zeros(1).to(device)

            else:
                encoder_half2.requires_grad = True
                geo_loss = geometric_loss(
                    new_points, box_size, ang_pix, distance_h2,
                    deformation=cons_dis_h2,
                    graph1=gr1_h2, graph2=gr2_h2, mode=mode)

            if epoch < n_warmup_epochs:
                loss = rec_loss + beta * kld_weight * latent_loss
            else:
                loss = rec_loss + beta * kld_weight * latent_loss + lam_reg * geo_loss

            loss.backward()
            if epoch < n_warmup_epochs:
                cons_optimizer_half2.step()

            else:
                encoder_half2.requires_grad = True
                deformation_half1.requires_grad = True
                deformation_half2.requires_grad = True
                enc_half2_optimizer.step()
                dec_half2_optimizer.step()
                cons_optimizer_half2.step()

            eval_t = time.time() - st
            latent_space[sample["idx"].cpu().numpy()] = mu.detach().cpu()
            running_recloss_half2 += rec_loss.item()
            running_latloss_half2 += latent_loss.item()
            running_total_loss_half2 += loss.item()
            var_total_loss_half2 += geo_loss.item()

            with torch.no_grad():
                mean_dist_half1 += torch.sum(
                    torch.linalg.norm(deformed_points, dim=2), 0).cpu()
                mean_dist_half2 += torch.sum(
                    torch.linalg.norm(d_points2, dim=2), 0).cpu()

        angles_op.step()
        shifts_op.step()

        current_angles = angles.detach().cpu().numpy()
        angular_error = np.mean(np.square(current_angles - original_angles))

        current_shifts = shifts.detach().cpu().numpy()
        translational_error = np.mean(
            np.square(current_shifts - original_shifts))

        poses = PoseModule(box_size, device, torch.tensor(
            current_angles), torch.tensor(current_shifts))
        try:
            gr_old = gr1
        except:
            pass

        if epoch > (n_warmup_epochs - 1) and consensus_update_rate != 0:
            mean_positions_h1 /= len(dataset_half1)
            mean_positions_h2 /= len(dataset_half2)

            # update half1
            dl = data_loader_half1
            print('Update consensus model')
            with torch.no_grad():
                for batch_ndx, sample in enumerate(dl):
                    r, y, ctf, shift = sample["rotation"], sample["image"], \
                        sample["ctf"], - \
                        sample['translation']
                    idx = sample['idx'].to(device)
                    r = angles[idx]
                    shift = shifts[idx]
                    y, r, ctf, shift = y.to(device), r.to(
                        device), ctf.to(device), shift.to(device)

                    data_preprocessor.set_device(device)
                    y_in = data_preprocessor.apply_square_mask(y)
                    y_in = data_preprocessor.apply_translation(
                        y_in.detach(), -shift[:, 0].detach(),
                        -shift[:, 1].detach())
                    y_in = data_preprocessor.apply_circular_mask(y_in)

                    ctf = torch.fft.fftshift(ctf, dim=[-1, -2])

                    mu, logsigma = encoder_half1(y_in, ctf)

                    z = mu + torch.exp(0.5 * logsigma) * torch.randn_like(mu)
                    z_in = z
                    Proj,  new_points, deformed_points = deformation_half1(
                        z_in,
                        r,
                        shift.to(
                            device))
                    defs = torch.linalg.norm(
                        torch.linalg.norm(deformed_points, dim=2), dim=1)
                    if batch_ndx == 0:
                        lat = mu
                        dis_norm = defs
                        idix = idx
                    else:
                        lat = torch.cat([lat, mu])
                        dis_norm = torch.cat([dis_norm, defs])
                        idix = torch.cat([idix, idx])

                _, bottom_ind = torch.topk(
                    dis_norm, consensus_update_pooled_particles, largest=False)
                min_indix = idix[bottom_ind].cpu()

            sub_data = torch.utils.data.Subset(particle_dataset, min_indix)
            data_loader_sub = DataLoader(
                dataset=sub_data,
                batch_size=batch_size,
                num_workers=8,
                shuffle=True,
                pin_memory=False
            )
            new_pos_h1 = torch.zeros(
                new_points.shape[1], new_points.shape[2]).to(device)
            with torch.no_grad():
                for batch_ndx, sample in enumerate(data_loader_sub):
                    r, y, ctf, shift = sample["rotation"], sample["image"], \
                        sample["ctf"], - \
                        sample['translation']
                    idx = sample['idx'].to(device)
                    r = angles[idx]
                    shift = shifts[idx]
                    y, r, ctf, shift = y.to(device), r.to(
                        device), ctf.to(device), shift.to(device)

                    data_preprocessor.set_device(device)
                    y_in = data_preprocessor.apply_square_mask(y)
                    y_in = data_preprocessor.apply_translation(
                        y_in.detach(), -shift[:, 0].detach(),
                        -shift[:, 1].detach())
                    y_in = data_preprocessor.apply_circular_mask(y_in)

                    ctf = torch.fft.fftshift(ctf, dim=[-1, -2])

                    mu, logsigma = encoder_half1(y_in, ctf)

                    z = mu + torch.exp(0.5 * logsigma) * torch.randn_like(mu)
                    z_in = z
                    Proj, new_points, deformed_points = deformation_half1(
                        z_in,
                        r,
                        shift.to(
                            device))
                    new_pos_h1 += torch.sum(new_points, 0).detach()

                new_pos_h1 /= consensus_update_pooled_particles

                # update half2
                dl = data_loader_half2
                print('Update consensus model')
                with torch.no_grad():
                    for batch_ndx, sample in enumerate(dl):
                        r, y, ctf, shift = sample["rotation"], sample["image"], \
                            sample["ctf"], - \
                            sample['translation']
                        idx = sample['idx'].to(device)
                        r = angles[idx]
                        shift = shifts[idx]
                        y, r, ctf, shift = y.to(device), r.to(
                            device), ctf.to(device), shift.to(device)

                        data_preprocessor.set_device(device)
                        y_in = data_preprocessor.apply_square_mask(y)
                        y_in = data_preprocessor.apply_translation(
                            y_in.detach(), -shift[:, 0].detach(),
                            -shift[:, 1].detach())
                        y_in = data_preprocessor.apply_circular_mask(y_in)

                        ctf = torch.fft.fftshift(ctf, dim=[-1, -2])

                        mu, logsigma = encoder_half2(y_in, ctf)

                        z = mu + torch.exp(0.5 * logsigma) * \
                            torch.randn_like(mu)
                        z_in = z
                        Proj, new_points, deformed_points = deformation_half2(
                            z_in,
                            r,
                            shift.to(
                                device))
                        defs = torch.linalg.norm(
                            torch.linalg.norm(deformed_points, dim=2), dim=1)
                        if batch_ndx == 0:
                            lat = mu
                            dis_norm = defs
                            idix = idx
                        else:
                            lat = torch.cat([lat, mu])
                            dis_norm = torch.cat([dis_norm, defs])
                            idix = torch.cat([idix, idx])

                    _, bottom_ind = torch.topk(
                        dis_norm, consensus_update_pooled_particles, largest=False)
                    min_indix = idix[bottom_ind].cpu()

                sub_data = torch.utils.data.Subset(particle_dataset, min_indix)
                data_loader_sub = DataLoader(
                    dataset=sub_data,
                    batch_size=batch_size,
                    num_workers=8,
                    shuffle=True,
                    pin_memory=False
                )
                new_pos_h2 = torch.zeros(
                    new_points.shape[1], new_points.shape[2]).to(device)
                with torch.no_grad():
                    for batch_ndx, sample in enumerate(data_loader_sub):
                        r, y, ctf, shift = sample["rotation"], sample["image"], \
                            sample["ctf"], - \
                            sample['translation']
                        idx = sample['idx'].to(device)
                        r = angles[idx]
                        shift = shifts[idx]
                        y, r, ctf, shift = y.to(device), r.to(
                            device), ctf.to(device), shift.to(device)

                        data_preprocessor.set_device(device)
                        y_in = data_preprocessor.apply_square_mask(y)
                        y_in = data_preprocessor.apply_translation(
                            y_in.detach(), -shift[:, 0].detach(),
                            -shift[:, 1].detach())
                        y_in = data_preprocessor.apply_circular_mask(y_in)

                        ctf = torch.fft.fftshift(ctf, dim=[-1, -2])

                        mu, logsigma = encoder_half1(y_in, ctf)

                        z = mu + torch.exp(0.5 * logsigma) * \
                            torch.randn_like(mu)
                        z_in = z
                        Proj, new_points, deformed_points = deformation_half1(
                            z_in,
                            r,
                            shift.to(
                                device))
                        new_pos_h2 += torch.sum(new_points, 0).detach()

                    new_pos_h2 /= consensus_update_pooled_particles

            if (
                    running_recloss_half1 < old_loss_half1 and running_recloss_half2 < old_loss_half2) and consensus_update_rate != 0:
                # cons_model.pos = torch.nn.Parameter((1-consensus_update_rate)*cons_model.pos+consensus_update_rate*min_npos,requires_grad=False)
                deformation_half1.model_positions = torch.nn.Parameter(
                    (
                        1 - consensus_update_rate) * deformation_half1.model_positions + consensus_update_rate * new_pos_h1,
                    requires_grad=False)
                deformation_half2.model_positions = torch.nn.Parameter(
                    (
                        1 - consensus_update_rate) * deformation_half2.model_positions + consensus_update_rate * new_pos_h2,
                    requires_grad=False)
                old_loss_half1 = running_recloss_half1
                old_loss_half2 = running_recloss_half2
                nosub_ind = 0
            if running_recloss_half1 > old_loss_half1 and running_recloss_half2 > old_loss_half2:
                nosub_ind += 1
                print('No consensus updates for ', nosub_ind, ' epochs')
                if nosub_ind == 1:
                    consensus_update_rate *= 0.8
                if consensus_update_rate < 0.1:
                    consensus_update_rate = 0
                    reset_all_linear_layer_weights(deformation_half1)
                    reset_all_linear_layer_weights(deformation_half2)
                    reset_all_linear_layer_weights(encoder_half1)
                    reset_all_linear_layer_weights(encoder_half2)
                    regularization_factor = 1
                    mode = 'model'

        with torch.no_grad():
            frc_half1 /= len(dataset_half1)
            frc_half2 /= len(dataset_half2)
            frc_both = (frc_half1 + frc_half2) / 2
            frc_both = frc_both[:box_size // 2]
            pos_h1 = deformation_half1.model_positions * box_size * ang_pix
            pos_h2 = deformation_half2.model_positions * box_size * ang_pix
            # grn = knn_graph(pos, 2, num_workers=8)
            grn_h1 = my_knn_graph(pos_h1, 2, workers=8)
            grn_h2 = my_knn_graph(pos_h2, 2, workers=8)
            N_graph_h1 = grn_h1.shape[1]
            N_graph_h2 = grn_h2.shape[1]
            mean_neighbour_dist_h1 = torch.mean(
                torch.pow(torch.sum((pos_h1[grn_h1[0]] - pos_h1[grn_h1[1]]) ** 2, 1), 0.5))
            mean_neighbour_dist_h2 = torch.mean(
                torch.pow(torch.sum((pos_h2[grn_h2[0]] - pos_h2[grn_h2[1]]) ** 2, 1), 0.5))
            print('mean distance in graph in Angstrom in half 1:',
                  mean_neighbour_dist_h1)
            print('mean distance in graph in Angstrom in half 2:',
                  mean_neighbour_dist_h2)
            distance_h1 = mean_neighbour_dist_h1
            distance_h2 = mean_neighbour_dist_h2
            displacement_variance_h1 /= len(dataset_half1)
            displacement_variance_h2 /= len(dataset_half2)
            D_var = torch.stack(
                [displacement_variance_h1, displacement_variance_h2], 1)

            if initialization_mode == ConsensusInitializationMode.EMPTY:
                gr_h1 = my_radius_graph(
                    pos_h1, distance_h1 + distance_h1 / 2, workers=8)
                gr_h2 = my_radius_graph(
                    pos_h3, distance_h2 + distance_h2 / 2, workers=8)

            ff2 = generate_form_factor(
                deformation_half1.image_smoother.A, deformation_half1.image_smoother.B, box_size)
            ff2b = generate_form_factor(
                deformation_half2.image_smoother.A, deformation_half2.image_smoother.B, box_size)
            FF = np.concatenate([ff2, ff2b], 1)

            ind1 = torch.randint(0, box_size - 1, (1, 1))
            ind2 = torch.randint(0, box_size - 1, (1, 1))
            err_pix = err_im[:, ind1, ind2]

            x = Proj[0]
            yd = y[0]
            if x.is_complex():
                pass
            else:
                x = torch.fft.fft2(x, dim=[-2, -1], norm='ortho')

            if tot_latent_dim > 2:
                if epoch % 5 == 0 and epoch > n_warmup_epochs:
                    summ.add_figure("Data/latent",
                                    visualize_latent(latent_space, c=diff / (
                                        np.max(diff) + 1e-7), s=3,
                                        alpha=0.2, method='pca'),
                                    epoch)
                    summ.add_figure(
                        "Data/latent2",
                        visualize_latent(latent_space, c=cols, s=3, alpha=0.2,
                                         method='pca'))

            else:
                summ.add_figure("Data/latent",
                                visualize_latent(
                                    latent_space,
                                    c=diff / (np.max(diff) + 1e-22), s=3,
                                    alpha=0.2),
                                epoch)
                summ.add_figure("Data/latent2",
                                visualize_latent(
                                    latent_space, c=cols, s=3, alpha=0.2),
                                epoch)
            summ.add_scalar("Loss/kld_loss",
                            (running_latloss_half1 + running_latloss_half2) / (
                                len(data_loader_half1) + len(
                                    data_loader_half2)), epoch)
            summ.add_scalar("Loss/mse_loss",
                            (running_recloss_half1 + running_recloss_half2) / (
                                len(data_loader_half1) + len(
                                    data_loader_half2)), epoch)
            summ.add_scalars("Loss/mse_loss_halfs",
                             {'half1': (running_recloss_half1) / (len(
                                 data_loader_half1)),
                              'half2': (running_recloss_half2) / (
                                  len(data_loader_half2))}, epoch)
            summ.add_scalar("Loss/total_loss", (
                running_total_loss_half1 + running_total_loss_half2) / (
                len(data_loader_half1) + len(
                    data_loader_half2)), epoch)
            summ.add_scalar("Loss/dist_loss",
                            (var_total_loss_half1 + var_total_loss_half2) / (
                                len(data_loader_half1) + len(
                                    data_loader_half2)), epoch)

            summ.add_scalar(
                "Loss/variance1a",
                deformation_half1.image_smoother.B[0].detach().cpu(), epoch)
            summ.add_scalar(
                "Loss/variance2a",
                deformation_half2.image_smoother.B[0].detach().cpu(), epoch)

            summ.add_figure("Data/cons_amp_h1",
                            tensor_plot(deformation_half1.amp.detach()), epoch)
            summ.add_figure("Data/cons_amp_h2",
                            tensor_plot(deformation_half2.amp.detach()), epoch)

            summ.add_scalar("Loss/N_graph_h1", N_graph_h1, epoch)
            summ.add_scalar("Loss/N_graph_h2", N_graph_h2, epoch)
            summ.add_scalar("Loss/reg_param", lam_reg, epoch)
            summ.add_scalar("Loss/substitute", consensus_update_rate, epoch)
            summ.add_scalar("Loss/pose_error", angular_error, epoch)
            summ.add_scalar("Loss/trans_error", translational_error, epoch)
            summ.add_figure("Data/output", tensor_imshow(torch.fft.fftshift(
                apply_ctf(Proj[0], ctf[0].float()).squeeze().cpu(),
                dim=[-1, -2])), epoch)
            summ.add_figure(
                "Data/input", tensor_imshow(y_in[0].squeeze().detach().cpu()),
                epoch)
            summ.add_figure("Data/target", tensor_imshow(torch.fft.fftshift(
                apply_ctf(y[0], BF.float()).squeeze().cpu(), dim=[-1, -2])),
                epoch)
            summ.add_figure("Data/cons_points_z_half1",
                            tensor_scatter(deformation_half1.model_positions[:, 0],
                                           deformation_half1.model_positions[:, 1],
                                           c=mean_dist_half1, s=3), epoch)
            summ.add_figure("Data/cons_points_z_half2",
                            tensor_scatter(deformation_half2.model_positions[:, 0],
                                           deformation_half2.model_positions[:, 1],
                                           c=mean_dist_half2, s=3), epoch)
            summ.add_figure(
                "Data/delta",
                tensor_scatter(new_points[0, :, 0], new_points[0, :, 1], 'b',
                               s=0.1), epoch)
            summ.add_figure(
                "Data/delta_h1",
                tensor_scatter(d_points2[0, :, 0], d_points2[0, :, 1], 'b',
                               s=0.1), epoch)
            # summ.add_figure(f"Data/min_n_pos", tensor_scatter(min_npos[:, 0], min_npos[:, 1], 'b', s=0.1), epoch)

            summ.add_figure("Data/projection_image",
                            tensor_imshow(torch.fft.fftshift(torch.real(
                                torch.fft.ifftn(Proj[0], dim=[-1,
                                                              -2])).squeeze().detach().cpu(),
                                dim=[-1, -2])),
                            epoch)
            summ.add_figure("Data/mod_model", tensor_imshow(apply_ctf(apply_ctf(
                y[0], ctf[0].float()),
                FRC_im.to(device)).squeeze().detach().cpu()), epoch)
            summ.add_figure("Data/shapes", tensor_plot(FF), epoch)
            summ.add_figure("Data/dis_var", tensor_plot(D_var), epoch)
            summ.add_figure("Data/model_spec", tensor_plot(m_a), epoch)
            summ.add_figure("Data/data_spec", tensor_plot(d_a), epoch)
            summ.add_figure("Data/err_prof", tensor_plot(e_a), epoch)
            summ.add_figure(
                "Data/avg_err",
                tensor_imshow(torch.fft.fftshift(err_avg, dim=[-1, -2]).cpu()),
                epoch)
            summ.add_figure(
                "Data/err_im", tensor_imshow(err_im_real[0].cpu()), epoch)
            summ.add_figure("Data/err_hist_r",
                            tensor_hist(torch.real(err_pix).cpu(), 40), epoch)
            summ.add_figure("Data/err_hist_c",
                            tensor_hist(torch.imag(err_pix).cpu(), 40), epoch)
            summ.add_figure("Data/frc_h1", tensor_plot(frc_half1), epoch)
            summ.add_figure("Data/frc_h2", tensor_plot(frc_half2), epoch)

            frc_half1 = torch.zeros_like(frc_half1)
            frc_half2 = torch.zeros_like(frc_half2)

        epoch_t = time.time() - start_t

        if epoch % 10 == 0 or epoch == (n_epochs - 1):
            with torch.no_grad():
                r0 = torch.zeros(2, 3)
                t0 = torch.zeros(2, 2)
                V_h1 = deformation_half1.generate_consensus_volume().cpu()
                V_h2 = deformation_half2.generate_consensus_volume().cpu()
                gaussian_widths = torch.argmax(torch.nn.functional.softmax(
                    deformation_half1.ampvar, dim=0), dim=0)
                checkpoint = {'encoder_half1': encoder_half1,
                              'encoder_half2': encoder_half2,
                              'decoder_half1': deformation_half1,
                              'decoder_half2': deformation_half2,
                              'poses': poses,
                              'encoder_half1_state_dict': encoder_half1.state_dict(),
                              'encoder_half2_state_dict': encoder_half2.state_dict(),
                              'decoder_half1_state_dict': deformation_half1.state_dict(),
                              'decoder_half2_state_dict': deformation_half2.state_dict(),
                              'poses_state_dict': poses.state_dict(),
                              'enc_half1_optimizer': enc_half1_optimizer.state_dict(),
                              'enc_half2_optimizer': enc_half2_optimizer.state_dict(),
                              'cons_optimizer_half1': cons_optimizer_half1.state_dict(),
                              'cons_optimizer_half2': cons_optimizer_half2.state_dict(),
                              'dec_half1_optimizer': dec_half1_optimizer.state_dict(),
                              'dec_half2_optimizer': dec_half2_optimizer.state_dict(),
                              'indices_half1': half1_indices,
                              'refinement_directory': refinement_star_file}
                if epoch == (n_epochs - 1):
                    checkpoint_file = deformations_directory / 'checkpoint_final.pth'
                    torch.save(checkpoint, checkpoint_file)
                else:
                    checkpoint_file = deformations_directory / \
                        f'{epoch:03}.pth'
                    torch.save(checkpoint, checkpoint_file)
                xyz_file = deformations_directory / f'{epoch:03}.xyz'
                write_xyz(
                    deformation_half1.model_positions,
                    xyz_file,
                    box_size=box_size,
                    ang_pix=ang_pix,
                    class_id=gaussian_widths
                )
                graph2bild(pos_h1, gr_h1, str(
                    output_directory) + '/graph' + str(
                    epoch).zfill(3), color=epoch % 65)

                with mrcfile.new(
                    str(output_directory) + '/volume_half1_' + str(
                        epoch).zfill(3) + '.mrc', overwrite=True) as mrc:
                    mrc.set_data(
                        (V_h1[0] / torch.mean(V_h1[0])).float().numpy())
                    mrc.voxel_size = ang_pix
                with mrcfile.new(
                    str(output_directory) + '/volume_half2_' + str(
                        epoch).zfill(3) + '.mrc', overwrite=True) as mrc:
                    mrc.set_data(
                        (V_h2[0] / torch.mean(V_h2[0])).float().numpy())
                    mrc.voxel_size = ang_pix
