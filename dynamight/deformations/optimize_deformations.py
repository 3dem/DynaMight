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

import numpy as np
import torch
import mrcfile
import typer

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torch_geometric.nn import knn_graph, radius_graph
from .regularization import calibrate_regularization_parameter
from ..data.handlers.particle_image_preprocessor import \
    ParticleImagePreprocessor
from ..data.dataloaders.relion import RelionDataset
from ..data.handlers.io_logger import IOLogger
from ..models.constants import ConsensusInitializationMode
from ..models.decoder import DisplacementDecoder
from ..models.encoder import HetEncoder
from ..models.blocks import LinearBlock
from ..models.pose import PoseModule
from ..models.utils import initialize_points_from_volume
from ..utils.utils_new import compute_threshold, load_models, add_weight_decay_to_named_parameters, \
    reset_all_linear_layer_weights, graph2bild, generate_form_factor, \
    visualize_latent, tensor_plot, tensor_imshow, tensor_scatter, \
    apply_ctf, write_xyz, calculate_grid_oversampling_factor, generate_data_normalization_mask, FSC
from ._train_single_epoch_half import train_epoch
from ._update_model import update_model_positions
# from coarse_grain import optimize_coarsegraining

# TODO: add coarse graining to GitHub


from typer import Option

from .._cli import cli


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
    weight_decay: float = 0.,
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

    # initialise poses and pose optimisers
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
    consensus_update_rate_h1 = consensus_update_rate
    consensus_update_rate_h2 = consensus_update_rate
    if initialization_mode == ConsensusInitializationMode.MODEL:
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

    if initialization_mode == ConsensusInitializationMode.MAP:
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

    # initialise the network parts, encoder and decoder for each half set
    if checkpoint_file is not None:
        encoder_half1, encoder_half2, decoder_half1, decoder_half2 = load_models(
            checkpoint_file, device, box_size, n_classes
        )
    else:
        encoder_half1 = HetEncoder(box_size, latent_dim, 1).to(device)
        encoder_half2 = HetEncoder(box_size, latent_dim, 1).to(device)
        decoder_kwargs = {
            'box_size': box_size,
            'ang_pix': ang_pix,
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
        decoder_half1 = DisplacementDecoder(**decoder_kwargs).to(device)
        decoder_half2 = DisplacementDecoder(**decoder_kwargs).to(device)
        for decoder in (decoder_half1, decoder_half2):
            decoder.initialize_physical_parameters(reference_volume=Ivol)

    if mask_file:
        with mrcfile.open(mask_file) as mrc:
            mask = torch.tensor(mrc.data)
        mask = mask.movedim(0, 2).movedim(0, 1)

    print('consensus model  initialization finished')
    if initialization_mode != ConsensusInitializationMode.MODEL.EMPTY:
        decoder_half1.model_positions.requires_grad = False
        decoder_half2.model_positions.requires_grad = False

    # setting up parameters for optimisers, want non-physical parameters
    enc_half1_params = encoder_half1.parameters()
    enc_half2_params = encoder_half2.parameters()
    enc_half1_optimizer = torch.optim.Adam(enc_half1_params, lr=LR)
    enc_half2_optimizer = torch.optim.Adam(enc_half2_params, lr=LR)

    dec_half1_params = add_weight_decay_to_named_parameters(
        decoder_half1, weight_decay=weight_decay)
    dec_half2_params = add_weight_decay_to_named_parameters(
        decoder_half2, weight_decay=weight_decay)
    dec_half1_optimizer = torch.optim.Adam(dec_half1_params, lr=LR)
    dec_half2_optimizer = torch.optim.Adam(dec_half2_params, lr=LR)

    physical_parameter_optimizer_half1 = torch.optim.Adam(
        decoder_half1.physical_parameters, lr=posLR)
    physical_parameter_optimizer_half2 = torch.optim.Adam(
        decoder_half2.physical_parameters, lr=posLR)

    mean_dist_half1 = torch.zeros(n_points)
    mean_dist_half2 = torch.zeros(n_points)

    data_normalization_mask = generate_data_normalization_mask(
        box_size, dampening_factor=apply_bfactor, device=device)

    '--------------------------------------------------------------------------------------------------------------------'
    'Start Training'
    '--------------------------------------------------------------------------------------------------------------------'

    kld_weight = batch_size / len(particle_dataset)
    # beta = 0  # kld_weight**2 * 0.0006
    beta = kld_weight**2 * 0.0006

    epoch_t = 0
    if n_warmup_epochs == None:
        n_warmup_epochs = 0

    old_loss_half1 = 1e8
    old_loss_half2 = 1e8

    # connectivity graphs are computed differently depending on the initialisation mode
    with torch.no_grad():
        if initialization_mode == ConsensusInitializationMode.MODEL:
            # compute the connectivity graph
            decoder_half1.loss_mode = 'model'
            decoder_half2.loss_mode = 'model'
            pos_h1 = decoder_half1.model_positions * box_size * ang_pix
            pos_h2 = decoder_half2.model_positions * box_size * ang_pix
            gr1 = gr
            gr2 = gr
            cons_dis = torch.pow(
                torch.sum((pos_h1[gr1[0]] - pos_h1[gr1[1]]) ** 2, 1), 0.5)
            distance = 0
            decoder_half1.image_smoother.B.requires_grad = False
            decoder_half2.image_smoother.B.requires_grad = False
            consensus_model.i2F.B.requires_grad = False
        else:  # no atomic model provided
            decoder_half1.loss_mode = 'density'
            decoder_half2.loss_mode = 'density'
            decoder_half1.compute_neighbour_graph()
            decoder_half2.compute_neighbour_graph()
            decoder_half1.compute_radius_graph()
            decoder_half2.compute_radius_graph()

            print('mean distance in graph for half 1:',
                  decoder_half1.mean_neighbour_distance.item(
                  ),
                  'Angstrom ;This distance is also used to construct the initial graph ')
            print('mean distance in graph for half 2:',
                  decoder_half2.mean_neighbour_distance.item(
                  ),
                  'Angstrom ;This distance is also used to construct the initial graph ')

    # assign indices to particles for half set division
    tot_latent_dim = encoder_half1.latent_dim
    half1_indices = []
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

    # the actual training loop
    for epoch in range(n_epochs):
        # first, recompute the graphs
        with torch.no_grad():
            if initialization_mode in (ConsensusInitializationMode.EMPTY,
                                       ConsensusInitializationMode.MAP):
                decoder_half1.compute_neighbour_graph()
                decoder_half2.compute_neighbour_graph()
                decoder_half1.compute_radius_graph()
                decoder_half2.compute_radius_graph()
            else:
                gr1 = gr
                gr2 = gr
                pos = decoder_half1.model_positions * box_size * ang_pix
                cons_dis = torch.pow(
                    torch.sum((pos[gr1[0]] - pos[gr1[1]]) ** 2, 1), 0.5)

        angles_op.zero_grad()
        shifts_op.zero_grad()
        if epoch > 0:
            print('Epoch:', epoch, 'Epoch time:', epoch_t)

        if epoch == n_warmup_epochs:
            dec_half1_params = add_weight_decay_to_named_parameters(
                decoder_half1, weight_decay=weight_decay
            )
            dec_half2_params = add_weight_decay_to_named_parameters(
                decoder_half2, weight_decay=weight_decay
            )
            dec_half1_optimizer = torch.optim.Adam(dec_half1_params, lr=LR)
            dec_half2_optimizer = torch.optim.Adam(dec_half2_params, lr=LR)

        # initialise running losses
        running_recloss_half1 = 0
        running_latloss_half1 = 0
        running_total_loss_half1 = 0
        var_total_loss_half1 = 0

        running_recloss_half2 = 0
        running_latloss_half2 = 0
        running_total_loss_half2 = 0
        var_total_loss_half2 = 0

        start_time = time.time()

        # initialise the latent space and the dis
        latent_space = np.zeros([len(particle_dataset), tot_latent_dim])
        diff = np.zeros([len(particle_dataset), 1])

        # mean of displacements per-gaussian
        mean_dist_h1 = torch.zeros(decoder_half1.n_points)
        mean_dist_h2 = torch.zeros(decoder_half2.n_points)

        # variance of displacements per-gaussian
        displacement_variance_h1 = torch.zeros_like(mean_dist_h1)
        displacement_variance_h2 = torch.zeros_like(mean_dist_h2)

        # calculate regularisation parameter as moving average over epochs
        if epoch == 0:
            if initialization_mode == ConsensusInitializationMode.MODEL:
                lambda_regularization_half1 = 1
                lambda_regularization_half2 = 1
            else:
                lambda_regularization_half1 = 0
                lambda_regularization_half2 = 0
        else:
            previous = lambda_regularization_half1
            current = calibrate_regularization_parameter(
                dataset=dataset_half1,
                data_preprocessor=data_preprocessor,
                encoder=encoder_half1,
                decoder=decoder_half1,
                particle_shifts=shifts,
                particle_euler_angles=angles,
                data_normalization_mask=data_normalization_mask,
                regularization_factor=regularization_factor,
                subset_percentage=10,
                batch_size=batch_size,
                mode=initialization_mode,
            )
            lambda_regularization_half1 = 0.9 * previous + 0.1 * current

            previous = lambda_regularization_half2
            current = calibrate_regularization_parameter(
                dataset=dataset_half2,
                data_preprocessor=data_preprocessor,
                encoder=encoder_half2,
                decoder=decoder_half2,
                particle_shifts=shifts,
                particle_euler_angles=angles,
                data_normalization_mask=data_normalization_mask,
                regularization_factor=regularization_factor,
                subset_percentage=10,
                batch_size=batch_size,
                mode=initialization_mode,
            )
            lambda_regularization_half2 = 0.9 * previous + 0.1 * current

        print('new regularization parameter for half 1 is ',
              lambda_regularization_half1)
        print('new regularization parameter for half 2 is ',
              lambda_regularization_half2)

        # training starts!
        latent_space, losses_half1, displacement_statistics_half1, idix_half1, visualization_data_half1 = train_epoch(
            encoder_half1,
            enc_half1_optimizer,
            decoder_half1,
            dec_half1_optimizer,
            physical_parameter_optimizer_half1,
            data_loader_half1,
            angles,
            shifts,
            data_preprocessor,
            epoch,
            n_warmup_epochs,
            data_normalization_mask,
            latent_space,
            latent_weight=beta,
            regularization_parameter=lambda_regularization_half1,
            consensus_update_pooled_particles=consensus_update_pooled_particles,
            mode=initialization_mode,
        )

        print(idix_half1)

        latent_space, losses_half2, displacement_statistics_half2, idix_half2, visualization_data_half2 = train_epoch(
            encoder_half2,
            enc_half2_optimizer,
            decoder_half2,
            dec_half2_optimizer,
            physical_parameter_optimizer_half2,
            data_loader_half2,
            angles,
            shifts,
            data_preprocessor,
            epoch,
            n_warmup_epochs,
            data_normalization_mask,
            latent_space,
            latent_weight=beta,
            regularization_parameter=lambda_regularization_half2,
            consensus_update_pooled_particles=consensus_update_pooled_particles,
            mode=initialization_mode,
        )

        print(idix_half2)

        angles_op.step()
        shifts_op.step()

        current_angles = angles.detach().cpu().numpy()
        angular_error = np.mean(np.square(current_angles - original_angles))

        current_shifts = shifts.detach().cpu().numpy()
        translational_error = np.mean(
            np.square(current_shifts - original_shifts))

        poses = PoseModule(box_size, device, torch.tensor(
            current_angles), torch.tensor(current_shifts))

        # update consensus model
        # todo: simplify update_consensus_model
        if epoch > (n_warmup_epochs - 1) and consensus_update_rate != 0:

            new_pos_h1 = update_model_positions(particle_dataset, data_preprocessor, encoder_half1,
                                                decoder_half1, shifts, angles,  idix_half1, consensus_update_pooled_particles, batch_size)
            new_pos_h2 = update_model_positions(particle_dataset, data_preprocessor, encoder_half2,
                                                decoder_half2, shifts, angles, idix_half2, consensus_update_pooled_particles, batch_size)

            if losses_half1['reconstruction_loss'] < old_loss_half1 and consensus_update_rate_h1 != 0:
                decoder_half1.model_positions = torch.nn.Parameter(
                    (
                        1 - consensus_update_rate_h1) * decoder_half1.model_positions + consensus_update_rate_h1 * new_pos_h1,
                    requires_grad=False)
                old_loss_half1 = losses_half1['reconstruction_loss']
                nosub_ind_h1 = 0

            if losses_half2['reconstruction_loss'] < old_loss_half2 and consensus_update_rate_h2 != 0:
                decoder_half2.model_positions = torch.nn.Parameter(
                    (
                        1 - consensus_update_rate_h2) * decoder_half2.model_positions + consensus_update_rate_h2 * new_pos_h2,
                    requires_grad=False)
                old_loss_half2 = losses_half2['reconstruction_loss']
                nosub_ind_h2 = 0

            if losses_half1['reconstruction_loss'] > old_loss_half1:
                nosub_ind_h1 += 1
                print('No consensus updates for ',
                      nosub_ind_h1, ' epochs on half-set 1')
                if nosub_ind_h1 == 1:
                    consensus_update_rate_h1 *= 0.8
                if consensus_update_rate_h1 < 0.1:
                    consensus_update_rate_h1 = 0
                    reset_all_linear_layer_weights(decoder_half1)
                    reset_all_linear_layer_weights(encoder_half1)
                    regularization_factor_h1 = 1
                    initialization_mode = ConsensusInitializationMode.MAP
            if losses_half2['reconstruction_loss'] > old_loss_half2:
                nosub_ind_h2 += 1
                print('No consensus updates for ',
                      nosub_ind_h2, ' epochs on half-set 2')
                if nosub_ind_h2 == 1:
                    consensus_update_rate_h2 *= 0.8
                if consensus_update_rate_h2 < 0.1:
                    consensus_update_rate_h2 = 0
                    reset_all_linear_layer_weights(decoder_half2)
                    reset_all_linear_layer_weights(encoder_half2)
                    regularization_factor_h2 = 1
                    initialization_mode = ConsensusInitializationMode.MAP

        with torch.no_grad():
            frc_half1 = losses_half1['fourier_ring_correlation'] / \
                len(dataset_half1)
            frc_half2 = losses_half2['fourier_ring_correlation'] / \
                len(dataset_half2)
            decoder_half1.compute_neighbour_graph()
            decoder_half2.compute_neighbour_graph()

            N_graph_h1 = decoder_half1.radius_graph.shape[1]
            N_graph_h2 = decoder_half2.radius_graph.shape[1]

            print('mean distance in graph in Angstrom in half 1:',
                  decoder_half1.mean_neighbour_distance.item(), ' Angstrom')
            print('mean distance in graph in Angstrom in half 2:',
                  decoder_half2.mean_neighbour_distance.item(), ' Angstrom')
            displacement_variance_half1 = displacement_statistics_half1[
                'displacement_variances']
            displacement_variance_half2 = displacement_statistics_half2[
                'displacement_variances']
            mean_dist_half1 = displacement_statistics_half1['mean_displacements']
            mean_dist_half2 = displacement_statistics_half2['mean_displacements']
            displacement_variance_half1 /= len(dataset_half1)
            displacement_variance_half2 /= len(dataset_half2)
            mean_dist_half1 /= len(dataset_half1)
            mean_dist_half2 /= len(dataset_half2)
            D_var = torch.stack(
                [displacement_variance_half1, displacement_variance_half2], 1)

            if initialization_mode in (
                    ConsensusInitializationMode.EMPTY, ConsensusInitializationMode.MAP):
                decoder_half1.compute_radius_graph()
                decoder_half2.compute_radius_graph()

            ff2 = generate_form_factor(
                decoder_half1.image_smoother.A, decoder_half1.image_smoother.B,
                box_size)
            ff2b = generate_form_factor(
                decoder_half2.image_smoother.A, decoder_half2.image_smoother.B,
                box_size)
            FF = np.concatenate([ff2, ff2b], 1)

            ind1 = torch.randint(0, box_size - 1, (1, 1))
            ind2 = torch.randint(0, box_size - 1, (1, 1))

            x = visualization_data_half1['projection_image'][0]
            yd = visualization_data_half1['target_image'][0]
            V_h1 = decoder_half1.generate_consensus_volume()
            V_h2 = decoder_half2.generate_consensus_volume()
            fourier_shell_correlation, res = FSC(
                V_h1[0].float(), V_h2[0].float(), ang_pix)
            if x.is_complex():
                pass
            else:
                x = torch.fft.fft2(x, dim=[-2, -1], norm='ortho')

            if tot_latent_dim > 2:
                if epoch % 5 == 0 and epoch > n_warmup_epochs:
                    summ.add_figure("Data/latent",
                                    visualize_latent(latent_space, c=torch.cat(
                                        [torch.zeros(len(dataset_half1)),
                                         torch.ones(len(dataset_half2))], 0), s=3,
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
                                    c=torch.cat([torch.zeros(len(dataset_half1)),
                                                 torch.ones(len(dataset_half2))], 0),
                                    s=3,
                                    alpha=0.2),
                                epoch)
                summ.add_figure("Data/latent2",
                                visualize_latent(
                                    latent_space, c=cols, s=3, alpha=0.2),
                                epoch)
            summ.add_scalar("Loss/kld_loss",
                            (losses_half1['latent_loss'] + losses_half2[
                                'latent_loss']) / (
                                len(data_loader_half1) + len(
                                    data_loader_half2)), epoch)
            summ.add_scalar("Loss/mse_loss",
                            (losses_half1['reconstruction_loss'] + losses_half2[
                                'reconstruction_loss']) / (
                                len(data_loader_half1) + len(
                                    data_loader_half2)), epoch)
            summ.add_scalars("Loss/mse_loss_halfs",
                             {'half1': (losses_half1['reconstruction_loss']) / (len(
                                 data_loader_half1)),
                              'half2': (losses_half2['reconstruction_loss']) / (
                                  len(data_loader_half2))}, epoch)
            summ.add_scalar("Loss/total_loss", (
                losses_half1['loss'] + losses_half2['loss']) / (
                len(data_loader_half1) + len(
                    data_loader_half2)), epoch)
            summ.add_scalar("Loss/dist_loss",
                            (losses_half1['geometric_loss'] + losses_half2[
                                'geometric_loss']) / (
                                len(data_loader_half1) + len(
                                    data_loader_half2)), epoch)

            summ.add_scalar(
                "Loss/variance1a",
                decoder_half1.image_smoother.B[0].detach().cpu(), epoch)
            summ.add_scalar(
                "Loss/variance2a",
                decoder_half2.image_smoother.B[0].detach().cpu(), epoch)

            summ.add_figure("Data/cons_amp_h1",
                            tensor_plot(decoder_half1.amp.detach()), epoch)
            summ.add_figure("Data/cons_amp_h2",
                            tensor_plot(decoder_half2.amp.detach()), epoch)
            summ.add_figure("Data/FSC_half_maps",
                            tensor_plot(fourier_shell_correlation), epoch)
            summ.add_scalar("Loss/N_graph_h1", N_graph_h1, epoch)
            summ.add_scalar("Loss/N_graph_h2", N_graph_h2, epoch)
            summ.add_scalar("Loss/reg_param_h1",
                            lambda_regularization_half1, epoch)
            summ.add_scalar("Loss/reg_param_h2",
                            lambda_regularization_half2, epoch)
            summ.add_scalar("Loss/substitute_h1",
                            consensus_update_rate_h1, epoch)
            summ.add_scalar("Loss/substitute_h2",
                            consensus_update_rate_h2, epoch)
            summ.add_scalar("Loss/pose_error", angular_error, epoch)
            summ.add_scalar("Loss/trans_error", translational_error, epoch)
            summ.add_figure("Data/output", tensor_imshow(torch.fft.fftshift(
                apply_ctf(visualization_data_half1['projection_image'][0],
                          visualization_data_half1['ctf'][0].float()).squeeze().cpu(),
                dim=[-1, -2])), epoch)
            summ.add_figure(
                "Data/input", tensor_imshow(
                    visualization_data_half1['input_image'][
                        0].squeeze().detach().cpu()),
                epoch)
            summ.add_figure("Data/target", tensor_imshow(torch.fft.fftshift(
                apply_ctf(visualization_data_half1['target_image'][0],
                          data_normalization_mask.float()
                          ).squeeze().cpu(),
                dim=[-1, -2])),
                epoch)
            summ.add_figure("Data/cons_points_z_half1",
                            tensor_scatter(decoder_half1.model_positions[:, 0],
                                           decoder_half1.model_positions[:, 1],
                                           c=mean_dist_half1, s=3), epoch)
            summ.add_figure("Data/cons_points_z_half2",
                            tensor_scatter(decoder_half2.model_positions[:, 0],
                                           decoder_half2.model_positions[:, 1],
                                           c=mean_dist_half2, s=3), epoch)
            summ.add_figure(
                "Data/deformed_points",
                tensor_scatter(visualization_data_half1['deformed_points'][0, :, 0],
                               visualization_data_half1['deformed_points'][0, :, 1],
                               'b',
                               s=0.1), epoch)

            summ.add_figure("Data/projection_image",
                            tensor_imshow(torch.fft.fftshift(torch.real(
                                torch.fft.ifftn(
                                    visualization_data_half1['projection_image'][0],
                                    dim=[-1,
                                         -2])).squeeze().detach().cpu(),
                                dim=[-1, -2])),
                            epoch)
            summ.add_figure("Data/mod_model", tensor_imshow(apply_ctf(
                visualization_data_half1['projection_image'][0],
                visualization_data_half1['ctf'][0].float())), epoch)
            summ.add_figure("Data/shapes", tensor_plot(FF), epoch)
            summ.add_figure("Data/dis_var", tensor_plot(D_var), epoch)

            summ.add_figure("Data/frc_h1", tensor_plot(frc_half1), epoch)
            summ.add_figure("Data/frc_h2", tensor_plot(frc_half2), epoch)

        epoch_t = time.time() - start_time

        if epoch % 1 == 0 or epoch == (n_epochs - 1):
            with torch.no_grad():
                V_h1 = decoder_half1.generate_consensus_volume().cpu()
                V_h2 = decoder_half2.generate_consensus_volume().cpu()
                gaussian_widths = torch.argmax(torch.nn.functional.softmax(
                    decoder_half1.ampvar, dim=0), dim=0)
                checkpoint = {'encoder_half1': encoder_half1,
                              'encoder_half2': encoder_half2,
                              'decoder_half1': decoder_half1,
                              'decoder_half2': decoder_half2,
                              'poses': poses,
                              'encoder_half1_state_dict': encoder_half1.state_dict(),
                              'encoder_half2_state_dict': encoder_half2.state_dict(),
                              'decoder_half1_state_dict': decoder_half1.state_dict(),
                              'decoder_half2_state_dict': decoder_half2.state_dict(),
                              'poses_state_dict': poses.state_dict(),
                              'enc_half1_optimizer': enc_half1_optimizer.state_dict(),
                              'enc_half2_optimizer': enc_half2_optimizer.state_dict(),
                              'cons_optimizer_half1': physical_parameter_optimizer_half1.state_dict(),
                              'cons_optimizer_half2': physical_parameter_optimizer_half2.state_dict(),
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
                    decoder_half1.model_positions,
                    xyz_file,
                    box_size=box_size,
                    ang_pix=ang_pix,
                    class_id=gaussian_widths
                )
                graph2bild(decoder_half1.model_positions, decoder_half1.radius_graph,
                           str(
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
