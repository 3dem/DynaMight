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
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torch_geometric.nn import knn_graph, radius_graph
from .regularization import calibrate_regularization_parameter
from ..data.handlers.particle_image_preprocessor import \
    ParticleImagePreprocessor
from ..data.dataloaders.relion import RelionDataset, write_relion_job_exit_status, abort_if_relion_abort, is_relion_abort
from ..data.handlers.io_logger import IOLogger
from ..models.constants import ConsensusInitializationMode, RegularizationMode
from ..models.decoder import DisplacementDecoder, align_halfs
from ..models.encoder import HetEncoder, N2F_Encoder
from ..models.blocks import LinearBlock
from ..models.pose import PoseModule
from ..models.utils import initialize_points_from_volume
from ..utils.utils_new import compute_threshold, load_models, add_weight_decay_to_named_parameters, graph2bild, generate_form_factor, \
    visualize_latent, tensor_plot, tensor_imshow, tensor_scatter, \
    apply_ctf, write_xyz, calculate_grid_oversampling_factor, generate_data_normalization_mask, FSC, radial_index_mask, radial_index_mask3
from ._train_single_epoch_half import train_epoch, val_epoch, get_edge_weights, get_edge_weights_mask
from ._update_model import update_model_positions
from ..utils.coarse_grain import optimize_coarsegraining

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
    atomic_model: Optional[Path] = None,
    initial_threshold: Optional[float] = None,
    initial_resolution: int = 8,
    mask_file: Optional[Path] = None,
    checkpoint_file: Optional[Path] = None,
    n_gaussians: int = 30000,
    n_gaussian_widths: int = 2,
    n_latent_dimensions: int = 5,
    n_positional_encoding_dimensions: int = 10,
    n_linear_layers: int = 8,
    n_neurons_per_layer: int = 32,
    n_warmup_epochs: int = 10,
    weight_decay: float = 0,
    consensus_update_rate: float = 1,
    consensus_update_decay: float = 0.95,
    consensus_update_pooled_particles: int = 500,
    regularization_factor: float = 0.9,
    apply_bfactor: float = 0,
    particle_diameter: Optional[float] = None,
    soft_edge_width: float = 20,
    batch_size: int = 128,
    gpu_id: Optional[int] = 0,
    n_epochs: int = Option(600),
    n_threads: int = 4,
    preload_images: bool = False,
    n_workers: int = 4,
    combine_resolution: Optional[float] = 8,
    pipeline_control=None,
    use_data_normalization: bool = True,
):

    try:
        # create directory structure
        deformations_directory = output_directory / 'forward_deformations'
        volumes_directory = deformations_directory / 'volumes'
        graphs_directory = deformations_directory / 'graphs'
        checkpoints_directory = deformations_directory / 'checkpoints'

        deformations_directory.mkdir(exist_ok=True, parents=True)
        volumes_directory.mkdir(exist_ok=True, parents=True)
        graphs_directory.mkdir(exist_ok=True, parents=True)
        checkpoints_directory.mkdir(exist_ok=True, parents=True)

        add_free_gaussians = 0

        # initialise logging
        summ = SummaryWriter(output_directory)
        sys.stdout = IOLogger(os.path.join(output_directory, 'std.out'))

        torch.set_num_threads(n_threads)

        # set consensus initialisation mode
        if initial_model == None:
            initialization_mode = ConsensusInitializationMode.EMPTY
            regularization_mode_half1 = RegularizationMode.EMPTY
            regularization_mode_half2 = RegularizationMode.EMPTY
        elif str(initial_model).endswith('.mrc') and atomic_model == None:
            initialization_mode = ConsensusInitializationMode.MAP
            regularization_mode_half1 = RegularizationMode.MAP
            regularization_mode_half2 = RegularizationMode.MAP
        else:
            initialization_mode = ConsensusInitializationMode.MODEL
            regularization_mode_half1 = RegularizationMode.MODEL
            regularization_mode_half2 = RegularizationMode.MODEL

        if gpu_id is None:
            typer.echo("Running on CPU")
        else:
            typer.echo(f"Training on GPU {gpu_id}")

        device = 'cpu' if gpu_id is None else 'cuda:' + str(gpu_id)
        torch.cuda.empty_cache()
        typer.echo('Initializing the particle dataset')

        relion_dataset = RelionDataset(
            path=refinement_star_file.resolve(),
            circular_mask_thickness=soft_edge_width,
            particle_diameter=particle_diameter,
        )
        particle_dataset = relion_dataset.make_particle_dataset()
        if preload_images:
            particle_dataset.preload_images()
        diameter_ang = relion_dataset.particle_diameter
        box_size = relion_dataset.box_size
        ang_pix = relion_dataset.pixel_spacing_angstroms

        print('Number of particles:', len(particle_dataset))

        consensus_update_pooled_particles = int(len(particle_dataset)/200)
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
            dataset_half1 = torch.utils.data.Subset(
                particle_dataset, inds_half1)
            dataset_half2 = torch.utils.data.Subset(
                particle_dataset, inds_half2)

        else:

            train_dataset, val_dataset = torch.utils.data.dataset.random_split(
                particle_dataset, [len(particle_dataset)-len(particle_dataset)//10, len(particle_dataset)//10])
            dataset_half1, dataset_half2 = torch.utils.data.dataset.random_split(
                train_dataset, [len(train_dataset) // 2,
                                len(train_dataset) - len(
                    train_dataset) // 2])

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

        data_loader_val = DataLoader(
            dataset=val_dataset,
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
        regularization_factor_h1 = regularization_factor
        regularization_factor_h2 = regularization_factor
        if initialization_mode == ConsensusInitializationMode.MODEL:
            print('generating coarse grained model')
            initial_points, gr, amp = optimize_coarsegraining(
                atomic_model, box_size, ang_pix, device, str(output_directory),
                n_gaussian_widths, add_free_gaussians)

            if consensus_update_rate == None:
                consensus_update_rate = 0

        if initialization_mode in (ConsensusInitializationMode.MAP,
                                   ConsensusInitializationMode.EMPTY):
            n_points = n_gaussians

        else:
            n_points = initial_points.shape[0]

        with mrcfile.open(initial_model) as mrc:
            Ivol = torch.tensor(mrc.data)

        if initial_threshold == None:
            initial_threshold = compute_threshold(Ivol, percentage=99)
        initial_points = initialize_points_from_volume(
            Ivol.movedim(0, 2).movedim(0, 1),
            threshold=initial_threshold,
            n_points=n_points,
        )

        # define optimisation parameters
        pos_enc_dim = n_positional_encoding_dimensions

        LR = 0.001
        posLR = 0.001
        final = 0
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

            if initialization_mode == ConsensusInitializationMode.MAP:
                with mrcfile.open(initial_model) as mrc:
                    Ivol = torch.tensor(mrc.data)
                    fits = False
                    while fits == False:
                        try:
                            for decoder in (decoder_half1, decoder_half2):
                                decoder.initialize_physical_parameters(
                                    reference_volume=Ivol)
                                summ.add_figure("Data/cons_points_z_half1",
                                                tensor_scatter(decoder_half1.model_positions[:, 0],
                                                               decoder_half1.model_positions[:, 1], c=torch.ones(decoder_half1.model_positions.shape[0]), s=3), -1)
                                summ.add_figure("Data/cons_points_z_half2",
                                                tensor_scatter(decoder_half2.model_positions[:, 0],
                                                               decoder_half2.model_positions[:, 1], c=torch.ones(decoder_half2.model_positions.shape[0]), s=3), -1)
                            fits = True
                            print('consensus gaussian models initialized')
                            torch.cuda.empty_cache()
                        except Exception as error:
                            torch.cuda.empty_cache()
                            print(
                                'volume too large: change size of output volumes. (If you want the original box size for the output volumes use a bigger gpu.', error)
                            Ivol = torch.nn.functional.avg_pool3d(
                                Ivol[None, None], (2, 2, 2))
                            Ivol = Ivol[0, 0]
                            decoder_half1.vol_box = decoder_half1.vol_box//2
                            decoder_half2.vol_box = decoder_half2.vol_box//2

            if mask_file:
                with mrcfile.open(mask_file) as mrc:
                    mask = torch.tensor(mrc.data)
                mask = mask.movedim(0, 2).movedim(0, 1)
                decoder_half1.mask = mask
                decoder_half2.mask = mask
                decoder_half1.mask_model_positions()
                decoder_half2.mask_model_positions()

        print('consensus model  initialization finished')
        if initialization_mode != ConsensusInitializationMode.MODEL.EMPTY:
            # decoder_half1.model_positions.requires_grad = True
            # decoder_half2.model_positions.requires_grad = True
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

        baseline_parameter_optimizer_half1 = torch.optim.Adam(
            decoder_half1.baseline_parameters, lr=100*posLR)
        baseline_parameter_optimizer_half2 = torch.optim.Adam(
            decoder_half2.baseline_parameters, lr=100*posLR)

        mean_dist_half1 = torch.zeros(n_points)
        mean_dist_half2 = torch.zeros(n_points)

        data_normalization_mask = generate_data_normalization_mask(
            box_size, dampening_factor=apply_bfactor, device=device)
        Sig = data_normalization_mask

        '--------------------------------------------------------------------------------------------------------------------'
        'Start Training'
        '-----------------------------------------------------------------------------------------------f---------------------'

        kld_weight = batch_size / len(particle_dataset)
        beta = kld_weight**2 * 0.0001  # 0.006

        epoch_t = 0
        if n_warmup_epochs == None:
            n_warmup_epochs = 0

        old_loss_half1 = 1e8
        old_loss_half2 = 1e8
        old2_loss_half1 = 1e8
        old2_loss_half2 = 1e8
        finalization_epochs = 1e4

        # connectivity graphs are computed differently depending on the initialisation mode
        with torch.no_grad():
            if initialization_mode == ConsensusInitializationMode.MODEL:
                # compute the connectivity graph
                pos_h1 = decoder_half1.model_positions * box_size * ang_pix
                pos_h2 = decoder_half2.model_positions * box_size * ang_pix
                gr1 = gr
                gr2 = gr
                cons_dis = torch.pow(
                    torch.sum((pos_h1[gr1[0]] - pos_h1[gr1[1]]) ** 2, 1), 0.5)
                distance = 0
                decoder_half1.image_smoother.B.requires_grad = False
                decoder_half2.image_smoother.B.requires_grad = False
                #consensus_model.i2F.B.requires_grad = False
            else:  # no atomic model provided
                decoder_half1.compute_neighbour_graph()
                decoder_half2.compute_neighbour_graph()
                decoder_half1.compute_radius_graph()
                # decoder_half1.combine_graphs()
                decoder_half2.compute_radius_graph()
                # decoder_half2.combine_graphs()
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
        half1_indices = torch.zeros(len(dataset_half1))
        val_indices = torch.zeros(len(val_dataset))
        total = 0
        with torch.no_grad():
            print('Computing half-set indices')
            for batch_ndx, sample in enumerate(data_loader_half1):
                idx = sample['idx']
                half1_indices[total:(total+idx.shape[0])] = idx
                total += idx.shape[0]
            total = 0
            for batch_ndx, sample in enumerate(tqdm(data_loader_val, file=sys.stdout)):
                idx = sample['idx']
                val_indices[total:(total+idx.shape[0])] = idx
                total += idx.shape[0]

            half1_indices = half1_indices.long()
            val_indices = val_indices.long()

            cols = torch.ones(len(particle_dataset))
            cols[half1_indices] = 0
            cols[val_indices] = 2
            cols = torch.arange(len(particle_dataset))/len(particle_dataset)

        # the actual training loop
        for epoch in range(n_epochs):
            abort_if_relion_abort(output_directory)
            # first, recompute the graphs
            with torch.no_grad():
                if initialization_mode in (ConsensusInitializationMode.EMPTY,
                                           ConsensusInitializationMode.MAP):
                    decoder_half1.compute_neighbour_graph()
                    decoder_half2.compute_neighbour_graph()
                    decoder_half1.compute_radius_graph()
                    # decoder_half1.combine_graphs()
                    decoder_half2.compute_radius_graph()

                    # decoder_half2.combine_graphs()
                    if mask_file != None and epoch > n_warmup_epochs:
                        noise_h1, noise_h2, signal_h1, signal_h2, snr1, snr2, w1, w2, snr_dis1, snr_dis2, snr_e1, snr_e2 = get_edge_weights_mask(
                            encoder_half1,
                            encoder_half2,
                            decoder_half1,
                            decoder_half2,
                            data_loader_val,
                            angles,
                            shifts,
                            data_preprocessor,
                        )
                        old_pos_h1 = decoder_half1.unmasked_positions.detach()
                        old_pos_h2 = decoder_half2.unmasked_positions.detach()

                    else:
                        noise_h1, noise_h2, signal_h1, signal_h2, snr1, snr2, w1, w2, snr_dis1, snr_dis2, snr_e1, snr_e2 = get_edge_weights(
                            encoder_half1,
                            encoder_half2,
                            decoder_half1,
                            decoder_half2,
                            data_loader_val,
                            angles,
                            shifts,
                            data_preprocessor,
                        )

                    w1 = 1/torch.maximum(snr_e1, torch.tensor(0.05))
                    w2 = 1/torch.maximum(snr_e2, torch.tensor(0.05))

                    w1_dis = 1/torch.maximum(snr_dis1, torch.tensor(0.05))
                    w2_dis = 1/torch.maximum(snr_dis2, torch.tensor(0.05))

                    if epoch > n_warmup_epochs:
                        # edge_weights_h1 = w1_dis  # nsr_g1
                        # edge_weights_h2 = w2_dis  # nsr_g2

                        #edge_weights_dis_h1 = w1_dis
                        #edge_weights_dis_h2 = w2_dis

                        edge_weights_h1 = torch.ones_like(w1)
                        edge_weights_h2 = torch.ones_like(w2)
                        edge_weights_dis_h1 = torch.ones_like(w1_dis)
                        edge_weights_dis_h2 = torch.ones_like(w2_dis)
                    else:
                        edge_weights_h1 = torch.ones_like(w1)
                        edge_weights_h2 = torch.ones_like(w2)
                        edge_weights_dis_h1 = torch.ones_like(w1_dis)
                        edge_weights_dis_h2 = torch.ones_like(w2_dis)
                        noise_h1 = torch.ones_like(noise_h1)
                        noise_h2 = torch.ones_like(noise_h2)

                    cols_h1 = torch.round(
                        65*snr_e1/torch.max(snr_e1))
                    cols_h2 = torch.round(
                        65*snr_e2/torch.max(snr_e2))
                    if epoch % 10 == 0:
                        if mask_file:
                            graph2bild(decoder_half1.unmasked_positions*box_size/ang_pix, decoder_half1.neighbour_graph,
                                       graphs_directory / ('graph_half1' + f'{epoch:03}.bild'), edge_thickness=cols_h1, color=epoch % 65)
                            graph2bild(decoder_half2.unmasked_positions*box_size/ang_pix, decoder_half2.neighbour_graph,
                                       graphs_directory / ('graph_half2' + f'{epoch:03}.bild'), edge_thickness=cols_h2, color=epoch % 65)
                        else:
                            graph2bild(decoder_half1.model_positions*box_size/ang_pix, decoder_half1.neighbour_graph,
                                       graphs_directory / ('graph_half1' + f'{epoch:03}.bild'), edge_thickness=cols_h1, color=epoch % 65)
                            graph2bild(decoder_half2.model_positions*box_size/ang_pix, decoder_half2.neighbour_graph,
                                       graphs_directory / ('graph_half2' + f'{epoch:03}.bild'), edge_thickness=cols_h2, color=epoch % 65)
                else:
                    edge_weights_h1 = torch.ones(gr.shape[1]).to(device)
                    edge_weights_h2 = torch.ones(gr.shape[1]).to(device)
                    edge_weights_dis_h1 = torch.ones(gr.shape[1]).to(device)
                    edge_weights_dis_h2 = torch.ones(gr.shape[1]).to(device)
                    noise_h1 = torch.ones(n_points).to(device)
                    noise_h2 = torch.ones(n_points).to(device)
                    decoder_half1.radius_graph = gr.to(device)
                    decoder_half2.radius_graph = gr.to(device)
                    decoder_half1.neighbour_graph = gr.to(device)
                    decoder_half2.neighbour_graph = gr.to(device)
                    positions_ang = decoder_half1.model_positions.detach(
                    ) * decoder_half1.box_size * decoder_half1.ang_pix
                    differences = positions_ang[gr[0]
                                                ] - positions_ang[gr[1]]

                    decoder_half1.model_distances = torch.pow(
                        torch.sum(differences**2, 1), 0.5).to(device)
                    decoder_half2.model_distances = torch.pow(
                        torch.sum(differences**2, 1), 0.5).to(device)
            angles_op.zero_grad()
            shifts_op.zero_grad()
            if epoch > 0:
                print('----------------------------------------------')
                print('Epoch:', epoch, 'Epoch time:', epoch_t)
                print('----------------------------------------------')

            if epoch == n_warmup_epochs:
                dec_half1_params = add_weight_decay_to_named_parameters(
                    decoder_half1, weight_decay=weight_decay
                )
                dec_half2_params = add_weight_decay_to_named_parameters(
                    decoder_half2, weight_decay=weight_decay
                )
                dec_half1_optimizer = torch.optim.Adam(dec_half1_params, lr=LR)
                dec_half2_optimizer = torch.optim.Adam(dec_half2_params, lr=LR)
                physical_parameter_optimizer_half1 = torch.optim.Adam(
                    decoder_half1.physical_parameters, lr=0.1*posLR)
                physical_parameter_optimizer_half2 = torch.optim.Adam(
                    decoder_half2.physical_parameters, lr=0.1*posLR)

                if initialization_mode in (ConsensusInitializationMode.EMPTY,
                                           ConsensusInitializationMode.MAP):
                    decoder_half1.model_positions.requires_grad = True
                    decoder_half2.model_positions.requires_grad = True
                else:
                    decoder_half1.model_positions.requires_grad = False
                    decoder_half2.model_positions.requires_grad = False

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

            if epoch < n_warmup_epochs and (initialization_mode in (ConsensusInitializationMode.EMPTY,
                                                                    ConsensusInitializationMode.MAP)):
                decoder_half1.warmup = True
                decoder_half2.warmup = True
                decoder_half1.model_positions.requires_grad = True
                decoder_half2.model_positions.requires_grad = True

                #decoder_half1.image_smoother.B.requires_grad = False
                #decoder_half2.image_smoother.B.requires_grad = False

            else:
                decoder_half1.warmup = False
                decoder_half2.warmup = False
                decoder_half1.amp.requires_grad = True
                decoder_half2.amp.requires_grad = True
                decoder_half1.image_smoother.B.requires_grad = True
                decoder_half2.image_smoother.B.requires_grad = True

                if initialization_mode in (ConsensusInitializationMode.EMPTY,
                                           ConsensusInitializationMode.MAP):
                    decoder_half1.model_positions.requires_grad = True
                    decoder_half2.model_positions.requires_grad = True
                else:
                    decoder_half1.model_positions.requires_grad = False
                    decoder_half2.model_positions.requires_grad = False

            # calculate regularisation parameter as moving average over epochs
            if epoch < (n_warmup_epochs+1):
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
                    lambda_regularization=lambda_regularization_half1,
                    subset_percentage=10,
                    batch_size=batch_size,
                    mode=regularization_mode_half1,
                    edge_weights=edge_weights_h1,
                    edge_weights_dis=edge_weights_dis_h1
                )

                lambda_regularization_half1 = regularization_factor_h1 * \
                    (0.9 * previous + 0.1 * current)

                previous = lambda_regularization_half2

                current = calibrate_regularization_parameter(
                    dataset=dataset_half2,
                    data_preprocessor=data_preprocessor,
                    encoder=encoder_half2,
                    decoder=decoder_half2,
                    particle_shifts=shifts,
                    particle_euler_angles=angles,
                    data_normalization_mask=data_normalization_mask,
                    lambda_regularization=lambda_regularization_half2,
                    subset_percentage=10,
                    batch_size=batch_size,
                    mode=regularization_mode_half2,
                    edge_weights=edge_weights_h2,
                    edge_weights_dis=edge_weights_dis_h2
                )

                lambda_regularization_half2 = regularization_factor_h2 * \
                    (0.9 * previous + 0.1 * current)

            abort_if_relion_abort(output_directory)
            # if epoch-n_warmup_epochs < 20:
            #    print('now regularization starts')
            #    lambda_regularization_half2 = 0
            #    lambda_regularization_half1 = 0

            print('new regularization parameter for half 1 is ',
                  lambda_regularization_half1)
            print('new regularization parameter for half 2 is ',
                  lambda_regularization_half2)

            # training starts!

            if epoch == 0:
                fits = False
                while fits == False:
                    try:
                        decoder_half1.warmup = False
                        decoder_half1.amp.requires_grad = True
                        decoder_half1.image_smoother.B.requires_grad = True

                        latent_space, losses_half1, displacement_statistics_half1, idix_half1, visualization_data_half1 = train_epoch(
                            encoder_half1,
                            enc_half1_optimizer,
                            decoder_half1,
                            dec_half1_optimizer,
                            physical_parameter_optimizer_half1,
                            baseline_parameter_optimizer_half1,
                            data_loader_half1,
                            angles,
                            shifts,
                            data_preprocessor,
                            epoch,
                            0,
                            torch.zeros_like(data_normalization_mask),
                            latent_space,
                            latent_weight=beta,
                            regularization_parameter=lambda_regularization_half1,
                            consensus_update_pooled_particles=consensus_update_pooled_particles,
                            regularization_mode=regularization_mode_half1,
                            edge_weights=edge_weights_h1,
                            edge_weights_dis=edge_weights_dis_h1
                        )
                        fits = True
                        torch.cuda.empty_cache()
                        print('Using a batch size of', batch_size)
                        decoder_half1.warmup = True
                        decoder_half1.amp.requires_grad = decoder_half2.amp.requires_grad
                        decoder_half1.image_smoother.B.requires_grad = decoder_half2.image_smoother.B.requires_grad
                    except Exception as error:
                        print(error)
                        torch.cuda.empty_cache()
                        batch_size = batch_size//2
                        print(
                            'WARNING! batch size too large for gpu, trying with new batch size of:', batch_size)
                        kld_weight = batch_size / len(particle_dataset)
                        beta = kld_weight**2 * 0.01

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

                        data_loader_val = DataLoader(
                            dataset=val_dataset,
                            batch_size=batch_size,
                            num_workers=n_workers,
                            shuffle=True,
                            pin_memory=True
                        )

            latent_space, losses_half1, displacement_statistics_half1, idix_half1, visualization_data_half1 = train_epoch(
                encoder_half1,
                enc_half1_optimizer,
                decoder_half1,
                dec_half1_optimizer,
                physical_parameter_optimizer_half1,
                baseline_parameter_optimizer_half1,
                data_loader_half1,
                angles,
                shifts,
                data_preprocessor,
                epoch,
                n_warmup_epochs,
                data_normalization_mask,
                latent_space,
                latent_weight=np.minimum(100, epoch)/100*beta,
                regularization_parameter=lambda_regularization_half1,
                consensus_update_pooled_particles=consensus_update_pooled_particles,
                regularization_mode=regularization_mode_half1,
                edge_weights=edge_weights_h1,
                edge_weights_dis=edge_weights_dis_h1
            )

            ref_pos_h1 = update_model_positions(particle_dataset, data_preprocessor, encoder_half1,
                                                decoder_half1, shifts, angles,  idix_half1, consensus_update_pooled_particles, batch_size)
            if epoch > (n_warmup_epochs - 1) and consensus_update_rate != 0:

                if losses_half1['reconstruction_loss'] < (old_loss_half1+old2_loss_half1)/2 and consensus_update_rate_h1 != 0 and (initialization_mode in (ConsensusInitializationMode.EMPTY,
                                                                                                                                                           ConsensusInitializationMode.MAP)):
                    decoder_half1.model_positions = torch.nn.Parameter(
                        ref_pos_h1, requires_grad=True)

                    old2_loss_half1 = old_loss_half1
                    old_loss_half1 = losses_half1['reconstruction_loss']
                    nosub_ind_h1 = 0

            if epoch % 20 == 0 or (consensus_update_rate_h1 == 0 and consensus_update_rate_h2 == 0) and (initialization_mode in (ConsensusInitializationMode.EMPTY,
                                                                                                                                 ConsensusInitializationMode.MAP)):

                with torch.no_grad():
                    ref_vol = decoder_half1.generate_consensus_volume().detach()
                    ref_threshold = compute_threshold(
                        ref_vol[0], percentage=99)
                    ref_mask = ref_vol[0] > ref_threshold

                    filter_width = int(np.round(combine_resolution/ang_pix))
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
                    smooth_ref_mask = torch.nn.functional.conv3d(
                        ref_mask[None, None, :, :, :].float(), mean_filter.to(device), padding=filter_width//2)
                    if box_size > 360:
                        smooth_ref_mask = torch.nn.functional.upsample(
                            smooth_ref_mask, scale_factor=2)

                    ref_mask = smooth_ref_mask > 2/filter_width**3

                    ref_mask = ref_mask[0, 0]

                    with mrcfile.new(volumes_directory / ('mask_half1_' + f'{epoch:03}.mrc'), overwrite=True) as mrc:
                        mrc.set_data(ref_mask.cpu().float().numpy())
                        mrc.voxel_size = ang_pix
            else:
                ref_mask = None

            abort_if_relion_abort(output_directory)

            if initialization_mode in (ConsensusInitializationMode.EMPTY, ConsensusInitializationMode.MAP):
                latent_space, losses_half2, displacement_statistics_half2, idix_half2, visualization_data_half2 = train_epoch(
                    encoder_half2,
                    enc_half2_optimizer,
                    decoder_half2,
                    dec_half2_optimizer,
                    physical_parameter_optimizer_half2,
                    baseline_parameter_optimizer_half2,
                    data_loader_half2,
                    angles,
                    shifts,
                    data_preprocessor,
                    epoch,
                    n_warmup_epochs,
                    data_normalization_mask,
                    latent_space,
                    latent_weight=np.minimum(epoch, 100)/100*beta,
                    regularization_parameter=lambda_regularization_half2,
                    consensus_update_pooled_particles=consensus_update_pooled_particles,
                    regularization_mode=regularization_mode_half2,
                    edge_weights=edge_weights_h2,
                    edge_weights_dis=edge_weights_dis_h2,
                    ref_mask=ref_mask,
                )
            else:

                latent_space, losses_half2, displacement_statistics_half2, idix_half2, visualization_data_half2 = train_epoch(
                    encoder_half2,
                    enc_half2_optimizer,
                    decoder_half2,
                    dec_half2_optimizer,
                    physical_parameter_optimizer_half2,
                    baseline_parameter_optimizer_half2,
                    data_loader_half2,
                    angles,
                    shifts,
                    data_preprocessor,
                    epoch,
                    n_warmup_epochs,
                    data_normalization_mask,
                    latent_space,
                    latent_weight=np.minimum(epoch, 100)/100*beta,
                    regularization_parameter=lambda_regularization_half2,
                    consensus_update_pooled_particles=consensus_update_pooled_particles,
                    regularization_mode=regularization_mode_half2,
                    edge_weights=edge_weights_h2,
                    edge_weights_dis=edge_weights_dis_h2,
                )

            angles_op.step()
            shifts_op.step()

            abort_if_relion_abort(output_directory)

            latent_space, idix_half1_useless, sig1, Err1 = val_epoch(
                encoder_half1,
                enc_half1_optimizer,
                decoder_half1,
                data_loader_val,
                angles,
                shifts,
                data_preprocessor,
                epoch,
                n_warmup_epochs,
                data_normalization_mask,
                latent_space,
                latent_weight=beta,
                consensus_update_pooled_particles=consensus_update_pooled_particles,
            )

            latent_space, idix_half2_useless, sig2, Err2 = val_epoch(
                encoder_half2,
                enc_half2_optimizer,
                decoder_half2,
                data_loader_val,
                angles,
                shifts,
                data_preprocessor,
                epoch,
                n_warmup_epochs,
                data_normalization_mask,
                latent_space,
                latent_weight=beta,
                consensus_update_pooled_particles=consensus_update_pooled_particles,
            )

            current_angles = angles.detach().cpu().numpy()
            angular_error = np.mean(
                np.square(current_angles - original_angles))

            current_shifts = shifts.detach().cpu().numpy()
            translational_error = np.mean(
                np.square(current_shifts - original_shifts))

            poses = PoseModule(box_size, device, torch.tensor(
                current_angles), torch.tensor(current_shifts))

            # update consensus model
            if epoch > (n_warmup_epochs - 1) and (initialization_mode in (ConsensusInitializationMode.EMPTY,
                                                                          ConsensusInitializationMode.MAP)):

                if consensus_update_rate_h1 != 0:
                    new_pos_h1 = update_model_positions(particle_dataset, data_preprocessor, encoder_half1,
                                                        decoder_half1, shifts, angles,  idix_half1, consensus_update_pooled_particles, batch_size)
                else:
                    new_pos_h1 = decoder_half1.model_positions
                new_pos_h2 = update_model_positions(particle_dataset, data_preprocessor, encoder_half2,
                                                    decoder_half2, shifts, angles, idix_half2, consensus_update_pooled_particles, batch_size)

                if (losses_half2['reconstruction_loss'] < (old_loss_half2+old2_loss_half2)/2 and consensus_update_rate_h2 != 0) or epoch % 20 == 0:
                    # decoder_half2.model_positions = torch.nn.Parameter((
                    #    1 - consensus_update_rate_h2) * decoder_half2.model_positions + consensus_update_rate_h2 * new_pos_h2, requires_grad=True)
                    # decoder_half2.model_positions = torch.nn.Parameter(
                    #    new_pos_h2, requires_grad=True)
                    decoder_half2.model_positions = torch.nn.Parameter(

                        new_pos_h2, requires_grad=True)

                    old2_loss_half2 = old_loss_half2
                    old_loss_half2 = losses_half2['reconstruction_loss']
                    nosub_ind_h2 = 0

                if consensus_update_rate_h1 == 0:

                    regularization_factor_h1 = regularization_factor_h1
                    regularization_mode_half1 = RegularizationMode.MODEL

                if consensus_update_rate_h2 == 0:
                    regularization_factor_h2 = regularization_factor_h2
                    regularization_mode_half2 = RegularizationMode.MODEL

                if losses_half1['reconstruction_loss'] > (old_loss_half1+old2_loss_half1)/2 and consensus_update_rate_h1 != 0:
                    nosub_ind_h1 += 1

                    if nosub_ind_h1 == 1:
                        consensus_update_rate_h1 *= consensus_update_decay
                    if consensus_update_rate_h1 < 0.1:
                        consensus_update_rate_h1 = 0

                if losses_half2['reconstruction_loss'] > (old_loss_half2+old2_loss_half2)/2 and consensus_update_rate_h2 != 0:
                    nosub_ind_h2 += 1

                    # for g in dec_half2_optimizer.param_groups:
                    #     g['lr'] *= 0.9
                    # print('new learning rate for half 2 is', g['lr'])
                    if nosub_ind_h2 == 1:
                        consensus_update_rate_h2 *= consensus_update_decay
                    if consensus_update_rate_h2 < 0.1:
                        consensus_update_rate_h2 = 0

                if consensus_update_rate_h1 == 0 and consensus_update_rate_h2 == 0 and initialization_mode != ConsensusInitializationMode.MODEL:
                    if final == 0:
                        update_epochs = epoch
                        finalization_epochs = update_epochs//3
                    final += 1

                    print('Epoch', final, 'of', finalization_epochs,
                          'epochs without consensus update')
            if mask_file:
                decoder_half1.mask_model_positions()
                decoder_half2.mask_model_positions()

            with torch.no_grad():
                frc_half1 = losses_half1['fourier_ring_correlation'] / \
                    len(dataset_half1)
                frc_half2 = losses_half2['fourier_ring_correlation'] / \
                    len(dataset_half2)
                if initialization_mode in (ConsensusInitializationMode.EMPTY,
                                           ConsensusInitializationMode.MAP):
                    decoder_half1.compute_neighbour_graph()
                    decoder_half2.compute_neighbour_graph()
                    N_graph_h1 = decoder_half1.radius_graph.shape[1]
                    N_graph_h2 = decoder_half2.radius_graph.shape[1]
                else:
                    N_graph_h1 = gr.shape[1]
                    N_graph_h2 = gr.shape[1]

                rad_inds, rad_mask = radial_index_mask(
                    grid_oversampling_factor * box_size)
                R = torch.stack(decoder_half1.n_classes *
                                [rad_inds.to(decoder_half1.device)], 0)
                F = torch.exp(-(1/decoder_half1.image_smoother.B[:, None, None]**2) *
                              R**2) * (decoder_half1.image_smoother.A[:, None, None] ** 2)
                FF0 = torch.real(torch.fft.fft2(torch.fft.fftshift(
                    F, dim=[-1, -2]), dim=[-1, -2], norm='ortho'))

                R3, M3 = radial_index_mask3(decoder_half1.vol_box)
                R3 = torch.stack(
                    decoder_half1.image_smoother.n_classes * [R3.to(decoder_half1.device)], 0)

                F3 = torch.exp(-(1/decoder_half1.image_smoother.B[:, None, None,
                                                                  None]**2) * R3**2) * decoder_half1.image_smoother.A[
                    :, None,
                    None,
                    None] ** 2

                if epoch > (n_warmup_epochs+10) and epoch < (n_warmup_epochs + 60):
                    if use_data_normalization == True:
                        Sig = 0.5*Sig + 0.5*(sig1+sig2)/2

                        data_normalization_mask = 1 / \
                            Sig
                        data_normalization_mask /= torch.max(
                            data_normalization_mask)
                        R2, r_mask = radial_index_mask(decoder.box_size)

                        data_normalization_mask *= torch.fft.fftshift(
                            r_mask.to(decoder.device), dim=[-1, -2])

                        data_normalization_mask = (data_normalization_mask /
                                                   torch.sum(data_normalization_mask**2))*(box_size**2)

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

                # max_active = np.argmax(
                #     [displacement_variance_half1.shape[0], displacement_variance_half2.shape[0]])
                # print(max_active)
                # if max_active == 0:
                #     dv = torch.zeros_like(displacement_variance_half1)
                #     dv[:displacement_variance_half2.shape[0]
                #        ] = displacement_variance_half2
                #     displacement_variance_half2 = dv
                # else:
                #     dv = torch.zeros_like(displacement_variance_half2)
                #     dv[:displacement_variance_half1.shape[0]
                #        ] = displacement_variance_half1
                #     displacement_variance_half1 = dv

                # D_var = torch.stack(
                #     [displacement_variance_half1, displacement_variance_half2], 1)
                # print(displacement_variance_half1.shape)
                # print(displacement_variance_half2.shape)
                # print(D_var.shape)

                if initialization_mode in (
                        ConsensusInitializationMode.EMPTY, ConsensusInitializationMode.MAP):
                    decoder_half1.compute_neighbour_graph()
                    decoder_half2.compute_neighbour_graph()
                    decoder_half1.compute_radius_graph()
                    # decoder_half1.combine_graphs()
                    decoder_half2.compute_radius_graph()
                    # decoder_half2.combine_graphs()

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

                if epoch % 1 == 0:
                    if tot_latent_dim > 2:
                        if epoch % 5 == 0 and epoch > n_warmup_epochs:
                            summ.add_figure("Data/latent",
                                            visualize_latent(latent_space, c=cols, s=3,
                                                             alpha=0.2, method='pca'),
                                            epoch)
                            summ.add_figure("Data/latent_val",
                                            visualize_latent(latent_space[val_indices], c=cols[val_indices], s=3,
                                                             alpha=0.2, method='pca'),
                                            epoch)

                    else:
                        summ.add_figure("Data/latent",
                                        visualize_latent(
                                            latent_space,
                                            c=cols,
                                            s=3,
                                            alpha=0.2),
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
                    summ.add_scalar("Loss/geometric_loss",
                                    (losses_half1['geometric_loss'] + losses_half2[
                                        'geometric_loss']) / (
                                        len(data_loader_half1) + len(
                                            data_loader_half2)), epoch)

                    summ.add_scalar(
                        "Loss/variance_h1_1",
                        decoder_half1.image_smoother.B[0].detach().cpu(), epoch)
                    summ.add_scalar(
                        "Loss/variance_h2_1",
                        decoder_half2.image_smoother.B[0].detach().cpu(), epoch)

                    summ.add_scalar(
                        "Loss/amplitude_h1",
                        decoder_half1.image_smoother.A[0].detach().cpu(), epoch)
                    summ.add_scalar(
                        "Loss/amplitude_h2",
                        decoder_half2.image_smoother.A[0].detach().cpu(), epoch)
                    if decoder_half1.n_classes > 1:
                        summ.add_scalar(
                            "Loss/amplitude_h1_2",
                            decoder_half1.image_smoother.A[1].detach().cpu(), epoch)
                        summ.add_scalar(
                            "Loss/amplitude_h2_2",
                            decoder_half2.image_smoother.A[1].detach().cpu(), epoch)
                        summ.add_scalar(
                            "Loss/variance_h1_2",
                            decoder_half1.image_smoother.B[1].detach().cpu(), epoch)
                        summ.add_scalar(
                            "Loss/variance_h2_2",
                            decoder_half2.image_smoother.B[1].detach().cpu(), epoch)

                    summ.add_figure("Data/FSC_half_maps",
                                    tensor_plot(fourier_shell_correlation, fix=[0, 1]), epoch)
                    summ.add_figure("Loss/amplitudes_h1",
                                    tensor_plot(decoder_half1.amp[0].cpu()), epoch)
                    summ.add_figure("Loss/amplitudes_h2",
                                    tensor_plot(decoder_half2.amp[0].cpu()), epoch)
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
                    summ.add_scalar("Loss/trans_error",
                                    translational_error, epoch)
                    summ.add_figure("Data/output", tensor_imshow(torch.fft.fftshift(
                        apply_ctf(visualization_data_half1['projection_image'][0],
                                  visualization_data_half1['ctf'][0].float()).squeeze().cpu(),
                        dim=[-1, -2])), epoch)

                    if initialization_mode in (
                            ConsensusInitializationMode.EMPTY, ConsensusInitializationMode.MAP):
                        summ.add_figure("Data/snr1",
                                        tensor_plot(snr1), epoch)
                        summ.add_figure("Data/snr2",
                                        tensor_plot(snr2), epoch)
                        summ.add_figure("Data/signal1",
                                        tensor_plot(signal_h1), epoch)
                        summ.add_figure("Data/signal2",
                                        tensor_plot(signal_h2), epoch)

                    summ.add_figure(
                        "Data/sig", tensor_imshow(data_normalization_mask), epoch)
                    # summ.add_figure(
                    #    "Data/errsig", tensor_imshow(data_err), epoch)
                    summ.add_figure("Data/target", tensor_imshow(torch.fft.fftshift(
                        apply_ctf(visualization_data_half1['target_image'][0],
                                  data_normalization_mask.float()
                                  ).squeeze().cpu(),
                        dim=[-1, -2])),
                        epoch)

                    if mask_file == None or epoch <= n_warmup_epochs:
                        summ.add_figure("Data/cons_points_z_half1_noise",
                                        tensor_scatter(decoder_half1.model_positions[:, 0],
                                                       decoder_half1.model_positions[:, 1],
                                                       c=(noise_h1/torch.max(noise_h1)).cpu(), s=3), epoch)
                    else:
                        summ.add_figure("Data/cons_points_z_half1_noise",
                                        tensor_scatter(old_pos_h1[:, 0],
                                                       old_pos_h1[:, 1],
                                                       c=(noise_h1/torch.max(noise_h1)).cpu(), s=3), epoch)

                    if mask_file == None or epoch <= n_warmup_epochs:
                        summ.add_figure("Data/cons_points_z_half2_noise",
                                        tensor_scatter(decoder_half2.model_positions[:, 0],
                                                       decoder_half2.model_positions[:, 1],
                                                       c=(noise_h2/torch.max(noise_h2)).cpu(), s=3), epoch)
                    else:
                        summ.add_figure("Data/cons_points_z_half2_noise",
                                        tensor_scatter(old_pos_h2[:, 0],
                                                       old_pos_h2[:, 1],
                                                       c=(noise_h2/torch.max(noise_h2)).cpu(), s=3), epoch)

                    if initialization_mode in (
                            ConsensusInitializationMode.EMPTY, ConsensusInitializationMode.MAP):
                        if mask_file == None or epoch <= n_warmup_epochs:
                            summ.add_figure("Data/cons_points_z_half2_nsr",
                                            tensor_scatter(decoder_half2.model_positions[:, 0],
                                                           decoder_half2.model_positions[:, 1],
                                                           c=(snr2/torch.max(snr2)).cpu(), s=3), epoch)
                        else:
                            summ.add_figure("Data/cons_points_z_half2_nsr",
                                            tensor_scatter(old_pos_h2[:, 0],
                                                           old_pos_h2[:, 1],
                                                           c=(snr2/torch.max(snr2)).cpu(), s=3), epoch)
                    if initialization_mode in (
                            ConsensusInitializationMode.EMPTY, ConsensusInitializationMode.MAP):
                        if mask_file == None or epoch <= n_warmup_epochs:
                            summ.add_figure("Data/cons_points_z_half1_nsr",
                                            tensor_scatter(decoder_half1.model_positions[:, 0],
                                                           decoder_half1.model_positions[:, 1],
                                                           c=(snr1/torch.max(snr1)).cpu(), s=3), epoch)
                        else:
                            summ.add_figure("Data/cons_points_z_half1_nsr",
                                            tensor_scatter(old_pos_h1[:, 0],
                                                           old_pos_h1[:, 1],
                                                           c=(snr1/torch.max(snr1)).cpu(), s=3), epoch)

                    summ.add_figure(
                        "Data/deformed_points",
                        tensor_scatter(visualization_data_half1['deformed_points'][0, :, 0],
                                       visualization_data_half1['deformed_points'][0, :, 1],
                                       c='b',
                                       s=0.1), epoch)

                    summ.add_figure("Data/projection_image",
                                    tensor_imshow(torch.fft.fftshift(torch.real(
                                        torch.fft.ifftn(
                                            visualization_data_half1['projection_image'][0],
                                            dim=[-1,
                                                 -2])).squeeze().detach().cpu(),
                                        dim=[-1, -2])),
                                    epoch)

                    # summ.add_figure("Data/dis_var", tensor_plot(D_var), epoch)

                    summ.add_figure(
                        "Data/frc_h1", tensor_plot(frc_half1), epoch)
                    summ.add_figure(
                        "Data/frc_h2", tensor_plot(frc_half2), epoch)
            epoch_t = time.time() - start_time

            if epoch % 5 == 0 or (final > finalization_epochs) or (epoch == n_epochs-1):

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
                                  'indices_val': val_indices,
                                  'refinement_directory': refinement_star_file,
                                  'batch_size': batch_size}
                    if final > finalization_epochs or epoch == n_epochs-1:
                        checkpoint_file = checkpoints_directory / 'checkpoint_final.pth'
                        torch.save(checkpoint, checkpoint_file)
                    else:
                        checkpoint_file = checkpoints_directory / \
                            f'{epoch:03}.pth'
                        torch.save(checkpoint, checkpoint_file)
                    xyz_file = graphs_directory / ('points'+f'{epoch:03}.xyz')
                    write_xyz(
                        decoder_half1.model_positions,
                        xyz_file,
                        box_size=box_size,
                        ang_pix=ang_pix,
                        class_id=gaussian_widths
                    )

                    with mrcfile.new(volumes_directory / ('volume_half1_' + f'{epoch:03}.mrc'), overwrite=True) as mrc:
                        mrc.set_data(
                            (V_h1[0] / torch.mean(V_h1[0])).float().numpy())
                        mrc.voxel_size = ang_pix
                    with mrcfile.new(volumes_directory / ('volume_half2_' + f'{epoch:03}.mrc'), overwrite=True) as mrc:
                        mrc.set_data(
                            (V_h2[0] / torch.mean(V_h2[0])).float().numpy())
                        mrc.voxel_size = ang_pix
            abort_if_relion_abort(output_directory)

            if final > 0:
                if final > finalization_epochs:
                    write_relion_job_exit_status(
                        output_directory, 'SUCCESS', pipeline_control)
                    break
    except Exception as e:
        print(e)
        write_relion_job_exit_status(
            output_directory, 'FAILURE', pipeline_control)
