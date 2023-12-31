
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
from ..data.dataloaders.relion import RelionDataset
from ..data.handlers.io_logger import IOLogger
from ..models.constants import ConsensusInitializationMode, RegularizationMode
from ..models.decoder import DisplacementDecoder, align_halfs
from ..models.encoder import HetEncoder
from ..models.blocks import LinearBlock
from ..models.pose import PoseModule
from ..models.utils import initialize_points_from_volume
from ..utils.utils_new import compute_threshold, load_models, add_weight_decay_to_named_parameters, \
    reset_all_linear_layer_weights, graph2bild, generate_form_factor, \
    visualize_latent, tensor_plot, tensor_imshow, tensor_scatter, \
    apply_ctf, write_xyz, calculate_grid_oversampling_factor, generate_data_normalization_mask, FSC, radial_index_mask, radial_index_mask3, mask_from_positions
from ._train_single_epoch_half import train_epoch, val_epoch, validate_epoch, get_edge_weights
from ._update_model import update_model_positions
from ..utils.coarse_grain import optimize_coarsegraining

# TODO: add coarse graining to GitHub


from typer import Option

from .._cli import cli


@cli.command(no_args_is_help=True)
def optimize_deformations_single(
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
    n_gaussians: int = 20000,
    n_gaussian_widths: int = 1,
    n_latent_dimensions: int = 2,
    n_positional_encoding_dimensions: int = 10,
    n_linear_layers: int = 8,
    n_neurons_per_layer: int = 32,
    n_warmup_epochs: int = 3,
    weight_decay: float = 0,
    consensus_update_rate: float = 1,
    consensus_update_decay: float = 0.8,
    consensus_update_pooled_particles: int = 100,
    regularization_factor: float = 0.5,
    apply_bfactor: float = 0,
    particle_diameter: Optional[float] = None,
    soft_edge_width: float = 20,
    batch_size: int = 100,
    gpu_id: Optional[int] = 0,
    n_epochs: int = Option(300),
    n_threads: int = 4,
    preload_images: bool = True,
    n_workers: int = 4,
):
    # create directory structure
    deformations_directory = output_directory / 'forward_deformations'
    volumes_directory = deformations_directory / 'volumes'
    graphs_directory = deformations_directory / 'graphs'
    checkpoints_directory = deformations_directory / 'checkpoints'

    deformations_directory.mkdir(exist_ok=True, parents=True)
    volumes_directory.mkdir(exist_ok=True, parents=True)
    graphs_directory.mkdir(exist_ok=True, parents=True)
    checkpoints_directory.mkdir(exist_ok=True, parents=True)
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
        regularization_mode_half1 = RegularizationMode.EMPTY

    elif str(initial_model).endswith('.mrc') and atomic_model == None:
        initialization_mode = ConsensusInitializationMode.MAP
        regularization_mode_half1 = RegularizationMode.MAP

    else:
        initialization_mode = ConsensusInitializationMode.MODEL
        regularization_mode_half1 = RegularizationMode.MODEL

    if gpu_id is None:
        typer.echo("Running on CPU")
    else:
        typer.echo(f"Training on GPU {gpu_id}")

    device = 'cpu' if gpu_id is None else 'cuda:' + str(gpu_id)

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
        dataset_half1 = torch.utils.data.Subset(particle_dataset, inds_half1)

    else:

        # train_dataset = particle_dataset
        dataset_half1 = particle_dataset

    data_loader_half1 = DataLoader(
        dataset=particle_dataset,
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
    print('Initialized data loader of size',
          len(dataset_half1))

    print('box size:', box_size, 'pixel_size:', ang_pix, 'virtual pixel_size:',
          1 / (box_size + 1), ' dimension of latent space: ',
          n_latent_dimensions)
    latent_dim = n_latent_dimensions
    n_classes = n_gaussian_widths

    grid_oversampling_factor = calculate_grid_oversampling_factor(box_size)

    # initialise the gaussian model
    consensus_update_rate_h1 = consensus_update_rate
    regularization_factor_h1 = regularization_factor
    if initialization_mode == ConsensusInitializationMode.MODEL:
        initial_points, gr, amp = optimize_coarsegraining(
            initial_model, box_size, ang_pix, device, str(output_directory),
            n_gaussian_widths, add_free_gaussians)
        print(amp)
        amp = amp / torch.max(amp)

        if consensus_update_rate == None:
            consensus_update_rate = 0

    if initialization_mode in (ConsensusInitializationMode.MAP,
                               ConsensusInitializationMode.EMPTY):
        n_points = n_gaussians

    else:
        n_points = initial_points.shape[0]

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
                initial_threshold = compute_threshold(Ivol, percentage=99)
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
    final = 0
    typer.echo(f'Number of used gaussians: {n_points}')

    # initialise the network parts, encoder and decoder for each half set
    if checkpoint_file is not None:
        encoder_half1, encoder_half2, decoder_half1, decoder_half2 = load_models(
            checkpoint_file, device, box_size, n_classes
        )
    else:
        encoder_half1 = HetEncoder(box_size, latent_dim, 1).to(device)
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

        if initialization_mode != ConsensusInitializationMode.MODEL:
            decoder_half1.initialize_physical_parameters(reference_volume=Ivol)
        summ.add_figure("Data/cons_points_z_half1",
                        tensor_scatter(decoder_half1.model_positions[:, 0],
                                       decoder_half1.model_positions[:, 1], c=torch.ones(decoder_half1.model_positions.shape[0]), s=3), -1)

        if mask_file:
            with mrcfile.open(mask_file) as mrc:
                mask = torch.tensor(mrc.data)
            mask = mask.movedim(0, 2).movedim(0, 1)
            decoder_half1.mask = mask
            decoder_half1.mask_model_positions()

    if initialization_mode == ConsensusInitializationMode.MODEL:
        decoder_half1.amp.data = amp.to(device)

    print('consensus model  initialization finished')
    if initialization_mode != ConsensusInitializationMode.MODEL.EMPTY:
        decoder_half1.model_positions.requires_grad = False

    # setting up parameters for optimisers, want non-physical parameters
    enc_half1_params = encoder_half1.parameters()
    enc_half1_optimizer = torch.optim.Adam(enc_half1_params, lr=LR)

    dec_half1_params = add_weight_decay_to_named_parameters(
        decoder_half1, weight_decay=weight_decay)
    dec_half1_optimizer = torch.optim.Adam(dec_half1_params, lr=LR)

    physical_parameter_optimizer_half1 = torch.optim.Adam(
        decoder_half1.physical_parameters, lr=posLR)

    baseline_parameter_optimizer_half1 = torch.optim.Adam(
        decoder_half1.baseline_parameters, lr=100*posLR)

    mean_dist_half1 = torch.zeros(n_points)

    data_normalization_mask = generate_data_normalization_mask(
        box_size, dampening_factor=apply_bfactor, device=device)
    #data_normalization_mask[0, 0] = 0
    Sig = data_normalization_mask

    '--------------------------------------------------------------------------------------------------------------------'
    'Start Training'
    '--------------------------------------------------------------------------------------------------------------------'

    kld_weight = batch_size / len(particle_dataset)
    beta = kld_weight**2 * 0.0006

    epoch_t = 0
    if n_warmup_epochs == None:
        n_warmup_epochs = 0

    old_loss_half1 = 1e8
    old_loss_half2 = 1e8
    old2_loss_half1 = 1e8
    old2_loss_half2 = 1e8

    # connectivity graphs are computed differently depending on the initialisation mode
    with torch.no_grad():
        if initialization_mode == ConsensusInitializationMode.MODEL:
            # compute the connectivity graph
            pos_h1 = decoder_half1.model_positions * box_size * ang_pix
            decoder_half1.radius_graph = gr
            decoder_half1.neighbour_graph = gr
            gr1 = gr
            cons_dis = torch.pow(
                torch.sum((pos_h1[gr1[0]] - pos_h1[gr1[1]]) ** 2, 1), 0.5)
            distance = 0
            positions_ang = decoder_half1.model_positions.detach()*decoder_half1.box_size * \
                decoder_half1.ang_pix
            differences = positions_ang[gr[0]] - positions_ang[gr[1]]
            neighbour_distances = torch.pow(
                torch.sum(differences**2, dim=1), 0.5)
            decoder_half1.mean_neighbour_distance = torch.mean(
                neighbour_distances)
            decoder_half1.model_distances = torch.pow(
                torch.sum(differences**2, 1), 0.5)
            decoder_half1.mean_graph_distance = torch.mean(
                decoder_half1.model_distances)
            #decoder_half1.image_smoother.B.requires_grad = False
        else:  # no atomic model provided
            decoder_half1.compute_neighbour_graph()
            decoder_half1.compute_radius_graph()
            decoder_half1.combine_graphs()
            print('mean distance in graph for half 1:',
                  decoder_half1.mean_neighbour_distance.item(
                  ),
                  'Angstrom ;This distance is also used to construct the initial graph ')

    # assign indices to particles for half set division
    tot_latent_dim = encoder_half1.latent_dim
    half1_indices = torch.zeros(len(dataset_half1))
    total = 0
    with torch.no_grad():
        print('Computing half-set indices')
        for batch_ndx, sample in tqdm(enumerate(data_loader_half1)):
            idx = sample['idx']
            half1_indices[total:(total+idx.shape[0])] = idx
            total += idx.shape[0]
        total = 0

        half1_indices = half1_indices.long()

        # half1_indices = half1_indices.tolist()
        # val_indices = val_indices.tolist()

        # half1_indices = torch.tensor(
        #     [item for sublist in half1_indices for item in sublist])
        # val_indices = torch.tensor(
        #     [item for sublist in val_indices for item in sublist])
        cols = torch.ones(len(particle_dataset))

    # the actual training loop
    for epoch in range(n_epochs):
        # first, recompute the graphs
        with torch.no_grad():
            if initialization_mode in (ConsensusInitializationMode.EMPTY,
                                       ConsensusInitializationMode.MAP):
                decoder_half1.compute_neighbour_graph()
                decoder_half1.compute_radius_graph()
                decoder_half1.combine_graphs()

        angles_op.zero_grad()
        shifts_op.zero_grad()
        if epoch > 0:
            print('Epoch:', epoch, 'Epoch time:', epoch_t)

        if epoch == n_warmup_epochs:
            dec_half1_params = add_weight_decay_to_named_parameters(
                decoder_half1, weight_decay=weight_decay
            )

            dec_half1_optimizer = torch.optim.Adam(dec_half1_params, lr=LR)
            physical_parameter_optimizer_half1 = torch.optim.SGD(
                decoder_half1.physical_parameters, lr=1*posLR)
            decoder_half1.model_positions.requires_grad = False

        # initialise running losses
        running_recloss_half1 = 0
        running_latloss_half1 = 0
        running_total_loss_half1 = 0
        var_total_loss_half1 = 0

        edge_weights_h1 = torch.ones(
            decoder_half1.radius_graph.shape[1]).to(decoder_half1.device)
        edge_weights_dis_h1 = torch.ones(
            decoder_half1.radius_graph.shape[1]).to(decoder_half1.device)
        start_time = time.time()

        # initialise the latent space and the dis
        latent_space = np.zeros([len(particle_dataset), tot_latent_dim])
        diff = np.zeros([len(particle_dataset), 1])

        # mean of displacements per-gaussian
        mean_dist_h1 = torch.zeros(decoder_half1.n_points)

        # variance of displacements per-gaussian
        displacement_variance_h1 = torch.zeros_like(mean_dist_h1)

        if epoch < n_warmup_epochs:
            decoder_half1.warmup = True

        else:
            decoder_half1.warmup = False
            decoder_half1.amp.requires_grad = True
            decoder_half1.image_smoother.B.requires_grad = True
            decoder_half1.model_positions.requires_grad = False

        # calculate regularisation parameter as moving average over epochs
        if epoch < (n_warmup_epochs+1):
            if initialization_mode == ConsensusInitializationMode.MODEL:
                lambda_regularization_half1 = 1
            else:
                lambda_regularization_half1 = 0
        else:
            previous = lambda_regularization_half1
            print('previous:', previous)
            current, Sigma, Err1 = calibrate_regularization_parameter(
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
                edge_weights_dis=edge_weights_dis_h1,
                recompute_data_normalization=True
            )
            print('current:', current)
            lambda_regularization_half1 = regularization_factor_h1 * \
                (0.9 * previous + 0.1 * current)

        print('new regularization parameter for half 1 is ',
              lambda_regularization_half1)

        # training starts!
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
            latent_weight=beta,
            regularization_parameter=lambda_regularization_half1,
            consensus_update_pooled_particles=consensus_update_pooled_particles,
            regularization_mode=regularization_mode_half1,
            edge_weights=edge_weights_h1,
            edge_weights_dis=edge_weights_dis_h1
        )

        ref_pos_h1 = update_model_positions(particle_dataset, data_preprocessor, encoder_half1,
                                            decoder_half1, shifts, angles,  idix_half1, consensus_update_pooled_particles, batch_size)

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
        if initialization_mode != ConsensusInitializationMode.MODEL:
            if epoch > (n_warmup_epochs - 1) and consensus_update_rate != 0:

                new_pos_h1 = update_model_positions(particle_dataset, data_preprocessor, encoder_half1,
                                                    decoder_half1, shifts, angles,  idix_half1, consensus_update_pooled_particles, batch_size)

                if losses_half1['reconstruction_loss'] < (old_loss_half1+old2_loss_half1)/2 and consensus_update_rate_h1 != 0:
                    # decoder_half1.model_positions = torch.nn.Parameter((
                    #    1 - consensus_update_rate_h1) * decoder_half1.model_positions + consensus_update_rate_h1 * new_pos_h1, requires_grad=True)
                    decoder_half1.model_positions = torch.nn.Parameter(
                        new_pos_h1, requires_grad=False)
                    old2_loss_half1 = old_loss_half1
                    old_loss_half1 = losses_half1['reconstruction_loss']
                    nosub_ind_h1 = 0

                if consensus_update_rate_h1 == 0:
                    regularization_factor_h1 = np.minimum(
                        regularization_factor_h1 + 1/100, 0.6)

                if losses_half1['reconstruction_loss'] > (old_loss_half1+old2_loss_half1)/2 and consensus_update_rate_h1 != 0:
                    nosub_ind_h1 += 1
                    print('No consensus updates for ',
                          nosub_ind_h1, ' epochs on half-set 1')
                    # for g in dec_half1_optimizer.param_groups:
                    #     g['lr'] *= 0.9
                    # print('new learning rate for half 1 is', g['lr'])
                    if nosub_ind_h1 == 1:
                        consensus_update_rate_h1 *= consensus_update_decay
                    if consensus_update_rate_h1 < 0.1:
                        consensus_update_rate_h1 = 0
                        # reset_all_linear_layer_weights(decoder_half1)
                        # reset_all_linear_layer_weights(encoder_half1)
                        # regularization_factor_h1 = 0
                        # regularization_mode_half1 = RegularizationMode.MODEL
                        # decoder_half1.compute_neighbour_graph()
                        # decoder_half1.compute_radius_graph()
                        # physical_parameter_optimizer_half1 = torch.optim.SGD(
                        #     decoder_half1.physical_parameters, lr=0.001*posLR)

                if consensus_update_rate_h1 == 0 and initialization_mode != ConsensusInitializationMode.MODEL:
                    final += 1
                    update_epochs = epoch
                    finalization_epochs = update_epochs//10
                    print('Optimizing the last', finalization_epochs,
                          'epochs without consensus update')

        with torch.no_grad():
            frc_half1 = losses_half1['fourier_ring_correlation'] / \
                len(dataset_half1)
            if initialization_mode != ConsensusInitializationMode.MODEL:
                decoder_half1.compute_neighbour_graph()

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
            # FF = torch.exp(-self.image_smoother.B[:, None, None,
            #                                       None] ** 2 * R) * self.image_smoother.A[
            #     :, None,
            #     None,
            #     None] ** 2
            F3 = torch.exp(-(1/decoder_half1.image_smoother.B[:, None, None,
                                                              None]**2) * R3**2) * decoder_half1.image_smoother.A[
                :, None,
                None,
                None] ** 2
            N_graph_h1 = decoder_half1.radius_graph.shape[1]
            if epoch > n_warmup_epochs:
                Sig = 0.5*Sig + 0.5*(Err1)
                data_normalization_mask = 1 / \
                    Sig
                data_normalization_mask /= torch.max(data_normalization_mask)
                R2, mask = radial_index_mask(decoder_half1.vol_box)

                data_normalization_mask *= torch.fft.fftshift(
                    mask.to(decoder_half1.device), dim=[-1, -2])

                data_normalization_mask = (data_normalization_mask /
                                           torch.sum(data_normalization_mask**2))*(box_size**2)
            print('mean distance in graph in Angstrom in half 1:',
                  decoder_half1.mean_neighbour_distance.item(), ' Angstrom')
            displacement_variance_half1 = displacement_statistics_half1[
                'displacement_variances']

            mean_dist_half1 = displacement_statistics_half1['mean_displacements']

            displacement_variance_half1 /= len(dataset_half1)
            mean_dist_half1 /= len(dataset_half1)

            if initialization_mode in (
                    ConsensusInitializationMode.EMPTY, ConsensusInitializationMode.MAP):
                decoder_half1.compute_neighbour_graph()

                decoder_half1.compute_radius_graph()
                decoder_half1.combine_graphs()

            ff2 = generate_form_factor(
                decoder_half1.image_smoother.A, decoder_half1.image_smoother.B,
                box_size)

            ind1 = torch.randint(0, box_size - 1, (1, 1))
            ind2 = torch.randint(0, box_size - 1, (1, 1))

            x = visualization_data_half1['projection_image'][0]
            yd = visualization_data_half1['target_image'][0]
            V_h1 = decoder_half1.generate_consensus_volume()

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

                else:
                    summ.add_figure("Data/latent",
                                    visualize_latent(
                                        latent_space,
                                        c=cols,
                                        s=3,
                                        alpha=0.2),
                                    epoch)

                summ.add_scalar("Loss/kld_loss",
                                (losses_half1['latent_loss'] /
                                    len(data_loader_half1)), epoch)
                summ.add_scalar("Loss/mse_loss",
                                (losses_half1['reconstruction_loss'] /
                                    len(data_loader_half1)), epoch)
                summ.add_scalar("Loss/total_loss", (
                    losses_half1['loss']) /
                    len(data_loader_half1), epoch)
                summ.add_scalar("Loss/geometric_loss",
                                (losses_half1['geometric_loss'] /
                                    len(data_loader_half1)), epoch)

                summ.add_scalar(
                    "Loss/variance_h1_1",
                    decoder_half1.image_smoother.B[0].detach().cpu(), epoch)

                summ.add_scalar(
                    "Loss/amplitude_h1",
                    decoder_half1.image_smoother.A[0].detach().cpu(), epoch)

                if decoder_half1.n_classes > 1:
                    summ.add_scalar(
                        "Loss/amplitude_h1_2",
                        decoder_half1.image_smoother.A[1].detach().cpu(), epoch)

                    summ.add_scalar(
                        "Loss/variance_h1_2",
                        decoder_half1.image_smoother.B[1].detach().cpu(), epoch)

                summ.add_figure("Loss/amplitudes_h1",
                                tensor_plot(decoder_half1.amp[0].cpu()), epoch)
                summ.add_scalar("Loss/N_graph_h1", N_graph_h1, epoch)
                summ.add_scalar("Loss/reg_param_h1",
                                lambda_regularization_half1, epoch)
                summ.add_scalar("Loss/substitute_h1",
                                consensus_update_rate_h1, epoch)
                summ.add_scalar("Loss/pose_error", angular_error, epoch)
                summ.add_scalar("Loss/trans_error", translational_error, epoch)
                summ.add_figure("Data/output", tensor_imshow(torch.fft.fftshift(
                    apply_ctf(visualization_data_half1['projection_image'][0],
                              visualization_data_half1['ctf'][0].float()).squeeze().cpu(),
                    dim=[-1, -2])), epoch)

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
                # summ.add_figure("Data/cons_points_z_half1_signal",
                #                 tensor_scatter(decoder_half1.model_positions[:, 0],
                #                                decoder_half1.model_positions[:, 1],
                #                                c=(signal1/torch.max(signal1)).cpu(), s=3), epoch)
                summ.add_figure("Data/cons_points_z_half1_noise",
                                tensor_scatter(decoder_half1.model_positions[:, 0],
                                               decoder_half1.model_positions[:, 1],
                                               c=(mean_dist_h1/torch.max(mean_dist_h1)).cpu(), s=3), epoch)

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

                summ.add_figure("Data/frc_h1", tensor_plot(frc_half1), epoch)
        epoch_t = time.time() - start_time

        if epoch % 10 == 0 or (final > 0 and epoch == (update_epochs+finalization_epochs-1)):
            with torch.no_grad():
                V_h1 = decoder_half1.generate_consensus_volume().cpu()

                gaussian_widths = torch.argmax(torch.nn.functional.softmax(
                    decoder_half1.ampvar, dim=0), dim=0)
                checkpoint = {'encoder_half1': encoder_half1,
                              'decoder_half1': decoder_half1,
                              'poses': poses,
                              'encoder_half1_state_dict': encoder_half1.state_dict(),
                              'decoder_half1_state_dict': decoder_half1.state_dict(),
                              'poses_state_dict': poses.state_dict(),
                              'enc_half1_optimizer': enc_half1_optimizer.state_dict(),
                              'cons_optimizer_half1': physical_parameter_optimizer_half1.state_dict(),
                              'dec_half1_optimizer': dec_half1_optimizer.state_dict(),
                              'indices_half1': half1_indices,
                              'refinement_directory': refinement_star_file}
                if epoch == (n_epochs - 1):
                    checkpoint_file = deformations_directory / 'checkpoints/checkpoint_final.pth'
                    torch.save(checkpoint, checkpoint_file)
                else:
                    checkpoint_file = checkpoints_directory / f'{epoch:03}.pth'
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

        if final > 0:
            if final > finalization_epochs:
                break
