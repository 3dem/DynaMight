from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from typer import Option
import os

from ..data.handlers.particle_image_preprocessor import \
    ParticleImagePreprocessor
from ..models.blocks import LinearBlock
from ..models.decoder import InverseDisplacementDecoder
from ..utils.utils_new import initialize_dataset, add_weight_decay_to_named_parameters
from ..data.dataloaders.relion import RelionDataset
from ._optimize_single_epoch import optimize_epoch
from tqdm import tqdm
from .._cli import cli


@cli.command()
def optimize_inverse_deformations(
    output_directory: Path,
    refinement_star_file: Optional[Path] = None,
    vae_checkpoint: Optional[Path] = None,
    batch_size: int = Option(100),
    n_epochs: int = Option(50),
    gpu_id: Optional[int] = Option(0),
    preload_images: bool = Option(False),
    add_noise: bool = Option(False),
    particle_diameter: Optional[float] = Option(None),
    mask_soft_edge_width: int = Option(20),
    data_loader_threads: int = Option(4),
):

    backward_deformations_directory = output_directory / 'inverse_deformations'
    backward_deformations_directory.mkdir(exist_ok=True, parents=True)
    forward_deformations_directory = output_directory / \
        'forward_deformations' / 'checkpoints'
    if not forward_deformations_directory.exists():
        raise NotADirectoryError(
            f'{forward_deformations_directory} does not exist. Please run dynamight optimize-deformations or use a checkpoint file')
    device = 'cuda:' + str(gpu_id)
    if vae_checkpoint is None:
        vae_checkpoint = forward_deformations_directory / 'checkpoint_final.pth'

    checkpoint = torch.load(vae_checkpoint, map_location=device)

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

    if preload_images:
        particle_dataset.preload_images()

    inds_half1 = checkpoint['indices_half1'].cpu().numpy()
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

    inv_half1 = InverseDisplacementDecoder(device, latent_dim, n_points, 6, 96,
                                           LinearBlock, 6, box_size).to(device)
    inv_half2 = InverseDisplacementDecoder(device, latent_dim, n_points, 6, 96,
                                           LinearBlock, 6, box_size).to(device)
    inv_half1_params = inv_half1.parameters()
    inv_half2_params = inv_half2.parameters()
    inv_half1_params = add_weight_decay_to_named_parameters(
        inv_half1, weight_decay=0.3)
    inv_half2_params = add_weight_decay_to_named_parameters(
        inv_half2, weight_decay=0.3)
    inv_half1_optimizer = torch.optim.Adam(inv_half1_params, lr=5e-4)
    inv_half2_optimizer = torch.optim.Adam(inv_half2_params, lr=5e-4)

    N_inv = n_epochs

    latent_space = torch.zeros(len(particle_dataset), latent_dim)
    deformed_positions = torch.zeros(
        len(particle_dataset), n_points, 3)

    for epoch in tqdm(range(N_inv)):
        inv_loss_h1, latent_space, deformed_positions = optimize_epoch(
            encoder_half1,
            decoder_half1,
            inv_half1,
            inv_half1_optimizer,
            data_loader_half1,
            poses,
            data_preprocessor,
            epoch,
            add_noise,
            latent_space,
            deformed_positions
        )
        inv_loss_h2, latent_space, deformed_positions = optimize_epoch(
            encoder_half2,
            decoder_half2,
            inv_half2,
            inv_half2_optimizer,
            data_loader_half2,
            poses,
            data_preprocessor,
            epoch,
            add_noise,
            latent_space,
            deformed_positions
        )

    print('Inversion loss for half 1 at iteration', epoch, inv_loss_h1)
    print('Inversion loss for half 2 at iteration', epoch, inv_loss_h2)

    checkpoint = {'inv_half1': inv_half1, 'inv_half2': inv_half2,
                  'inv_half1_state_dict': inv_half1.state_dict(),
                  'inv_half2_state_dict': inv_half2.state_dict()}
    torch.save(checkpoint, str(output_directory) +
               '/inverse_deformations/inv_chkpt.pth')
