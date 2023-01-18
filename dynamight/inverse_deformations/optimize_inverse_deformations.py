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

    backward_deformations_directory = output_directory / 'bwd_deformations'
    backward_deformations_directory.mkdir(exist_ok=True, parents=True)
    forward_deformations_directory = output_directory / 'forward_deformations'
    if not forward_deformations_directory.exists():
        raise NotADirectoryError(f'{forward_deformations_directory} does not exist.')
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

    n_classes = decoder_half1.n_classes
    n_points = decoder_half1.n_points

    points = decoder_half1.model_positions.detach().cpu()
    points = torch.tensor(points)
    cons_model.pos = torch.nn.Parameter(points, requires_grad=False)
    cons_model.n_points = len(points)
    decoder_half1.n_points = cons_model.n_points
    decoder_half2.n_points = cons_model.n_points
    cons_model.ampvar = torch.nn.Parameter(
        torch.ones(n_classes, decoder_half1.n_points).to(device),
        requires_grad=False)

    cons_model.p2i.device = device
    decoder_half1.p2i.device = device
    decoder_half2.p2i.device = device
    cons_model.projector.device = device
    decoder_half1.projector.device = device
    decoder_half2.projector.device = device
    cons_model.image_smoother.device = device
    decoder_half1.image_smoother.device = device
    decoder_half2.image_smoother.device = device
    cons_model.p2v.device = device
    decoder_half1.p2v.device = device
    decoder_half2.p2v.device = device
    decoder_half1.device = device
    decoder_half2.device = device
    cons_model.device = device
    decoder_half1.to(device)
    decoder_half2.to(device)
    cons_model.to(device)

    encoder_half1.to(device)
    encoder_half2.to(device)

    latent_dim = encoder_half1.latent_dim

    dataset, diameter_ang, box_size, ang_pix, optics_group = initialize_dataset(
        refinement_star_file, mask_soft_edge_width, preload_images,
        particle_diameter)

    max_diameter_ang = box_size * optics_group[
        'pixel_size'] - mask_soft_edge_width

    if particle_diameter is None:
        diameter_ang = box_size * 1 * optics_group[
            'pixel_size'] - mask_soft_edge_width
        print(f"Assigning a diameter of {round(diameter_ang)} angstrom")
    else:
        if particle_diameter > max_diameter_ang:
            print(
                f"WARNING: Specified particle diameter {round(particle_diameter)} angstrom is too large\n"
                f" Assigning a diameter of {round(max_diameter_ang)} angstrom"
            )
            diameter_ang = max_diameter_ang
        else:
            diameter_ang = particle_diameter

    if preload_images:
        dataset.preload_images()

    inds_half1 = checkpoint['indices_half1'].cpu().numpy()
    inds_half2 = list(set(range(len(dataset))) - set(list(inds_half1)))
    print(len(inds_half1))
    print(len(inds_half2))
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
        circular_mask_radius=diameter_ang / (2 * optics_group['pixel_size']),
        circular_mask_thickness=mask_soft_edge_width / optics_group[
            'pixel_size']
    )

    box_size = optics_group['image_size']
    ang_pix = optics_group['pixel_size']

    inv_half1 = InverseDisplacementDecoder(device, latent_dim, n_points, 6, 96,
                                           LinearBlock, 6, box_size).to(device)
    inv_half2 = InverseDisplacementDecoder(device, latent_dim, n_points, 6, 96,
                                           LinearBlock, 6, box_size).to(device)
    inv_half1_params = inv_half1.parameters()
    inv_half2_params = inv_half2.parameters()
    inv_half1_params = add_weight_decay_to_named_parameters(inv_half1, decay=0.3)
    inv_half2_params = add_weight_decay_to_named_parameters(inv_half2, decay=0.3)
    inv_half1_optimizer = torch.optim.Adam(inv_half1_params, lr=5e-4)
    inv_half2_optimizer = torch.optim.Adam(inv_half2_params, lr=5e-4)

    N_inv = n_epochs
    inv_loss_h1 = 0
    inv_loss_h2 = 0

    losslist = torch.zeros(N_inv, 2)

    inds_half1 = checkpoint['indices_half1'].cpu().numpy()
    inds_half2 = list(set(range(len(dataset))) - set(list(inds_half1)))
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

    mu_in_tot = torch.zeros(len(dataset), latent_dim)
    pos_tot = torch.zeros(len(dataset), n_points, 3)

    for epochs in range(N_inv):
        print('Inversion loss for half 1 at iteration', epochs, inv_loss_h1)
        inv_loss_h1 = 0
        print('Inversion loss for half 2 at iteration', epochs, inv_loss_h2)
        inv_loss_h2 = 0
        for batch_ndx, sample in enumerate(data_loader_half1):
            inv_half1_optimizer.zero_grad()
            if epochs > 0 and add_noise == 'False':
                # print('FUCK YOU!')
                idx = sample['idx']
                c_pos = inv_half1([mu_in_tot[idx].to(device)],
                                  pos_tot[idx].to(device))
            else:
                with torch.no_grad():
                    r, y, ctf = sample["rotation"], sample["image"], sample[
                        "ctf"]
                    idx = sample['idx']
                    r, t = poses(idx)
                    batch_size = y.shape[0]
                    ctfs = torch.fft.fftshift(ctf, dim=[-1, -2])
                    y_in = data_preprocessor.apply_square_mask(y)
                    y_in = data_preprocessor.apply_translation(y_in, -t[:, 0],
                                                               -t[:, 1])
                    y_in = data_preprocessor.apply_circular_mask(y_in)
                    mu, sig = encoder_half1(y_in.to(device), ctfs.to(device))
                    mu_in = [mu]
                    if add_noise == 'True':
                        noise_real = (5 / (
                            ang_pix * box_size)) * torch.randn_like(
                            cons_model.pos).to(device)
                        evalpos = cons_model.pos + noise_real
                        proj, proj_im, proj_pos, pos, dis = decoder_half1.forward(
                            mu_in, r.to(device), evalpos.to(device),
                            torch.ones_like(evalpos[:, 0]),
                            torch.ones(n_classes, evalpos.shape[0]).to(device),
                            t.to(device))
                    else:
                        proj, proj_im, proj_pos, pos, dis = decoder_half1.forward(
                            mu_in, r.to(device), cons_model.pos.to(device),
                            cons_model.amp.to(device),
                            cons_model.ampvar.to(device), t.to(device))
                        mu_in_tot[idx] = mu.cpu()
                        pos_tot[idx] = pos.cpu()
                c_pos = inv_half1(mu_in, pos)

            if add_noise == 'True':
                loss = torch.sum(
                    (c_pos - cons_model.pos.to(device) + noise_real) ** 2)
            else:
                loss = torch.sum((c_pos - cons_model.pos.to(device)) ** 2)
            loss.backward()
            inv_half1_optimizer.step()
            inv_loss_h1 += loss.item()

        losslist[epochs, 0] = inv_loss_h1

        for batch_ndx, sample in enumerate(data_loader_half2):
            inv_half2_optimizer.zero_grad()
            if epochs > 0 and add_noise == 'False':
                idx = sample['idx']
                c_pos = inv_half2([mu_in_tot[idx].to(device)],
                                  pos_tot[idx].to(device))
            else:
                with torch.no_grad():
                    r, y, ctf = sample["rotation"], sample["image"], sample[
                        "ctf"]
                    idx = sample['idx']
                    r, t = poses(idx)
                    batch_size = y.shape[0]
                    ctfs = torch.fft.fftshift(ctf, dim=[-1, -2])
                    y_in = data_preprocessor.apply_square_mask(y)
                    y_in = data_preprocessor.apply_translation(y_in, -t[:, 0],
                                                               -t[:, 1])
                    y_in = data_preprocessor.apply_circular_mask(y_in)
                    mu, sig = encoder_half2(y_in.to(device), ctfs.to(device))
                    mu_in = [mu]
                    if add_noise == 'True':
                        noise_real = (5 / (
                            ang_pix * box_size)) * torch.randn_like(
                            cons_model.pos).to(device)

                        evalpos = cons_model.pos + noise_real
                        proj, proj_im, proj_pos, pos, dis = decoder_half1.forward(
                            mu_in, r.to(device), evalpos.to(device),
                            torch.ones_like(evalpos[:, 0]),
                            torch.ones(n_classes, evalpos.shape[0]).to(device),
                            t.to(device))
                    else:
                        proj, proj_im, proj_pos, pos, dis = decoder_half2.forward(
                            mu_in, r.to(device), cons_model.pos.to(device),
                            cons_model.amp.to(device),
                            cons_model.ampvar.to(device), t.to(device))
                        mu_in_tot[idx] = mu.cpu()
                        pos_tot[idx] = pos.cpu()
                c_pos = inv_half2(mu_in, pos)
            if add_noise == 'True':
                loss = torch.sum(
                    (c_pos - cons_model.pos.to(device) + noise_real) ** 2)
            else:
                loss = torch.sum((c_pos - cons_model.pos.to(device)) ** 2)
            loss.backward()
            inv_half2_optimizer.step()
            inv_loss_h2 += loss.item()
        losslist[epochs, 1] = inv_loss_h2
    del mu_in_tot, pos_tot

    checkpoint = {'inv_half1': inv_half1, 'inv_half2': inv_half2,
                  'inv_half1_state_dict': inv_half1.state_dict(),
                  'inv_half2_state_dict': inv_half2.state_dict()}
    torch.save(checkpoint, str(output_directory) +
               '/bwd_deformations/inv_chkpt.pth')
