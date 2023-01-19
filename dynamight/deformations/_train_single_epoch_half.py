import torch
import numpy as np

from ..models.constants import ConsensusInitializationMode
from ..models.losses import GeometricLoss
from ..utils.utils_new import frc, fourier_loss

from tqdm import tqdm


def train_epoch(
    encoder: torch.nn.Module,
    encoder_optimizer: torch.optim.Optimizer,
    decoder: torch.nn.Module,
    decoder_optimizer: torch.optim.Optimizer,
    physical_parameter_optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    angles: torch.nn.Parameter,
    shifts: torch.nn.Parameter,
    data_preprocessor,
    epoch,
    n_warmup_epochs,
    data_normalization_mask,
    latent_space,
    latent_weight,
    regularization_parameter,
    consensus_update_pooled_particles,
    mode: ConsensusInitializationMode,
):
    # todo: schwab implement and substitute in optimize deformations
    device = decoder.device
    frc_vals = 0
    displacement_variance = torch.zeros(decoder.n_points)
    mean_dist = torch.zeros(decoder.n_points)
    running_reconstruction_loss = 0
    running_latent_loss = 0
    running_total_loss = 0
    running_geometric_loss = 0
    dis_norm = 0

    geometric_loss = GeometricLoss(
        mode=mode,
        neighbour_loss_weight=0.01,
        repulsion_weight=0.01,
        outlier_weight=1,
        deformation_regularity_weight=1,
    )

    for batch_ndx, sample in tqdm(enumerate(dataloader)):
        if batch_ndx % dataloader.batch_size == 0:
            print('Processing batch', batch_ndx / dataloader.batch_size, 'of',
                  int(np.ceil(len(dataloader) / dataloader.batch_size)),
                  ' from half 1')

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        physical_parameter_optimizer.zero_grad()

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

        mu, logsigma = encoder(y_in, ctf)
        z = mu + torch.exp(0.5 * logsigma) * torch.randn_like(mu)
        z_in = z

        if epoch < n_warmup_epochs:  # Set latent code for consensus reconstruction to zero
            Proj, new_points, deformed_points = decoder(
                z_in,
                r,
                shift.to(
                    device))
        else:
            Proj, new_points, deformed_points = decoder(
                z_in,
                r,
                shift.to(
                    device))

            with torch.no_grad():
                try:
                    frc_vals += frc(Proj, y, ctf)
                except:
                    frc_vals = frc(Proj, y, ctf)

        y = sample["image"].to(device)
        y = data_preprocessor.apply_circular_mask(y.detach())
        rec_loss = fourier_loss(
            Proj.squeeze(), y.squeeze(), ctf.float(),
            W=data_normalization_mask[None, :, :])
        latent_loss = -0.5 * \
            torch.mean(torch.sum(1 + logsigma - mu ** 2 -
                                 torch.exp(logsigma), dim=1),
                       dim=0)

        if epoch < n_warmup_epochs:  # and cons_model.n_points<args.n_gauss:
            geo_loss = torch.zeros(1).to(device)

        else:
            encoder.requires_grad = True
            geo_loss = geometric_loss(
                deformed_positions=new_points,
                mean_neighbour_distance=decoder.mean_neighbour_distance,
                consensus_pairwise_distances=decoder.model_distances,
                knn=decoder.neighbour_graph,
                radius_graph=decoder.radius_graph,
                box_size=decoder.box_size,
                ang_pix=decoder.ang_pix
            )

        if epoch < n_warmup_epochs:
            loss = rec_loss + latent_weight * latent_loss
        else:
            loss = rec_loss + latent_weight * latent_loss + \
                regularization_parameter * geo_loss

        loss.backward()
        if epoch < n_warmup_epochs:
            physical_parameter_optimizer.step()

        else:
            encoder.requires_grad = True
            decoder.requires_grad = True
            decoder.requires_grad = True
            encoder_optimizer.step()
            decoder_optimizer.step()
            physical_parameter_optimizer.step()

        with torch.no_grad():
            displacement_variance += torch.sum(
                torch.linalg.norm(deformed_points.detach().cpu(), dim=2) ** 2, dim=0
            )
            mean_dist += torch.sum(
                torch.linalg.norm(deformed_points.detach().cpu(), dim=2), dim=0
            )
            defs = torch.linalg.norm(
                torch.linalg.norm(deformed_points.detach().cpu(), dim=2), dim=1)
            if batch_ndx == 0 or dis_norm.shape[0] < consensus_update_pooled_particles:
                dis_norm = defs
                idix = idx.cpu()
            else:
                dis_norm = torch.cat([dis_norm, defs])
                _, bottom_ind = torch.topk(
                    dis_norm, consensus_update_pooled_particles, largest=False)
                dis_norm = dis_norm[bottom_ind]
                idix = torch.cat([idix, idx.cpu()])
                idix = idix[bottom_ind]
            latent_space[sample["idx"].cpu().numpy()] = mu.detach().cpu()
            running_reconstruction_loss += rec_loss.item()
            running_latent_loss += latent_loss.item()
            running_total_loss += loss.item()
            running_geometric_loss += geo_loss.item()

        losses = {'loss': running_total_loss,
                  'reconstruction_loss': running_reconstruction_loss,
                  'latent_loss': running_latent_loss,
                  'geometric_loss': running_geometric_loss,
                  'fourier_ring_correlation': frc_vals}
        visualization_data = {'projection_image': Proj, 'input_image': y_in,
                              'target_image': y,
                              'ctf': ctf, 'deformed_points': new_points,
                              'displacements': deformed_points}
        displacement_statistics = {'mean_displacements': mean_dist,
                                   'displacement_variances': displacement_variance}

    return latent_space, losses, displacement_statistics, idix, visualization_data
