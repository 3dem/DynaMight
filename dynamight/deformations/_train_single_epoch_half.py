import torch
import numpy as np
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

from ..models.constants import ConsensusInitializationMode, RegularizationMode
from ..models.losses import GeometricLoss, denoisloss
from ..utils.utils_new import frc, fourier_loss, power_spec2, count_interior_points


def train_epoch(
    encoder: torch.nn.Module,
    encoder_optimizer: torch.optim.Optimizer,
    decoder: torch.nn.Module,
    decoder_optimizer: torch.optim.Optimizer,
    physical_parameter_optimizer: torch.optim.Optimizer,
    baseline_parameter_optimizer: torch.optim.Optimizer,
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
    regularization_mode: RegularizationMode,
    edge_weights,
    edge_weights_dis,
    ref_mask=None,
    pos_epoch=False,
):

    device = decoder.device
    frc_vals = 0
    if decoder.mask == None or decoder.warmup == True:
        displacement_variance = torch.zeros(decoder.n_points)
        mean_dist = torch.zeros(decoder.n_points)
    else:
        displacement_variance = torch.zeros(decoder.n_active_points)
        mean_dist = torch.zeros(decoder.n_active_points)

    running_reconstruction_loss = 0
    running_latent_loss = 0
    running_total_loss = 0
    running_geometric_loss = 0
    dis_norm = torch.zeros(1, 1)
    ref_counts = torch.zeros(1, 1)

    geometric_loss = GeometricLoss(
        mode=regularization_mode,
        neighbour_loss_weight=0.0,
        repulsion_weight=0.01,
        outlier_weight=0.0,
        deformation_regularity_weight=1.0,
        deformation_coherence_weight=0.0

    )
    denoising_loss = torch.nn.BCELoss()

    for batch_ndx, sample in enumerate(tqdm(dataloader, file=sys.stdout)):

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

        ctf = torch.fft.fftshift(ctf, dim=[-1, -2])

        mu, logsigma = encoder(
            y_in.detach(), ctf.detach())
        # mu, logsigma, denois1, out1,  target1 = encoder(
        #    y_in.detach(), ctf.detach())
        z = mu + torch.exp(0.5 * logsigma) * torch.randn_like(mu)
        z_in = z

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
            Proj.squeeze(), y.squeeze().detach(), ctf.float().detach(),
            W=data_normalization_mask[None, :, :])
        latent_loss = -0.5 * \
            torch.mean(torch.sum(1 + logsigma - mu ** 2 -
                                 torch.exp(logsigma), dim=1),
                       dim=0)

        #d_loss = denoising_loss(out1, target1)

        if epoch < n_warmup_epochs:  # and cons_model.n_points<args.n_gauss:
            geo_loss = torch.zeros(1).to(device)

        else:
            encoder.requires_grad = True
            geo_loss = geometric_loss(
                deformed_positions=new_points,
                displacements=deformed_points,
                mean_neighbour_distance=decoder.mean_neighbour_distance,
                mean_graph_distance=decoder.mean_graph_distance,
                consensus_pairwise_distances=decoder.model_distances,
                knn=decoder.neighbour_graph,
                radius_graph=decoder.radius_graph,
                box_size=decoder.box_size,
                ang_pix=decoder.ang_pix,
                active_indices=decoder.active_indices,
                edge_weights=edge_weights,
                edge_weights_dis=edge_weights_dis
            )

        if epoch < n_warmup_epochs:
            loss = rec_loss
        else:
            if regularization_parameter > 0 and pos_epoch == False:
                loss = rec_loss + latent_weight * latent_loss + \
                    regularization_parameter * geo_loss
            else:
                loss = rec_loss + latent_weight * latent_loss

        loss.backward()

        if epoch < n_warmup_epochs:

            baseline_parameter_optimizer.step()

        elif pos_epoch == True:
            physical_parameter_optimizer.step()

        else:
            encoder.requires_grad = True
            decoder.requires_grad = True
            encoder_optimizer.step()
            decoder_optimizer.step()
            physical_parameter_optimizer.step()

        with torch.no_grad():
            val_loss = fourier_loss(
                Proj.squeeze(), y.squeeze().detach(), ctf.float().detach(),
                W=torch.ones_like(data_normalization_mask[None, :, :]))
            displacement_variance += torch.sum(
                torch.linalg.norm(deformed_points.detach().cpu(), dim=2) ** 2, dim=0
            )
            mean_dist += torch.sum(
                torch.linalg.norm(deformed_points.detach().cpu(), dim=2), dim=0
            )
            defs = torch.linalg.norm(
                torch.linalg.norm(deformed_points.detach().cpu(), dim=2), dim=1)
            if ref_mask != None:
                counts = count_interior_points(
                    new_points, ref_mask, decoder.vol_box, decoder.ang_pix).cpu()
            if ref_mask != None and batch_ndx == 0:
                ref_counts = counts
                idix = idx.cpu()
            elif ref_mask != None and ref_counts.shape[0] < consensus_update_pooled_particles:
                ref_counts = torch.cat([ref_counts, counts])
                idix = torch.cat([idix, idx.cpu()])
            if batch_ndx == 0 and ref_mask == None:
                dis_norm = defs
                idix = idx.cpu()
            elif dis_norm.shape[0] < consensus_update_pooled_particles and ref_mask == None:
                dis_norm = torch.cat([dis_norm, defs])
                idix = torch.cat([idix, idx.cpu()])

            elif ref_mask == None and dis_norm.shape[0] >= consensus_update_pooled_particles:
                dis_norm = torch.cat([dis_norm, defs])
                _, bottom_ind = torch.topk(
                    dis_norm, consensus_update_pooled_particles, largest=False)
                dis_norm = dis_norm[bottom_ind]
                idix = torch.cat([idix, idx.cpu()])
                idix = idix[bottom_ind]
            elif ref_mask != None and ref_counts.shape[0] >= consensus_update_pooled_particles:
                ref_counts = torch.cat([ref_counts, counts])
                _, bottom_ind = torch.topk(
                    ref_counts, consensus_update_pooled_particles, largest=True)
                ref_counts = ref_counts[bottom_ind]
                idix = torch.cat([idix, idx.cpu()])
                idix = idix[bottom_ind]

            latent_space[sample["idx"].cpu().numpy()] = mu.detach().cpu()
            running_reconstruction_loss += val_loss.item()
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
    # ,'denoised_image': denois1}
    displacement_statistics = {'mean_displacements': mean_dist,
                               'displacement_variances': displacement_variance}

    return latent_space, losses, displacement_statistics, idix, visualization_data


def val_epoch(
    encoder: torch.nn.Module,
    encoder_optimizer: torch.optim.Optimizer,
    decoder: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    angles: torch.nn.Parameter,
    shifts: torch.nn.Parameter,
    data_preprocessor,
    epoch,
    n_warmup_epochs,
    data_normalization_mask,
    latent_space,
    latent_weight,
    consensus_update_pooled_particles,
):

    device = decoder.device
    dis_norm = torch.tensor(np.zeros((1, 1)))
    Sig = torch.zeros(decoder.box_size, decoder.box_size).to(device)
    Err = torch.zeros_like(Sig)
    count = 0
    #mus = []
    #inds = []

    for batch_ndx, sample in enumerate(dataloader):
        encoder.requires_grad = True
        encoder_optimizer.zero_grad()

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

        mu, logsigma = encoder(y_in, ctf)
        #mu, logsigma, denois1, out1, target1 = encoder(y_in, ctf)
        z = mu + torch.exp(0.5 * logsigma) * torch.randn_like(mu)
        z_in = z

        Proj, new_points, deformed_points = decoder(
            z_in,
            r,
            shift.to(
                device))

        y = sample["image"].to(device)
        y = data_preprocessor.apply_circular_mask(y.detach())
        rec_loss = fourier_loss(
            Proj.squeeze(), y.squeeze().detach(), ctf.float(),
            W=data_normalization_mask[None, :, :])
        latent_loss = -0.5 * \
            torch.mean(torch.sum(1 + logsigma - mu ** 2 -
                                 torch.exp(logsigma), dim=1),
                       dim=0)

        loss = rec_loss + latent_weight * latent_loss

        loss.backward()
        encoder.requires_grad = True
        if epoch > n_warmup_epochs:
            encoder_optimizer.step()

        with torch.no_grad():
            yf = torch.fft.fft2(torch.fft.fftshift(
                y, dim=[-1, -2]), dim=[-1, -2], norm='ortho')
            Projf = torch.multiply(Proj, ctf)
            SR, sr = power_spec2(
                (yf-Projf), batch_reduce='mean')
            S2, s2 = power_spec2(y, batch_reduce='mean')
            count += 1
            try:
                ssig += s2
                serr += sr
            except:
                ssig = s2
                serr = sr
            Sig += S2
            Err += SR
            defs = torch.linalg.norm(
                torch.linalg.norm(deformed_points.detach().cpu(), dim=2), dim=1)
            # mus.append(mu.detach().cpu())
            # inds.append(sample['idx'].cpu())
            latent_space[sample["idx"].cpu().numpy()] = mu.detach().cpu()
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

    #mus = torch.cat(mus, 0)
    #inds = torch.cat(inds, 0)
    #latent_space[inds] = mus
    # plt.subplot(121)
    # plt.scatter(mus[:, 0], mus[:, 1], alpha=0.1)
    # plt.subplot(122)
    # plt.scatter(latent_space[inds, 0], latent_space[inds, 1], alpha=0.1)
    # plt.show()
    return latent_space, idix, Sig/count, Err/count


def validate_epoch(
    encoder_half1: torch.nn.Module,
    encoder_half2: torch.nn.Module,
    encoder_half1_optimizer: torch.optim.Optimizer,
    encoder_half2_optimizer: torch.optim.Optimizer,
    decoder_half1: torch.nn.Module,
    decoder_half2: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    angles: torch.nn.Parameter,
    shifts: torch.nn.Parameter,
    data_preprocessor,
    epoch,
    n_warmup_epochs,
    data_normalization_mask,
    latent_space,
    latent_weight,
    consensus_update_pooled_particles,
):

    device = decoder_half1.device
    dis_norm_h1 = 0
    dis_norm_h2 = 0
    noise = torch.zeros(decoder_half1.model_positions.shape[0]).to(device)
    pair_noise = torch.zeros(decoder_half1.radius_graph.shape[-1]).to(device)
    pair_signal = torch.zeros(decoder_half1.radius_graph.shape[-1]).to(device)
    Sig = torch.zeros(decoder_half1.box_size,
                      decoder_half1.box_size).to(device)
    Err_h1 = torch.zeros_like(Sig)
    Err_h2 = torch.zeros_like(Sig)
    count = 0
    consensus_pairwise_distances = torch.pow(torch.sum((decoder_half1.model_positions[
        decoder_half1.radius_graph[0, :]]-decoder_half1.model_positions[decoder_half1.radius_graph[1, :]])**2), 0.5)

    for batch_ndx, sample in enumerate(dataloader):

        encoder_half1_optimizer.zero_grad()
        encoder_half2_optimizer.zero_grad()
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

        mu_h1, logsigma_h1 = encoder_half1(y_in.detach(), ctf.detach())
        mu_h2, logsigma_h2 = encoder_half2(y_in.detach(), ctf.detach())
        # + torch.exp(0.5 * logsigma_h1) * torch.randn_like(mu_h1)
        z_h1 = mu_h1 + torch.exp(0.5 * logsigma_h1) * torch.randn_like(mu_h1)
        # + torch.exp(0.5 * logsigma_h2) * torch.randn_like(mu_h2)
        z_h2 = mu_h2 + torch.exp(0.5 * logsigma_h2) * torch.randn_like(mu_h2)
        z_in_h1 = z_h1
        z_in_h2 = z_h2

        Proj_h1, new_points_h1, deformed_points_h1 = decoder_half1(
            z_in_h1,
            r.detach(),
            shift.detach().to(
                device))

        Proj_h2, new_points_h2, deformed_points_h2 = decoder_half2(
            z_in_h2,
            r,
            shift.to(
                device))

        y = sample["image"].to(device)
        y = data_preprocessor.apply_circular_mask(y.detach())
        rec_loss_h1 = fourier_loss(
            Proj_h1.squeeze(), y.squeeze().detach(), ctf.float().detach(),
            W=data_normalization_mask[None, :, :])
        rec_loss_h2 = fourier_loss(
            Proj_h2.squeeze(), y.squeeze().detach(), ctf.float().detach(),
            W=data_normalization_mask[None, :, :])
        latent_loss_h1 = -0.5 * \
            torch.mean(torch.sum(1 + logsigma_h1 - mu_h1 ** 2 -
                                 torch.exp(logsigma_h1), dim=1),
                       dim=0)
        latent_loss_h2 = -0.5 * \
            torch.mean(torch.sum(1 + logsigma_h2 - mu_h2 ** 2 -
                                 torch.exp(logsigma_h2), dim=1),
                       dim=0)
        loss_h1 = rec_loss_h1 + latent_weight * latent_loss_h1
        loss_h2 = rec_loss_h2 + latent_weight * latent_loss_h2

        loss_h1.backward()
        loss_h2.backward()
        encoder_half1.requires_grad = True
        encoder_half2.requires_grad = True
        if epoch > n_warmup_epochs:
            encoder_half1_optimizer.step()
            encoder_half2_optimizer.step()

        with torch.no_grad():
            defs_h2 = torch.linalg.norm(
                torch.linalg.norm(deformed_points_h2.detach().cpu(), dim=2), dim=1)
            Proj_h2, new_points_h2, deformed_points_h2 = decoder_half2(
                z_in_h1,
                r,
                shift.to(
                    device),
                positions=decoder_half1.model_positions)
            pair_dis_h1 = torch.pow(torch.sum((new_points_h1[:,
                                                             decoder_half1.radius_graph[0, :]]-new_points_h1[:, decoder_half1.radius_graph[1, :]])**2, -1), 0.5)
            pair_dis_h2 = torch.pow(torch.sum((new_points_h2[:,
                                                             decoder_half1.radius_graph[0, :]]-new_points_h2[:, decoder_half1.radius_graph[1, :]])**2, -1), 0.5)
            pair_err_h1 = pair_dis_h1-consensus_pairwise_distances
            pair_err_h2 = pair_dis_h2-consensus_pairwise_distances
            pair_noise += torch.mean((pair_err_h1-pair_err_h2)**2, 0)
            pair_signal += (torch.mean(pair_err_h1**2) +
                            torch.mean(pair_err_h2**2))/2
            noise += torch.mean(torch.sum((new_points_h1 -
                                new_points_h2)**2, -1), 0)
            yf = torch.fft.fft2(torch.fft.fftshift(
                y, dim=[-1, -2]), dim=[-1, -2], norm='ortho')
            Projf_h1 = torch.multiply(Proj_h1, ctf)
            Projf_h2 = torch.multiply(Proj_h2, ctf)
            SR_h1, sr_h1 = power_spec2(
                (yf-Projf_h1), batch_reduce='mean')
            SR_h2, sr_h2 = power_spec2(
                (yf-Projf_h2), batch_reduce='mean')
            S2, s2 = power_spec2(y, batch_reduce='mean')
            count += 1
            try:
                ssig += s2
                serr_h1 += sr_h1
                serr_h2 += sr_h2
            except:
                ssig = s2
                serr_h1 = sr_h1
                serr_h2 = sr_h2
            Sig += S2
            Err_h1 += SR_h1
            Err_h2 += SR_h2
            defs_h1 = torch.linalg.norm(
                torch.linalg.norm(deformed_points_h1.detach().cpu(), dim=2), dim=1)

            if batch_ndx == 0 or dis_norm_h1.shape[0] < consensus_update_pooled_particles:
                dis_norm_h1 = defs_h1
                dis_norm_h2 = defs_h2
                idix_h1 = idx.cpu()
                idix_h2 = idx.cpu()
            else:
                dis_norm_h1 = torch.cat([dis_norm_h1, defs_h1])
                _, bottom_ind_h1 = torch.topk(
                    dis_norm_h1, consensus_update_pooled_particles, largest=False)
                _, bottom_ind_h2 = torch.topk(
                    dis_norm_h2, consensus_update_pooled_particles, largest=False)
                dis_norm_h1 = dis_norm_h1[bottom_ind_h1]
                dis_norm_h2 = dis_norm_h2[bottom_ind_h2]
                idix_h1 = torch.cat([idix_h1, idx.cpu()])
                idix_h2 = torch.cat([idix_h2, idx.cpu()])
                idix_h1 = idix_h1[bottom_ind_h1]
                idix_h2 = idix_h2[bottom_ind_h2]
            latent_space[sample["idx"].cpu().numpy()
                         ] = mu_h1.detach().cpu()

    return latent_space, idix_h1, idix_h2, Sig/count, Err_h1/count, Err_h2/count, noise/count, pair_noise/count, pair_signal/count


def get_edge_weights(
    encoder_half1: torch.nn.Module,
    encoder_half2: torch.nn.Module,
    decoder_half1: torch.nn.Module,
    decoder_half2: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    angles: torch.nn.Parameter,
    shifts: torch.nn.Parameter,
    data_preprocessor,
):

    device = decoder_half1.device
    dis_norm_h1 = 0
    dis_norm_h2 = 0
    noise = torch.zeros(decoder_half1.model_positions.shape[0]).to(device)

    mean_noise1 = torch.zeros(
        decoder_half1.model_positions.shape[0]).to(device)
    mean_noise2 = torch.zeros(
        decoder_half2.model_positions.shape[0]).to(device)

    mean_signal1 = torch.zeros(
        decoder_half1.model_positions.shape[0]).to(device)
    mean_signal2 = torch.zeros(
        decoder_half2.model_positions.shape[0]).to(device)

    mean_corr1 = torch.zeros(decoder_half1.model_positions.shape[0]).to(device)
    mean_corr2 = torch.zeros(decoder_half2.model_positions.shape[0]).to(device)

    mean_pair_corr1 = torch.zeros(
        decoder_half1.radius_graph.shape[-1]).to(device)
    mean_pair_corr2 = torch.zeros(
        decoder_half2.radius_graph.shape[-1]).to(device)

    mean_norm_11 = torch.zeros(
        decoder_half1.model_positions.shape[0]).to(device)
    mean_norm_12 = torch.zeros(
        decoder_half1.model_positions.shape[0]).to(device)
    mean_norm_21 = torch.zeros(
        decoder_half2.model_positions.shape[0]).to(device)
    mean_norm_22 = torch.zeros(
        decoder_half2.model_positions.shape[0]).to(device)
    mean_pair_signoise11 = torch.zeros(
        decoder_half1.radius_graph.shape[-1]).to(device)
    mean_pair_signoise12 = torch.zeros(
        decoder_half1.radius_graph.shape[-1]).to(device)
    mean_pair_signoise21 = torch.zeros(
        decoder_half2.radius_graph.shape[-1]).to(device)
    mean_pair_signoise22 = torch.zeros(
        decoder_half2.radius_graph.shape[-1]).to(device)
    pair_noise_h1 = torch.zeros(
        decoder_half1.radius_graph.shape[-1]).to(device)
    pair_noise_h2 = torch.zeros(
        decoder_half2.radius_graph.shape[-1]).to(device)
    pair_signal_h1 = torch.zeros(
        decoder_half1.radius_graph.shape[-1]).to(device)
    pair_signal_h2 = torch.zeros(
        decoder_half2.radius_graph.shape[-1]).to(device)
    Sig = torch.zeros(decoder_half1.box_size,
                      decoder_half1.box_size).to(device)
    disp_norm_prod1 = torch.zeros(
        decoder_half1.radius_graph.shape[-1]).to(device)
    disp_norm_prod2 = torch.zeros(
        decoder_half2.radius_graph.shape[-1]).to(device)
    mean_disp_signal1 = torch.zeros(
        decoder_half1.radius_graph.shape[-1]).to(device)
    mean_disp_signal2 = torch.zeros(
        decoder_half2.radius_graph.shape[-1]).to(device)
    mean_disp_signoise11 = torch.zeros(
        decoder_half1.radius_graph.shape[-1]).to(device)
    mean_disp_signoise21 = torch.zeros(
        decoder_half2.radius_graph.shape[-1]).to(device)
    mean_disp_signoise12 = torch.zeros(
        decoder_half1.radius_graph.shape[-1]).to(device)
    mean_disp_signoise22 = torch.zeros(
        decoder_half2.radius_graph.shape[-1]).to(device)

    Err_h1 = torch.zeros_like(Sig)
    Err_h2 = torch.zeros_like(Sig)
    count = 0
    consensus_pairwise_distances_h1 = torch.pow(torch.sum((decoder_half1.model_positions[
        decoder_half1.radius_graph[0, :]]-decoder_half1.model_positions[decoder_half1.radius_graph[1, :]])**2, -1), 0.5)
    consensus_pairwise_distances_h2 = torch.pow(torch.sum((decoder_half2.model_positions[
        decoder_half2.radius_graph[0, :]]-decoder_half2.model_positions[decoder_half2.radius_graph[1, :]])**2, -1), 0.5)
    with torch.no_grad():
        for batch_ndx, sample in enumerate(tqdm(dataloader, file=sys.stdout)):
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

            mu_h1, logsigma_h1 = encoder_half1(
                y_in.detach(), ctf.detach())
            mu_h2, logsigma_h2 = encoder_half2(
                y_in.detach(), ctf.detach())

            # mu_h1, logsigma_h1, denois1, out1, target1 = encoder_half1(
            #    y_in.detach(), ctf.detach())
            # mu_h2, logsigma_h2, denois2, out2, target2 = encoder_half2(
            #    y_in.detach(), ctf.detach())
            # + torch.exp(0.5 * logsigma_h1) * torch.randn_like(mu_h1)
            z_h1 = mu_h1
            # + torch.exp(0.5 * logsigma_h2) * torch.randn_like(mu_h2)
            z_h2 = mu_h2
            z_in_h1 = z_h1
            z_in_h2 = z_h2

            Proj_h1, new_points_h1, deformed_points_h1 = decoder_half1(
                z_in_h1,
                r.detach(),
                shift.detach().to(
                    device))

            Proj_h2, new_points_h2, deformed_points_h2 = decoder_half2(
                z_in_h2,
                r,
                shift.to(
                    device))

            Proj_h2_p1, new_points_h2_p1, deformed_points_h2_p1 = decoder_half2(
                z_in_h2,
                r,
                shift.to(
                    device),
                positions=decoder_half1.model_positions)

            Proj_h1_p2, new_points_h1_p2, deformed_points_h1_p2 = decoder_half1(
                z_in_h1,
                r,
                shift.to(
                    device),
                positions=decoder_half2.model_positions)

            # Compute distance between the points in the graph after deformation of consensus points from half 1 with both decoders, denoted by d_i^j where the lower index denotes the consensus half and the upper index denotes the decoder
            disp_norm_h11 = torch.pow(torch.sum(
                (new_points_h1[:, decoder_half1.radius_graph[0]]-new_points_h1[:, decoder_half1.radius_graph[1]])**2, -1), 0.5)            # d_1^1
            disp_norm_h12 = torch.pow(torch.sum(
                (new_points_h2_p1[:, decoder_half1.radius_graph[0]]-new_points_h2_p1[:, decoder_half1.radius_graph[1]])**2, -1), 0.5)      # d_1^2

            # Compute distance between the points in the graph after deformation of consensus points from half 2 with both decoders
            disp_norm_h21 = torch.pow(torch.sum(
                (new_points_h2[:, decoder_half2.radius_graph[0]]-new_points_h2[:, decoder_half2.radius_graph[1]])**2, -1), 0.5)            # d_2^2
            disp_norm_h22 = torch.pow(torch.sum(
                (new_points_h1_p2[:, decoder_half2.radius_graph[0]]-new_points_h1_p2[:, decoder_half2.radius_graph[1]])**2, -1), 0.5)      # d_2^1

            # Compute vectors between deformed points v_i^j where the lower index denotes the consensus half and the upper index denotes the decoder
            a1 = deformed_points_h1[:, decoder_half1.radius_graph[0]] - \
                deformed_points_h1[:,
                                   decoder_half1.radius_graph[1]]           # v_1^1
            b1 = deformed_points_h2_p1[:, decoder_half1.radius_graph[0]] - \
                deformed_points_h2_p1[:,
                                      decoder_half1.radius_graph[1]]        # v_1^2

            a2 = deformed_points_h2[:, decoder_half2.radius_graph[0]] - \
                deformed_points_h2[:,
                                   decoder_half2.radius_graph[1]]           # v_2^2
            b2 = deformed_points_h1_p2[:, decoder_half2.radius_graph[0]] - \
                deformed_points_h1_p2[:,
                                      decoder_half2.radius_graph[1]]        # v_2^1

            # Compute the correlation for vectors between points
            # <v_1^1,v_1^2>
            mean_pair_corr1 += torch.sum(torch.sum(a1*b1, -1), 0)
            # <v_2^1,v_2^2>
            mean_pair_corr2 += torch.sum(torch.sum(a2*b2, -1), 0)

            # Compute square norm of these vectors
            # |v_1^1|^2
            mean_pair_signoise11 += torch.sum(torch.sum(a1**2, -1), 0)
            # |v_1^2|^2
            mean_pair_signoise12 += torch.sum(torch.sum(b1**2, -1), 0)
            # |v_2^2|^2
            mean_pair_signoise21 += torch.sum(torch.sum(a2**2, -1), 0)
            # |v_2^1|^2
            mean_pair_signoise22 += torch.sum(torch.sum(b2**2, -1), 0)

            # Compute the correlation describing the change of distance after deformation (do points want to move apart/closer in both decoders or not)
            mean_disp_signal1 += torch.sum(1e3*(consensus_pairwise_distances_h1-disp_norm_h11)*1e3*(
                consensus_pairwise_distances_h1-disp_norm_h12), 0)                                            # <c_1-d_1^1,c_1-d_1^2>
            mean_disp_signal2 += torch.sum(1e3*(consensus_pairwise_distances_h2-disp_norm_h21)*1e3*(
                consensus_pairwise_distances_h2-disp_norm_h22), 0)                                            # <c_2-d_2^2,c_2-d_2^1>

            # Compute the squared norm of the distance changes
            mean_disp_signoise11 += torch.sum(
                (1e3*(consensus_pairwise_distances_h1-disp_norm_h11))**2, 0)    # |c_1-d_1^1|^2
            mean_disp_signoise12 += torch.sum(
                (1e3*(consensus_pairwise_distances_h1-disp_norm_h12))**2, 0)    # |c_1-d_1^2|^2
            mean_disp_signoise21 += torch.sum(
                (1e3*(consensus_pairwise_distances_h2-disp_norm_h21))**2, 0)    # |c_2-d_2^1|^2
            mean_disp_signoise22 += torch.sum(
                (1e3*(consensus_pairwise_distances_h2-disp_norm_h22))**2, 0)    # |c_2-d_2^2|^2

            # Compute correlation between displacement vectors g_i^j
            corr1 = torch.sum(deformed_points_h1 *
                              deformed_points_h2_p1, -1)     # <g_1^1,g_1^2>
            corr2 = torch.sum(deformed_points_h2 *
                              deformed_points_h1_p2, -1)     # <g_2^2,g_2^1>

            # stimate signal in deformations (do deformations at a given point coincide for the two decoders?)
            # 2<g_1^1,g_1^2>
            mean_signal1 += torch.mean(2*corr1, 0)
            # 2<g_2^2,g_2^1>
            mean_signal2 += torch.mean(2*corr2, 0)

            # Compute the squared norm power of the deformations per point
            n11 = torch.sum(deformed_points_h1**2, -1)
            n12 = torch.sum(deformed_points_h2_p1**2, -1)

            n21 = torch.sum(deformed_points_h2**2, -1)
            n22 = torch.sum(deformed_points_h1_p2**2, -1)

            # Compute the mean of the correlations and norms over all particles
            mean_corr1 += torch.mean(corr1, 0)
            mean_corr2 += torch.mean(corr2, 0)

            mean_norm_11 += torch.mean(n11, 0)
            mean_norm_12 += torch.mean(n12, 0)
            mean_norm_21 += torch.mean(n21, 0)
            mean_norm_22 += torch.mean(n22, 0)

            # Compute the square norm of the deformation differences between the two decoders
            diff1 = torch.sum(
                (deformed_points_h1-deformed_points_h2_p1)**2, -1)
            diff2 = torch.sum(
                (deformed_points_h2-deformed_points_h1_p2)**2, -1)

            # Use this es an estimate of the noise power
            mean_noise1 += torch.mean(diff1, 0)
            mean_noise2 += torch.mean(diff2, 0)

            count += 1

        # Compute the mean signal and noise for the deformation vectors itself
        mean_noise1 = mean_noise1/count
        mean_noise2 = mean_noise2/count
        mean_signal1 = mean_signal1/count
        mean_signal2 = mean_signal2/count

        # Compute the mean signal for the change of distances
        mean_disp_signal1 = mean_disp_signal1
        mean_disp_signal2 = mean_disp_signal2

        disp_norm_prod1 = torch.sqrt((mean_disp_signoise11)
                                     * (mean_disp_signoise12))
        disp_norm_prod2 = torch.sqrt((mean_disp_signoise21)
                                     * (mean_disp_signoise22))

        pair_norm_prod1 = torch.sqrt(
            (mean_pair_signoise11)*(mean_pair_signoise12))
        pair_norm_prod2 = torch.sqrt(
            (mean_pair_signoise21)*(mean_pair_signoise22))

        # Estimate the noise in the change of distances
        disp_noise1 = torch.clip(disp_norm_prod1-mean_disp_signal1, min=1e-7)
        disp_noise2 = torch.clip(disp_norm_prod2-mean_disp_signal2, min=1e-7)

        pair_noise1 = pair_norm_prod1-mean_pair_corr1
        pair_noise2 = pair_norm_prod2-mean_pair_corr2

        # compute regularization parameter for deformation smoothness of deformation
        alt_snr_edges1 = torch.clip(mean_pair_corr1/pair_noise1, min=0)
        alt_snr_edges2 = torch.clip(mean_pair_corr2/pair_noise2, min=0)

        snr_edges1 = torch.clip(mean_disp_signal1/disp_noise1, min=0)
        snr_edges2 = torch.clip(mean_disp_signal2/disp_noise2, min=0)

        norm_prod1 = torch.sqrt((mean_norm_11/count)*(mean_norm_12/count))
        norm_prod2 = torch.sqrt((mean_norm_21/count)*(mean_norm_22/count))

        den_corr1 = mean_corr1/count
        den_corr2 = mean_corr2/count

        nom1 = norm_prod1 - den_corr1
        nom2 = norm_prod2 - den_corr2

        snr1 = torch.clip(den_corr1/(nom1), min=0)
        snr2 = torch.clip(den_corr2/(nom2), min=0)

        if torch.min(den_corr1[decoder_half1.radius_graph[0, :]]+den_corr1[decoder_half1.radius_graph[1, :]]) < 0:

            den_corr1 -= 2*torch.min((den_corr1[decoder_half1.radius_graph[0, :]] +
                                      den_corr1[decoder_half1.radius_graph[1, :]]))

        if torch.min(den_corr2[decoder_half2.radius_graph[0, :]]+den_corr2[decoder_half2.radius_graph[1, :]]) < 0:

            den_corr2 -= 2*torch.min((den_corr2[decoder_half2.radius_graph[0, :]] +
                                      den_corr2[decoder_half2.radius_graph[1, :]]))

        coeff1 = (nom1[decoder_half1.radius_graph[0, :]]+nom1[decoder_half1.radius_graph[1, :]])/(
            den_corr1[decoder_half1.radius_graph[0, :]] + den_corr1[decoder_half1.radius_graph[1, :]])

        coeff2 = (nom2[decoder_half2.radius_graph[0, :]]+nom2[decoder_half2.radius_graph[1, :]])/(
            den_corr2[decoder_half2.radius_graph[0, :]] + den_corr2[decoder_half2.radius_graph[1, :]])

        # if torch.min(mean_signal1[decoder_half1.radius_graph[0, :]]+mean_signal1[decoder_half1.radius_graph[1, :]]) <= 0:
        #     coeff1 = (mean_noise1[decoder_half1.radius_graph[0, :]]+mean_noise1[decoder_half1.radius_graph[1, :]])/(torch.clip(
        #         mean_signal1[decoder_half1.radius_graph[0, :]], min=1e-7)+torch.clip(mean_signal1[decoder_half1.radius_graph[1, :]], min=1e-7))
        # else:
        #     coeff1 = (mean_noise1[decoder_half1.radius_graph[0, :]]+mean_noise1[decoder_half1.radius_graph[1, :]])/(
        #         mean_signal1[decoder_half1.radius_graph[0, :]]+mean_signal1[decoder_half1.radius_graph[1, :]])
        # if torch.min(mean_signal2[decoder_half2.radius_graph[0, :]]+mean_signal2[decoder_half2.radius_graph[1, :]]) <= 0:
        #     coeff2 = (mean_noise2[decoder_half2.radius_graph[0, :]]+mean_noise2[decoder_half2.radius_graph[1, :]])/(torch.clip(
        #         mean_signal2[decoder_half2.radius_graph[0, :]], min=1e-7)+torch.clip(mean_signal2[decoder_half2.radius_graph[1, :]], min=1e-7))
        # else:
        #     coeff2 = (mean_noise2[decoder_half2.radius_graph[0, :]]+mean_noise2[decoder_half2.radius_graph[1, :]])/(
        #         mean_signal2[decoder_half2.radius_graph[0, :]]+mean_signal2[decoder_half2.radius_graph[1, :]])

    return nom1, nom2, den_corr1, den_corr2, snr1, snr2, coeff1, coeff2, snr_edges1, snr_edges2, alt_snr_edges1, alt_snr_edges2


def get_edge_weights_mask(
    encoder_half1: torch.nn.Module,
    encoder_half2: torch.nn.Module,
    decoder_half1: torch.nn.Module,
    decoder_half2: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    angles: torch.nn.Parameter,
    shifts: torch.nn.Parameter,
    data_preprocessor,
):

    device = decoder_half1.device

    mean_noise1 = torch.zeros(
        decoder_half1.unmasked_positions.shape[0]).to(device)
    mean_noise2 = torch.zeros(
        decoder_half2.unmasked_positions.shape[0]).to(device)

    mean_signal1 = torch.zeros(
        decoder_half1.unmasked_positions.shape[0]).to(device)
    mean_signal2 = torch.zeros(
        decoder_half2.unmasked_positions.shape[0]).to(device)

    mean_corr1 = torch.zeros(
        decoder_half1.unmasked_positions.shape[0]).to(device)
    mean_corr2 = torch.zeros(
        decoder_half2.unmasked_positions.shape[0]).to(device)

    mean_pair_corr1 = torch.zeros(
        decoder_half1.radius_graph.shape[-1]).to(device)
    mean_pair_corr2 = torch.zeros(
        decoder_half2.radius_graph.shape[-1]).to(device)

    mean_norm_11 = torch.zeros(
        decoder_half1.unmasked_positions.shape[0]).to(device)
    mean_norm_12 = torch.zeros(
        decoder_half1.unmasked_positions.shape[0]).to(device)
    mean_norm_21 = torch.zeros(
        decoder_half2.unmasked_positions.shape[0]).to(device)
    mean_norm_22 = torch.zeros(
        decoder_half2.unmasked_positions.shape[0]).to(device)
    mean_pair_signoise11 = torch.zeros(
        decoder_half1.radius_graph.shape[-1]).to(device)
    mean_pair_signoise12 = torch.zeros(
        decoder_half1.radius_graph.shape[-1]).to(device)
    mean_pair_signoise21 = torch.zeros(
        decoder_half2.radius_graph.shape[-1]).to(device)
    mean_pair_signoise22 = torch.zeros(
        decoder_half2.radius_graph.shape[-1]).to(device)

    disp_norm_prod1 = torch.zeros(
        decoder_half1.radius_graph.shape[-1]).to(device)
    disp_norm_prod2 = torch.zeros(
        decoder_half2.radius_graph.shape[-1]).to(device)
    mean_disp_signal1 = torch.zeros(
        decoder_half1.radius_graph.shape[-1]).to(device)
    mean_disp_signal2 = torch.zeros(
        decoder_half2.radius_graph.shape[-1]).to(device)
    mean_disp_signoise11 = torch.zeros(
        decoder_half1.radius_graph.shape[-1]).to(device)
    mean_disp_signoise21 = torch.zeros(
        decoder_half2.radius_graph.shape[-1]).to(device)
    mean_disp_signoise12 = torch.zeros(
        decoder_half1.radius_graph.shape[-1]).to(device)
    mean_disp_signoise22 = torch.zeros(
        decoder_half2.radius_graph.shape[-1]).to(device)

    count = 0
    consensus_pairwise_distances_h1 = torch.pow(torch.sum((decoder_half1.unmasked_positions[
        decoder_half1.radius_graph[0, :]]-decoder_half1.unmasked_positions[decoder_half1.radius_graph[1, :]])**2, -1), 0.5)
    consensus_pairwise_distances_h2 = torch.pow(torch.sum((decoder_half2.unmasked_positions[
        decoder_half2.radius_graph[0, :]]-decoder_half2.unmasked_positions[decoder_half2.radius_graph[1, :]])**2, -1), 0.5)
    with torch.no_grad():
        for batch_ndx, sample in enumerate(tqdm(dataloader, file=sys.stdout)):
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

            mu_h1, logsigma_h1 = encoder_half1(
                y_in.detach(), ctf.detach())
            mu_h2, logsigma_h2 = encoder_half2(
                y_in.detach(), ctf.detach())

            # mu_h1, logsigma_h1, denois1, out1, target1 = encoder_half1(
            #    y_in.detach(), ctf.detach())
            # mu_h2, logsigma_h2, denois2, out2, target2 = encoder_half2(
            #    y_in.detach(), ctf.detach())
            # + torch.exp(0.5 * logsigma_h1) * torch.randn_like(mu_h1)
            z_h1 = mu_h1
            # + torch.exp(0.5 * logsigma_h2) * torch.randn_like(mu_h2)
            z_h2 = mu_h2
            z_in_h1 = z_h1
            z_in_h2 = z_h2

            Proj_h1, new_points_h1, deformed_points_h1 = decoder_half1(
                z_in_h1,
                r.detach(),
                shift.detach().to(
                    device), positions=decoder_half1.unmasked_positions)

            Proj_h2, new_points_h2, deformed_points_h2 = decoder_half2(
                z_in_h2,
                r,
                shift.to(
                    device), positions=decoder_half2.unmasked_positions)

            Proj_h2_p1, new_points_h2_p1, deformed_points_h2_p1 = decoder_half2(
                z_in_h2,
                r,
                shift.to(
                    device),
                positions=decoder_half1.unmasked_positions)

            Proj_h1_p2, new_points_h1_p2, deformed_points_h1_p2 = decoder_half1(
                z_in_h1,
                r,
                shift.to(
                    device),
                positions=decoder_half2.unmasked_positions)

            # Compute distance between the points in the graph after deformation of consensus points from half 1 with both decoders, denoted by d_i^j where the lower index denotes the consensus half and the upper index denotes the decoder
            disp_norm_h11 = torch.pow(torch.sum(
                (new_points_h1[:, decoder_half1.radius_graph[0]]-new_points_h1[:, decoder_half1.radius_graph[1]])**2, -1), 0.5)            # d_1^1
            disp_norm_h12 = torch.pow(torch.sum(
                (new_points_h2_p1[:, decoder_half1.radius_graph[0]]-new_points_h2_p1[:, decoder_half1.radius_graph[1]])**2, -1), 0.5)      # d_1^2

            # Compute distance between the points in the graph after deformation of consensus points from half 2 with both decoders
            disp_norm_h21 = torch.pow(torch.sum(
                (new_points_h2[:, decoder_half2.radius_graph[0]]-new_points_h2[:, decoder_half2.radius_graph[1]])**2, -1), 0.5)            # d_2^2
            disp_norm_h22 = torch.pow(torch.sum(
                (new_points_h1_p2[:, decoder_half2.radius_graph[0]]-new_points_h1_p2[:, decoder_half2.radius_graph[1]])**2, -1), 0.5)      # d_2^1

            # Compute vectors between deformed points v_i^j where the lower index denotes the consensus half and the upper index denotes the decoder
            a1 = deformed_points_h1[:, decoder_half1.radius_graph[0]] - \
                deformed_points_h1[:,
                                   decoder_half1.radius_graph[1]]           # v_1^1
            b1 = deformed_points_h2_p1[:, decoder_half1.radius_graph[0]] - \
                deformed_points_h2_p1[:,
                                      decoder_half1.radius_graph[1]]        # v_1^2

            a2 = deformed_points_h2[:, decoder_half2.radius_graph[0]] - \
                deformed_points_h2[:,
                                   decoder_half2.radius_graph[1]]           # v_2^2
            b2 = deformed_points_h1_p2[:, decoder_half2.radius_graph[0]] - \
                deformed_points_h1_p2[:,
                                      decoder_half2.radius_graph[1]]        # v_2^1

            # Compute the correlation for vectors between points
            # <v_1^1,v_1^2>
            mean_pair_corr1 += torch.sum(torch.sum(a1*b1, -1), 0)
            # <v_2^1,v_2^2>
            mean_pair_corr2 += torch.sum(torch.sum(a2*b2, -1), 0)

            # Compute square norm of these vectors
            # |v_1^1|^2
            mean_pair_signoise11 += torch.sum(torch.sum(a1**2, -1), 0)
            # |v_1^2|^2
            mean_pair_signoise12 += torch.sum(torch.sum(b1**2, -1), 0)
            # |v_2^2|^2
            mean_pair_signoise21 += torch.sum(torch.sum(a2**2, -1), 0)
            # |v_2^1|^2
            mean_pair_signoise22 += torch.sum(torch.sum(b2**2, -1), 0)

            # Compute the correlation describing the change of distance after deformation (do points want to move apart/closer in both decoders or not)
            mean_disp_signal1 += torch.sum(1e3*(consensus_pairwise_distances_h1-disp_norm_h11)*1e3*(
                consensus_pairwise_distances_h1-disp_norm_h12), 0)                                            # <c_1-d_1^1,c_1-d_1^2>
            mean_disp_signal2 += torch.sum(1e3*(consensus_pairwise_distances_h2-disp_norm_h21)*1e3*(
                consensus_pairwise_distances_h2-disp_norm_h22), 0)                                            # <c_2-d_2^2,c_2-d_2^1>

            # Compute the squared norm of the distance changes
            mean_disp_signoise11 += torch.sum(
                (1e3*(consensus_pairwise_distances_h1-disp_norm_h11))**2, 0)    # |c_1-d_1^1|^2
            mean_disp_signoise12 += torch.sum(
                (1e3*(consensus_pairwise_distances_h1-disp_norm_h12))**2, 0)    # |c_1-d_1^2|^2
            mean_disp_signoise21 += torch.sum(
                (1e3*(consensus_pairwise_distances_h2-disp_norm_h21))**2, 0)    # |c_2-d_2^1|^2
            mean_disp_signoise22 += torch.sum(
                (1e3*(consensus_pairwise_distances_h2-disp_norm_h22))**2, 0)    # |c_2-d_2^2|^2

            # Compute correlation between displacement vectors g_i^j
            corr1 = torch.sum(deformed_points_h1 *
                              deformed_points_h2_p1, -1)     # <g_1^1,g_1^2>
            corr2 = torch.sum(deformed_points_h2 *
                              deformed_points_h1_p2, -1)     # <g_2^2,g_2^1>

            # stimate signal in deformations (do deformations at a given point coincide for the two decoders?)
            # 2<g_1^1,g_1^2>
            mean_signal1 += torch.mean(2*corr1, 0)
            # 2<g_2^2,g_2^1>
            mean_signal2 += torch.mean(2*corr2, 0)

            # Compute the squared norm power of the deformations per point
            n11 = torch.sum(deformed_points_h1**2, -1)
            n12 = torch.sum(deformed_points_h2_p1**2, -1)

            n21 = torch.sum(deformed_points_h2**2, -1)
            n22 = torch.sum(deformed_points_h1_p2**2, -1)

            # Compute the mean of the correlations and norms over all particles
            mean_corr1 += torch.mean(corr1, 0)
            mean_corr2 += torch.mean(corr2, 0)

            mean_norm_11 += torch.mean(n11, 0)
            mean_norm_12 += torch.mean(n12, 0)
            mean_norm_21 += torch.mean(n21, 0)
            mean_norm_22 += torch.mean(n22, 0)

            # Compute the square norm of the deformation differences between the two decoders
            diff1 = torch.sum(
                (deformed_points_h1-deformed_points_h2_p1)**2, -1)
            diff2 = torch.sum(
                (deformed_points_h2-deformed_points_h1_p2)**2, -1)

            # Use this es an estimate of the noise power
            mean_noise1 += torch.mean(diff1, 0)
            mean_noise2 += torch.mean(diff2, 0)

            count += 1

        # Compute the mean signal and noise for the deformation vectors itself
        mean_noise1 = mean_noise1/count
        mean_noise2 = mean_noise2/count
        mean_signal1 = mean_signal1/count
        mean_signal2 = mean_signal2/count

        # Compute the mean signal for the change of distances
        mean_disp_signal1 = mean_disp_signal1
        mean_disp_signal2 = mean_disp_signal2

        disp_norm_prod1 = torch.sqrt((mean_disp_signoise11)
                                     * (mean_disp_signoise12))
        disp_norm_prod2 = torch.sqrt((mean_disp_signoise21)
                                     * (mean_disp_signoise22))

        pair_norm_prod1 = torch.sqrt(
            (mean_pair_signoise11)*(mean_pair_signoise12))
        pair_norm_prod2 = torch.sqrt(
            (mean_pair_signoise21)*(mean_pair_signoise22))

        # Estimate the noise in the change of distances
        disp_noise1 = torch.clip(disp_norm_prod1-mean_disp_signal1, min=1e-7)
        disp_noise2 = torch.clip(disp_norm_prod2-mean_disp_signal2, min=1e-7)

        pair_noise1 = pair_norm_prod1-mean_pair_corr1
        pair_noise2 = pair_norm_prod2-mean_pair_corr2

        # compute regularization parameter for deformation smoothness of deformation
        alt_snr_edges1 = torch.clip(mean_pair_corr1/pair_noise1, min=0)
        alt_snr_edges2 = torch.clip(mean_pair_corr2/pair_noise2, min=0)

        snr_edges1 = torch.clip(mean_disp_signal1/disp_noise1, min=0)
        snr_edges2 = torch.clip(mean_disp_signal2/disp_noise2, min=0)

        norm_prod1 = torch.sqrt((mean_norm_11/count)*(mean_norm_12/count))
        norm_prod2 = torch.sqrt((mean_norm_21/count)*(mean_norm_22/count))

        den_corr1 = mean_corr1/count
        den_corr2 = mean_corr2/count

        nom1 = norm_prod1 - den_corr1
        nom2 = norm_prod2 - den_corr2

        snr1 = torch.clip(den_corr1/(nom1), min=0)
        snr2 = torch.clip(den_corr2/(nom2), min=0)

        if torch.min(den_corr1[decoder_half1.radius_graph[0, :]]+den_corr1[decoder_half1.radius_graph[1, :]]) < 0:

            den_corr1 -= 2*torch.min((den_corr1[decoder_half1.radius_graph[0, :]] +
                                      den_corr1[decoder_half1.radius_graph[1, :]]))

        if torch.min(den_corr2[decoder_half2.radius_graph[0, :]]+den_corr2[decoder_half2.radius_graph[1, :]]) < 0:

            den_corr2 -= 2*torch.min((den_corr2[decoder_half2.radius_graph[0, :]] +
                                      den_corr2[decoder_half2.radius_graph[1, :]]))

        coeff1 = (nom1[decoder_half1.radius_graph[0, :]]+nom1[decoder_half1.radius_graph[1, :]])/(
            den_corr1[decoder_half1.radius_graph[0, :]] + den_corr1[decoder_half1.radius_graph[1, :]])

        coeff2 = (nom2[decoder_half2.radius_graph[0, :]]+nom2[decoder_half2.radius_graph[1, :]])/(
            den_corr2[decoder_half2.radius_graph[0, :]] + den_corr2[decoder_half2.radius_graph[1, :]])

        # if torch.min(mean_signal1[decoder_half1.radius_graph[0, :]]+mean_signal1[decoder_half1.radius_graph[1, :]]) <= 0:
        #     coeff1 = (mean_noise1[decoder_half1.radius_graph[0, :]]+mean_noise1[decoder_half1.radius_graph[1, :]])/(torch.clip(
        #         mean_signal1[decoder_half1.radius_graph[0, :]], min=1e-7)+torch.clip(mean_signal1[decoder_half1.radius_graph[1, :]], min=1e-7))
        # else:
        #     coeff1 = (mean_noise1[decoder_half1.radius_graph[0, :]]+mean_noise1[decoder_half1.radius_graph[1, :]])/(
        #         mean_signal1[decoder_half1.radius_graph[0, :]]+mean_signal1[decoder_half1.radius_graph[1, :]])
        # if torch.min(mean_signal2[decoder_half2.radius_graph[0, :]]+mean_signal2[decoder_half2.radius_graph[1, :]]) <= 0:
        #     coeff2 = (mean_noise2[decoder_half2.radius_graph[0, :]]+mean_noise2[decoder_half2.radius_graph[1, :]])/(torch.clip(
        #         mean_signal2[decoder_half2.radius_graph[0, :]], min=1e-7)+torch.clip(mean_signal2[decoder_half2.radius_graph[1, :]], min=1e-7))
        # else:
        #     coeff2 = (mean_noise2[decoder_half2.radius_graph[0, :]]+mean_noise2[decoder_half2.radius_graph[1, :]])/(
        #         mean_signal2[decoder_half2.radius_graph[0, :]]+mean_signal2[decoder_half2.radius_graph[1, :]])

    return nom1, nom2, den_corr1, den_corr2, snr1, snr2, coeff1, coeff2, snr_edges1, snr_edges2, alt_snr_edges1, alt_snr_edges2
