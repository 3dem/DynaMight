import torch
from torch.utils.data import DataLoader

from ..models.constants import ConsensusInitializationMode, RegularizationMode
from ..models.encoder import HetEncoder
from ..models.decoder import DisplacementDecoder
from ..data.handlers.particle_image_preprocessor import ParticleImagePreprocessor
from ..models.losses import GeometricLoss
from ..utils.utils_new import fourier_loss


def calibrate_regularization_parameter(
    dataset: torch.utils.data.Dataset,
    data_preprocessor: ParticleImagePreprocessor,
    encoder: HetEncoder,
    decoder: DisplacementDecoder,
    particle_shifts: torch.nn.Parameter,
    particle_euler_angles: torch.nn.Parameter,
    data_normalization_mask: torch.Tensor,
    lambda_regularization: torch.Tensor,
    mode: RegularizationMode,
    subset_percentage: float = 10,
    batch_size: int = 100,
):
    """Compute a regularisation parameter for the geometry regularisation function.

    Parameters
    ----------
    data_normalization_mask
    dataset: torch.utils.data.Dataset,
        half set of data from which a subset will be taken.
    data_preprocessor: ParticleImagePreprocessor
        preprocessor for data.
    encoder: HetEncoder
        encoder for the half set
    decoder: DisplacementDecoder
        decoder for the half set

    Returns
    -------
    lambda: float
    """

    # if lambda_regularization == 0:  # Fix for first epoch after warmup
    #    lambda_regularization = 1

    n_particles = round(len(dataset) * (subset_percentage / 100))
    particle_idx = torch.randint(0, len(dataset), (n_particles,))
    subset = torch.utils.data.Subset(dataset, particle_idx)
    dataloader = DataLoader(
        dataset=subset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=False
    )
    geometry_norm = _compute_geometry_norm(
        dataloader=dataloader,
        data_preprocessor=data_preprocessor,
        encoder=encoder,
        decoder=decoder,
        particle_euler_angles=particle_euler_angles,
        particle_shifts=particle_shifts,
        lambda_regularization=lambda_regularization,
        mode=mode,
    )
    data_norm = _compute_data_norm(
        dataloader=dataloader,
        data_preprocessor=data_preprocessor,
        encoder=encoder,
        decoder=decoder,
        particle_euler_angles=particle_euler_angles,
        particle_shifts=particle_shifts,
        data_normalization_mask=data_normalization_mask,
    )
    print('data_norm:', data_norm)
    print('geometry_norm:', geometry_norm)
    return (0.5 * (data_norm / geometry_norm))


def _compute_data_norm(
    dataloader: torch.utils.data.DataLoader,
    data_preprocessor: ParticleImagePreprocessor,
    encoder: HetEncoder,
    decoder: DisplacementDecoder,
    particle_euler_angles: torch.nn.Parameter,
    particle_shifts: torch.nn.Parameter,
    data_normalization_mask: torch.Tensor,
):
    """Compute the data norm part of the loss function calibration."""
    for batch_ndx, sample in enumerate(dataloader):
        # zero gradients
        encoder.zero_grad()
        decoder.zero_grad()

        # get data and move to correct device
        idx, y, ctf = sample['idx'], sample["image"], sample["ctf"]
        shift = particle_shifts[idx]
        r = particle_euler_angles[idx]
        device = decoder.device
        # for tensor in y, r, ctf, shift:
        #     tensor = tensor.to(decoder.device)
        y, r, ctf, shift = y.to(device), r.to(
            device), ctf.to(device), shift.to(device)
        # preprocess data for passing through encoder
        data_preprocessor.set_device(decoder.device)
        y_in = data_preprocessor.apply_square_mask(y)
        y_in = data_preprocessor.apply_translation(
            y_in.detach(), -shift[:, 0].detach(), -shift[:, 1].detach()
        )
        y_in = data_preprocessor.apply_circular_mask(y_in)
        ctf = torch.fft.fftshift(ctf, dim=[-1, -2])

        mu, logsigma = encoder(y_in, ctf)
        z = mu + torch.exp(0.5 * logsigma) * torch.randn_like(mu)
        z_in = z
        Proj, new_points, deformed_points = decoder(z_in, r, shift)

        # calculate loss
        y = sample["image"].to(decoder.device)
        y = data_preprocessor.apply_circular_mask(y.detach())
        reconstruction_loss = fourier_loss(
            Proj.squeeze(), y.squeeze(), ctf.float(),
            W=data_normalization_mask[None, :, :]
        )

        # backprop
        reconstruction_loss.backward()

        # calculate norm of gradients

        with torch.no_grad():

            try:
                total_norm = 0

                for p in decoder.parameters():
                    if p.requires_grad == True and p.grad != None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                data_norm += total_norm ** 0.5
            except:
                data_norm = 1
    return data_norm


def _compute_geometry_norm(
    dataloader: torch.utils.data.DataLoader,
    data_preprocessor: ParticleImagePreprocessor,
    encoder: HetEncoder,
    decoder: DisplacementDecoder,
    particle_euler_angles: torch.nn.Parameter,
    particle_shifts: torch.nn.Parameter,
    lambda_regularization: torch.nn.Parameter,
    mode: RegularizationMode,
):
    geometry_norm = 0
    geometric_loss = GeometricLoss(
        mode=mode,
        neighbour_loss_weight=0.0,
        repulsion_weight=0.01,
        outlier_weight=1,
        deformation_regularity_weight=1,
    )
    for batch_ndx, sample in enumerate(dataloader):
        # zero gradients
        encoder.zero_grad()
        decoder.zero_grad()

        # prepare data for passing through model
        idx, y, ctf = sample['idx'], sample["image"], sample["ctf"]
        r = particle_euler_angles[idx]
        shift = particle_shifts[idx]
        device = decoder.device
        # for tensor in y, r, ctf, shifts:
        #     tensor = tensor.to(decoder.device)
        y, r, ctf, shift = y.to(device), r.to(
            device), ctf.to(device), shift.to(device)
        data_preprocessor.set_device(decoder.device)
        y_in = data_preprocessor.apply_square_mask(y)
        y_in = data_preprocessor.apply_translation(
            y_in.detach(), -shift[:, 0].detach(), -shift[:, 1].detach()
        )
        y_in = data_preprocessor.apply_circular_mask(y_in)

        ctf = torch.fft.fftshift(ctf, dim=[-1, -2])

        # pass data through model
        mu, logsigma = encoder(y_in, ctf)
        z = mu + torch.exp(0.5 * logsigma) * torch.randn_like(mu)
        z_in = z
        Proj, new_points, deformed_points = decoder(z_in, r, shift)

        # compute loss
        geo_loss = geometric_loss(
            deformed_positions=new_points,
            mean_neighbour_distance=decoder.mean_neighbour_distance,
            mean_graph_distance=decoder.mean_graph_distance,
            consensus_pairwise_distances=decoder.model_distances,
            knn=decoder.neighbour_graph,
            radius_graph=decoder.radius_graph,
            box_size=decoder.box_size,
            ang_pix=decoder.ang_pix,
            active_indices=decoder.active_indices
        )
        try:
            geo_loss.backward()
        except:
            pass
        # compute norm of gradients

        with torch.no_grad():
            try:
                total_norm = 0

                for p in decoder.parameters():
                    if p.requires_grad == True and p.grad != None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                geometry_norm += total_norm ** 0.5
            except:
                geometry_norm = 1

    return geometry_norm
