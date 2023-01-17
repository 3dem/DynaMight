import torch
from torch.utils.data import DataLoader

from ..models.encoder import HetEncoder
from ..models.decoder import DisplacementDecoder
from ..data.handlers.particle_image_preprocessor import ParticleImagePreprocessor
from ..utils.utils_new import fourier_loss, geometric_loss


def calibrate_regularization_parameter(
    dataset: torch.utils.data.Dataset,
    data_preprocessor: ParticleImagePreprocessor,
    encoder: HetEncoder,
    decoder: DisplacementDecoder,
    particle_shifts: torch.nn.Parameter,
    particle_euler_angles: torch.nn.Parameter,
    data_normalization_mask: torch.Tensor,
    regularization_factor: float,
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
    return regularization_factor * (0.5 * (data_norm / geometry_norm))


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
                    if p.requires_grad == True:
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
):
    geometry_norm = 0
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
            pos=new_points,
            box_size=decoder.box_size,
            ang_pix=decoder.ang_pix,
            dist=decoder.mean_neighbour_distance,
            deformation=decoder.model_distances,
            graph1=decoder.radius_graph,
            graph2=decoder.neighbour_graph,
            mode=decoder.loss_mode
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
                    if p.requires_grad == True:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                geometry_norm += total_norm ** 0.5
            except:
                geometry_norm = 1
    return geometry_norm
