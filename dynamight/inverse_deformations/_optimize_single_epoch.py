

import torch
import numpy as np

from ..models.constants import ConsensusInitializationMode
from ..models.losses import GeometricLoss
from ..utils.utils_new import frc, fourier_loss

from tqdm import tqdm


def optimize_epoch(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    inverse_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    poses: torch.nn.Module,
    data_preprocessor,
    epoch,
    add_noise,
    latent_space,
    deformed_positions
):

    device = decoder.device

    inv_loss = 0

    for batch_ndx, sample in enumerate(dataloader):
        optimizer.zero_grad()
        if epoch > 0 and add_noise == False:
            idx = sample['idx']
            c_pos = inverse_model(latent_space[idx].to(device),
                                  deformed_positions[idx].to(device))
        else:
            with torch.no_grad():
                r, y, ctf = sample["rotation"], sample["image"], sample[
                    "ctf"]
                idx = sample['idx']
                r, t = poses(idx)
                ctfs = torch.fft.fftshift(ctf, dim=[-1, -2])
                y_in = data_preprocessor.apply_square_mask(y)
                y_in = data_preprocessor.apply_translation(y_in, -t[:, 0],
                                                           -t[:, 1])
                y_in = data_preprocessor.apply_circular_mask(y_in)
                mu, sig = encoder(y_in.to(device), ctfs.to(device))
                if add_noise == True:
                    noise_real = (5 / (
                        decoder.ang_pix * decoder.box_size)) * torch.randn_like(
                        decoder.model_positions).to(device)
                    evalpos = decoder.model_positions + noise_real
                    proj, pos, dis = decoder.forward(
                        mu, r.to(device), t.to(device), evalpos.to(device))
                else:
                    proj, pos, dis = decoder.forward(
                        mu, r.to(device), t.to(device))
                    latent_space[idx] = mu.cpu()
                    deformed_positions[idx] = pos.cpu()
            c_pos = inverse_model(mu, pos)

        if add_noise == True:
            loss = torch.sum(
                (c_pos - decoder.model_positions.to(device) + noise_real) ** 2)
        else:
            loss = torch.sum(
                (c_pos - decoder.model_positions.to(device)) ** 2)
        loss.backward()
        optimizer.step()
        inv_loss += loss.item()

    return inv_loss, latent_space, deformed_positions
