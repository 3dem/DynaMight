import torch
import numpy as np
from tsnecuda import TSNE
from sklearn.decomposition import PCA
import umap
import torch.nn.functional as F
from tqdm import tqdm


def compute_dimensionality_reduction(
        latent_space: torch.Tensor, method: str
) -> np.array:
    if method == 'TSNE':
        embedded_latent_space = TSNE(perplexity=1000.0, num_neighbors=1000,
                                     device=0).fit_transform(latent_space.cpu())
    elif method == 'UMAP':
        embedded_latent_space = umap.UMAP(
            random_state=12, n_neighbors=100).fit_transform(latent_space.cpu().numpy())
    elif method == 'PCA':
        embedded_latent_space = PCA(n_components=2).fit_transform(
            latent_space.cpu().numpy())
    return embedded_latent_space


def compute_latent_space_and_colors(
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        poses: torch.nn.Module,
        data_preprocessor,
        indices
):
    device = decoder.device

    color_euler_angles = poses.orientations[indices]
    color_euler_angles = F.normalize(color_euler_angles, dim=1)
    color_euler_angles = color_euler_angles.detach().cpu().numpy()
    color_euler_angles = color_euler_angles / 2 + 0.5
    color_euler_angles = color_euler_angles[:, 2]

    color_shifts = poses.translations[indices]
    color_shifts = torch.linalg.norm(color_shifts, dim=1)
    color_shifts = color_shifts.detach().cpu().numpy()
    color_shifts = color_shifts - np.min(color_shifts)
    color_shifts /= np.max(color_shifts)

    '-----------------------------------------------------------------------------'
    'Evaluate model on the half-set'

    global_distances = torch.zeros(decoder.model_positions.shape[0]).to(device)

    with torch.no_grad():
        for batch_ndx, sample in enumerate(tqdm(dataloader)):
            r, y, ctf = sample["rotation"], sample["image"], sample["ctf"]
            idx = sample['idx']
            r, t = poses(idx)
            y = data_preprocessor.apply_square_mask(y)
            y = data_preprocessor.apply_translation(y, -t[:, 0], -t[:, 1])
            y = data_preprocessor.apply_circular_mask(y)
            ctf = torch.fft.fftshift(ctf, dim=[-1, -2])
            y, r, ctf, t = y.to(device), r.to(
                device), ctf.to(device), t.to(device)
            mu, _ = encoder(y, ctf)
            _, _, displacements = decoder(mu, r, t)
            displacement_norm = torch.linalg.vector_norm(displacements, dim=-1)
            mean_displacement_norm = torch.mean(displacement_norm, 0)
            global_distances += mean_displacement_norm
            model_positions = torch.stack(
                displacement_norm.shape[0] * [decoder.model_positions], 0)
            if batch_ndx == 0:
                z = mu
                color_amount = torch.sum(displacement_norm, 1)
                color_direction = torch.mean(displacements.movedim(2, 0)
                                             * displacement_norm, -1)
                color_position = torch.mean(
                    model_positions.movedim(2, 0) * displacement_norm, -1)/torch.linalg.vector_norm(displacement_norm, -1)

            else:
                z = torch.cat([z, mu])
                color_amount = torch.cat(
                    [color_amount, torch.sum(displacement_norm, 1)])
                color_direction = torch.cat(
                    [color_direction, torch.mean(displacements.movedim(2, 0) * displacement_norm, -1)], 1)
                color_position = torch.cat([color_position,
                                            torch.mean(model_positions.movedim(
                                                2, 0) * displacement_norm, -1)/torch.linalg.vector_norm(displacement_norm, -1)], 1)

    color_direction = torch.movedim(color_direction, 0, 1)
    color_direction = F.normalize(color_direction, dim=1)
    color_direction = (color_direction + 1)/2
    color_position = torch.movedim(color_position, 0, 1)
    max_position = torch.max(torch.abs(color_position))
    color_position /= max_position
    color_position = (color_position+1)/2

    mean_deformation = global_distances / torch.max(global_distances)
    mean_deformation = mean_deformation.cpu().numpy()

    color_amount = color_amount.cpu().numpy()
    color_amount = color_amount / np.max(color_amount)
    color_direction = color_direction.cpu().numpy()
    color_position = color_position.cpu().numpy()

    consensus_positions = decoder.model_positions.detach().cpu().numpy()
    consensus_color = consensus_positions/max_position.cpu().numpy()
    consensus_color = np.mean((consensus_color+1)/4+0.5, -1)

    amps = decoder.amp.detach().cpu()
    amps = amps[0]
    amps -= torch.min(amps)
    amps /= torch.max(amps)
    widths = torch.nn.functional.softmax(decoder.ampvar, 0)[0].detach().cpu()

    lat_colors = {'amount': color_amount, 'direction': color_direction, 'location': color_position, 'index': indices, 'pose': color_euler_angles,
                  'shift': color_shifts}
    point_colors = {'activity': mean_deformation, 'amplitude': amps,
                    'width': widths, 'position': consensus_color}

    return z, lat_colors, point_colors
