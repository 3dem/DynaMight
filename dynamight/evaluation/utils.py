import torch
import numpy as np
from tsnecuda import TSNE
from sklearn.decomposition import PCA, FastICA
import torch.nn.functional as F
from tqdm import tqdm
from ..utils.utils_new import apply_ctf
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


def compute_dimensionality_reduction(
        latent_space: torch.Tensor, method: str
) -> np.array:
    # latent_space = (latent_space-torch.mean(latent_space, 1)) / \
    #    torch.std(latent_space, 1)
    if method == 'TSNE':
        embedded_latent_space = TSNE(perplexity=1000.0, num_neighbors=1000,
                                     device=0).fit_transform(latent_space.cpu())
    elif method == 'UMAP':
        import umap
        embedded_latent_space = umap.UMAP(
            random_state=12, n_neighbors=10).fit_transform(latent_space.cpu().numpy())
    elif method == 'PCA':
        embedded_latent_space = PCA(n_components=2).fit_transform(
            latent_space.cpu().numpy())
    elif method == 'ICA':
        embedded_latent_space = FastICA(
            n_components=2).fit_transform(latent_space.cpu().numpy())
    return embedded_latent_space


def compute_latent_space_and_colors(
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        poses: torch.nn.Module,
        data_preprocessor,
        indices,
        reduce_by_deformation=False,
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

    global_distances = torch.zeros(
        decoder.model_positions.shape[0]).to(device)

    feature_vec = []
    with torch.no_grad():
        for batch_ndx, sample in enumerate(tqdm(dataloader)):
            r, y, ctf = sample["rotation"], sample["image"], sample["ctf"]
            idx = sample['idx']
            r, t = poses(idx)
            # plt.figure()
            # plt.imshow(y[0], cmap='bone')
            # plt.axis('off')
            # plt.show()

            y = data_preprocessor.apply_square_mask(y)
            y = data_preprocessor.apply_translation(y, -t[:, 0], -t[:, 1])
            y = data_preprocessor.apply_circular_mask(y)
            ctf = torch.fft.fftshift(ctf, dim=[-1, -2])
            y, r, ctf, t = y.to(device), r.to(
                device), ctf.to(device), t.to(device)
            mu, _ = encoder(y, ctf)
            _, _, displacements = decoder(
                mu, r, t, positions=decoder.model_positions)

            if reduce_by_deformation is True:
                feature_vec.append(
                    displacements[:, ::4, :].flatten(start_dim=1))
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

    if reduce_by_deformation is True:
        feature_vec = torch.cat(feature_vec, 0)

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
    cluster_colors = torch.zeros_like(amps)

    lat_colors = {'amount': color_amount, 'direction': color_direction, 'location': color_position, 'index': indices, 'pose': color_euler_angles,
                  'shift': color_shifts, 'cluster': cluster_colors}
    point_colors = {'activity': mean_deformation, 'amplitude': amps,
                    'width': widths, 'position': consensus_color}

    return z, lat_colors, point_colors, feature_vec


def compute_max_deviation(
        latent_space_h1,
        latent_space_h2,
        decoder_h1: torch.nn.Module,
        decoder_h2: torch.nn.Module,
):
    device = decoder_h1.device
    dataset = TensorDataset(latent_space_h1, latent_space_h2)
    loader = DataLoader(dataset, batch_size=100)
    global_max = 0
    diff_col = []
    '-----------------------------------------------------------------------------'
    'Evaluate model on the half-set'
    with torch.no_grad():
        for i, sample in enumerate(loader):
            latent_1 = sample[0]
            latent_2 = sample[1]
            r = torch.zeros(latent_1.shape[0], 3).to(device)
            t = torch.zeros(latent_1.shape[0], 2).to(device)
            latent_1, latent_2 = latent_1.to(device), latent_2.to(device)
            _, _, dis_h1 = decoder_h1(latent_1, r, t)
            _, _, dis_h2 = decoder_h2(latent_2, r, t)

            difference = torch.sqrt(torch.sum(
                ((dis_h1[:]-dis_h2[:])*decoder_h1.box_size/decoder_h1.ang_pix)**2, -1))
            diff_col.append(torch.mean(difference, 1))
            max_diff = torch.max(difference)
            if max_diff > global_max:
                global_max = max_diff

    diff_col = torch.cat(diff_col, 0)
    np.savetxt('/cephfs/schwab/differences_wrong_new.txt',
               diff_col.cpu().numpy(), fmt='%.8f')
    diff_col /= torch.max(diff_col)
    return global_max, diff_col
