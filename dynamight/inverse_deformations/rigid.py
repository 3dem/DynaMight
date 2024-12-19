#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:41:12 2023

@author: schwab
"""


import torch
import numpy as np

from ..models.constants import ConsensusInitializationMode
from ..models.losses import GeometricLoss
from ..utils.utils_new import frc, fourier_loss, points2bild
from ..data.handlers.so3 import euler_to_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from tqdm import tqdm


def R_to_relion_scipy(rot: np.ndarray, degrees: bool = True) -> np.ndarray:
    """Nx3x3 rotation matrices to RELION euler angles (cryodrgn)"""
    from scipy.spatial.transform import Rotation as RR

    if rot.shape == (3, 3):
        rot = rot.reshape(1, 3, 3)
    assert len(rot.shape) == 3, "Input must have dim Nx3x3"
    f = np.ones((3, 3))
    f[0, 1] = -1
    f[1, 0] = -1
    f[1, 2] = -1
    f[2, 1] = -1
    euler = RR.from_matrix(rot * f).as_euler("zxz", degrees=True)
    euler[:, 0] -= 90
    euler[:, 2] += 90
    euler += 180
    euler %= 360
    euler -= 180
    if not degrees:
        euler *= np.pi / 180
    return euler


def get_rotation_translation(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    poses: torch.nn.Module,
    data_preprocessor,
    masked_points,
    star_file_data,
    body,
    half,
):

    device = decoder.device
    new_stars = star_file_data

    for batch_ndx, sample in tqdm(enumerate(dataloader)):
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
            i = 0

            for points in masked_points:

                # proj, pos, dis = decoder.forward(
                #    mu, r.to(device), t.to(device), points.to(device))

                # proj_all, pos_all, dis_all = decoder.forward(
                #    mu, r.to(device), t.to(device))
                if body != None:
                    proj, pos, dis = decoder.forward(
                        mu, r.to(device), t.to(device))
                    pos = pos[:, decoder.masked_indices[body]]
                    dis = dis[:, decoder.masked_indices[body]]
                    points = decoder.model_positions[decoder.masked_indices[body]]

                else:
                    proj, pos, dis = decoder.forward(
                        mu, r.to(device), t.to(device), points.to(device))

                B = pos.movedim(1, 2)
                A = points

                tB = torch.stack(points.shape[0]*[torch.mean(pos, 1)], 2)
                tA = torch.stack(points.shape[0]*[torch.mean(points, 0)], 0)

                AA = A - tA
                BB = B - tB

                C = torch.matmul(AA.movedim(0, 1), BB.movedim(1, 2))

                U, S, V = torch.svd(C)
                rot = torch.matmul(V, U.moveaxis(1, 2))
                V[torch.det(rot) < 0, :, 2] = -V[torch.det(rot) < 0, :, 2]
                rot = torch.matmul(V, U.moveaxis(1, 2))

                shift = tB[:, :, 0] - torch.matmul(rot, tA[0])
                shift_ang = shift*decoder.box_size*decoder.ang_pix

                rlnRot = torch.tensor(np.array(
                    new_stars[i]['particles']['rlnAngleRot'].loc[idx]))
                rlnTilt = torch.tensor(np.array(
                    new_stars[i]['particles']['rlnAngleTilt'].loc[idx]))
                rlnPsi = torch.tensor(np.array(
                    new_stars[i]['particles']['rlnAnglePsi'].loc[idx]))
                dynang, dynshift = poses(idx)
                dynshift *= decoder.ang_pix
                rlnang = dynang
                # Use oiriginal relion angles??
                # rlnang = torch.stack(
                #    [rlnRot, rlnTilt, rlnPsi], 1).float().to(device)
                rot_cons = euler_to_matrix(rlnang)
                rot_shift_ang = torch.matmul(rot_cons, shift_ang.unsqueeze(2))
                # Dont change rotations??
                # new_rot = rot_cons.float().to(device)
                new_rot = torch.matmul(rot_cons.float().to(device), rot)
                new_euler = R_to_relion_scipy(new_rot.cpu())
                rigid_points = torch.matmul(
                    rot, decoder.model_positions.movedim(0, 1))
                # plot approximated rigid transform on points ??
                # if batch_ndx == 0:
                #     points2bild(decoder.model_positions[::100], torch.ones_like(
                #         decoder.model_positions[::100, 0]), '/ceph/scheres_grp/schwab/empiar-10028/original_positions', decoder.box_size, decoder.ang_pix)
                #     points2bild(pos_all[0, ::100], torch.ones_like(decoder.model_positions[::100, 0]),
                #                 '/ceph/scheres_grp/schwab/empiar-10028/deformed_positions', decoder.box_size, decoder.ang_pix, color=30)
                #     points2bild(rigid_points[0, :, ::100].movedim(0, 1)+torch.stack(decoder.model_positions[::100, 0].shape[0]*[shift[0]], 0), torch.ones_like(
                #         decoder.model_positions[::100, 0]), '/ceph/scheres_grp/schwab/empiar-10028/rigid_deformed_positions', decoder.box_size, decoder.ang_pix, color=40)

                new_stars[i]['particles'].loc[idx,
                                              'rlnAngleRot'] = new_euler[:, 0]
                new_stars[i]['particles'].loc[idx,
                                              'rlnAngleTilt'] = new_euler[:, 1]
                new_stars[i]['particles'].loc[idx,
                                              'rlnAnglePsi'] = new_euler[:, 2]
                # Dont change translations
                # new_star['particles'].loc[idx, 'rlnOriginXAngst'] = np.array(
                #    -dynshift[:, 0].cpu())
                # new_star['particles'].loc[idx, 'rlnOriginYAngst'] = np.array(
                #    -dynshift[:, 1].cpu())

                new_stars[i]['particles'].loc[idx, 'rlnOriginXAngst'] = np.array(
                    -dynshift[:, 0].cpu()-rot_shift_ang[:, 0].squeeze().cpu())
                new_stars[i]['particles'].loc[idx, 'rlnOriginYAngst'] = np.array(
                    -dynshift[:, 1].cpu()-rot_shift_ang[:, 1].squeeze().cpu())
                new_stars[i]['particles'].loc[idx, 'rlnRandomSubset'] = half

                i += 1
    return new_stars


def initialize_masks(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    poses: torch.nn.Module,
    data_preprocessor,
    star_file_data,
    half,
    number_of_classes
):

    device = decoder.device
    new_star = star_file_data.copy()
    total_labels = torch.zeros(decoder.model_positions.shape[0])
    for batch_ndx, sample in tqdm(enumerate(dataloader)):
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

            proj_all, pos_all, dis_all = decoder.forward(
                mu, r.to(device), t.to(device))
            f = torch.cat([dis_all, torch.stack(
                dis_all.shape[0]*[0.05*decoder.model_positions], 0)], -1)
            f = torch.movedim(f, 0, 1)
            try:
                kmeans = KMeans(n_clusters=number_of_classes, init=current_lables, random_state=0, n_init=1).fit(
                    f.flatten(start_dim=-2).cpu().numpy())
            except:
                kmeans = KMeans(n_clusters=number_of_classes, random_state=0, n_init=1).fit(
                    f.flatten(start_dim=-2).cpu().numpy())

            current_labels = kmeans.labels_
            total_labels += current_labels
            current_centers = kmeans.cluster_centers_

    total_labels /= len(dataloader)

    return torch.round(total_labels), current_labels


def compute_spatial_cluster_probability(labels, points):
    means = []
    var = []
    probs = []
    with torch.no_grad():
        for i in range(np.max(labels)+1):
            mean = torch.mean(points[labels == i], 0)
            variance = torch.mean((points-mean[None])**2)
            means.append(mean)
            var.append(variance)
            probability = torch.exp(-(torch.sum((mean-points)
                                                ** 2, -1)/(2*variance)))
            probs.append(probability)

    probs = torch.stack(probs, 0)
    probs = probs/torch.sum(probs, 0)

    return probs


def compute_transformation_probability(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    poses: torch.nn.Module,
    data_preprocessor,
    masked_points,
):

    device = decoder.device
    total_probs = torch.zeros(
        len(masked_points), decoder.model_positions.shape[0]).to(device)
    for batch_ndx, sample in tqdm(enumerate(dataloader)):
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
            probs = []
            indi = 0
            for points in masked_points:

                proj, pos, dis = decoder.forward(
                    mu, r.to(device), t.to(device), points.to(device))

                proj_all, pos_all, dis_all = decoder.forward(
                    mu, r.to(device), t.to(device))
                B = pos.movedim(1, 2)
                A = points

                tB = torch.stack(points.shape[0]*[torch.mean(pos, 1)], 2)
                tA = torch.stack(points.shape[0]*[torch.mean(points, 0)], 0)

                AA = A - tA
                BB = B - tB

                C = torch.matmul(AA.movedim(0, 1), BB.movedim(1, 2))

                U, S, V = torch.svd(C)
                rot = torch.matmul(V, U.moveaxis(1, 2))
                V[torch.det(rot) < 0, :, 2] = -V[torch.det(rot) < 0, :, 2]
                rot = torch.matmul(V, U.moveaxis(1, 2))

                shift = tB[:, :, 0] - torch.matmul(rot, tA[0])
                shift_ang = shift*decoder.box_size*decoder.ang_pix

                dynang, dynshift = poses(idx)
                dynshift *= decoder.ang_pix
                rlnang = dynang
                # Use oiriginal relion angles??
                # rlnang = torch.stack(
                #    [rlnRot, rlnTilt, rlnPsi], 1).float().to(device)
                rot_cons = euler_to_matrix(rlnang)
                rot_shift_ang = torch.matmul(rot_cons, shift_ang.unsqueeze(2))
                # Dont change rotations??
                # new_rot = rot_cons.float().to(device)
                new_rot = torch.matmul(rot_cons.float().to(device), rot)
                new_euler = R_to_relion_scipy(new_rot.cpu())
                rigid_points = torch.matmul(
                    rot, decoder.model_positions.movedim(0, 1)) + torch.stack(decoder.model_positions.shape[0]*[shift], 1).movedim(1, 2)

                # points2bild(pos_all[0, ::100], torch.ones_like(decoder.model_positions[::100, 0]),
                #            '/ceph/scheres_grp/schwab/deformed_positions', decoder.box_size, decoder.ang_pix, color=30*torch.ones_like(decoder.model_positions[::100, 0]))
                # points2bild(rigid_points[0, :, ::100].movedim(0, 1), torch.ones_like(decoder.model_positions[::100, 0]), '/ceph/scheres_grp/schwab/rigid_deformed_positions',
                #            decoder.box_size, decoder.ang_pix, color=indi*5+40*torch.ones_like(decoder.model_positions[::100, 0]))
                indi += 1
                exp_err = torch.exp(-torch.sum((pos_all -
                                    rigid_points.movedim(1, 2))**2, -1))

                probs.append(exp_err)

            probs = torch.stack(probs, 0)
            probs = probs / torch.sum(probs, 0)
            maxprobs = torch.argmax(probs, dim=0)
            maxprobs = torch.nn.functional.one_hot(
                maxprobs, len(masked_points))
            # print(maxprobs.shape)
            maxprobs = torch.sum(maxprobs, 0).movedim(0, 1)

            probs = torch.sum(probs, 1)/len(dataloader)
            # plt.imshow(probs.detach().cpu(), aspect='auto', cmap='inferno')
            # plt.show()
            total_probs += maxprobs

    real_probs = total_probs/torch.sum(total_probs, 0)

    return real_probs
