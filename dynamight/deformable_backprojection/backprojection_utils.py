#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:35:15 2022

@author: schwab
"""

from scipy.spatial import KDTree
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import mrcfile


def get_ess_grid(grid, points, box_size, edge=600):
    """Find grid points which are close to points from gaussian model."""
    tree = KDTree(points.cpu().numpy())
    (dists, points) = tree.query(grid.cpu().numpy())
    ess_grid = grid[dists < edge / box_size]
    out_grid = grid[dists >= edge / box_size]
    return ess_grid, out_grid


class DeformationInterpolator:
    """Interpolator for resampling a deformation field."""

    def __init__(self, device, grid, points, box_size, ds):
        super(DeformationInterpolator, self).__init__()
        self.grid = grid.cpu().numpy()
        self.int_grid = torch.tensor(
            np.round((self.grid + 0.5) * (box_size - 1)) // ds).long()
        self.box_size = box_size
        self.ds = ds

    def compute_field(self, values):
        size = self.box_size // self.ds
        dispM = 2 * torch.ones(1, 3, size, size, size)
        values = values.cpu()
        values = torch.tensor(self.grid) - values + torch.tensor(self.grid)
        dispM[0, :, self.int_grid[:, 2], self.int_grid[:, 1],
              self.int_grid[:, 0]] = values.movedim(
            0, 1).float()
        dispL = torch.nn.functional.upsample(dispM, scale_factor=self.ds,
                                             mode='trilinear')

        return dispL[0] * 2


class RotateVolume(nn.Module):
    """Rotate 3D volume(s) by given rotations

    rotations should be provided as RELION Euler angles.
    """

    def __init__(self, box_size, device):
        super(RotateVolume, self).__init__()
        self.device = device
        self.box_size = box_size

    def forward(self, V, rr):
        batch_size = rr.shape[0]

        roll = rr[:, 2:3]
        yaw = rr[:, 0:1]
        pitch = -rr[:, 1:2]

        tensor_0 = torch.zeros_like(roll).to(self.device)
        tensor_1 = torch.ones_like(roll).to(self.device)

        RX = torch.stack([
            torch.stack([torch.cos(roll), -torch.sin(roll), tensor_0]),
            torch.stack([torch.sin(roll), torch.cos(roll), tensor_0]),
            torch.stack([tensor_0, tensor_0, tensor_1])])

        RY = torch.stack([
            torch.stack([torch.cos(pitch), tensor_0, -torch.sin(pitch)]),
            torch.stack([tensor_0, tensor_1, tensor_0]),
            torch.stack([torch.sin(pitch), tensor_0, torch.cos(pitch)])])

        RZ = torch.stack([
            torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
            torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
            torch.stack([tensor_0, tensor_0, tensor_1])])

        RX = torch.movedim(torch.movedim(RX, 3, 0), 3, 0)
        RY = torch.movedim(torch.movedim(RY, 3, 0), 3, 0)
        RZ = torch.movedim(torch.movedim(RZ, 3, 0), 3, 0)

        R = torch.matmul(RZ, RY)
        R = torch.matmul(R, RX)
        R = R.transpose(2, 3)
        R = torch.cat([R, torch.zeros([batch_size, 1, 3, 1]).to(self.device)],
                      dim=3).squeeze()
        if R.dim() == 2:
            R = R.unsqueeze(0)
        G = torch.nn.functional.affine_grid(R, (
            batch_size, 1, self.box_size + 1, self.box_size + 1,
            self.box_size + 1),
            align_corners=False).float()
        VV = F.grid_sample(input=V, grid=G.to(self.device), mode='bilinear',
                           align_corners=False)

        return VV


def generate_smooth_mask_and_grids(mask_file, device, soft_edge=10):
    """Soft edged mask and grids with points inside and outside the mask."""
    if type(mask_file) == str:
        with mrcfile.open(mask_file) as mrc:
            rec_area = torch.tensor(mrc.data).to(device)
    else:
        rec_area = mask_file
    box_size = rec_area.shape[-1]
    bin_mask = (rec_area > 0).float()
    sm_bin_mask = torch.nn.functional.conv3d(bin_mask.unsqueeze(0).unsqueeze(0),
                                             torch.ones(1, 1, soft_edge,
                                                        soft_edge,
                                                        soft_edge).to(
                                                 device) / (soft_edge ** 3),
                                             padding='same')
    sm_bin_mask = sm_bin_mask.squeeze()
    rec_area = rec_area.movedim(0, 1).movedim(2, 1).movedim(1, 0)
    ggv = torch.linspace(0, box_size - 1, box_size)
    GG = torch.meshgrid(ggv, ggv, ggv)
    Ginds = rec_area[GG[0].long(), GG[1].long(), GG[2].long()] > 0
    pi = torch.where(Ginds > 0)
    po = torch.where(Ginds <= 0)
    ppi = torch.stack(pi, 1)
    ppo = torch.stack(po, 1)
    pointsi = ppi / box_size - 0.5
    pointso = ppo / box_size - 0.5
    ess_grid = pointsi
    out_grid = pointso
    del rec_area, GG, Ginds, pointsi, pointso

    return ess_grid, out_grid, sm_bin_mask


def generate_smooth_mask_from_consensus(consensus, box_size, ang_pix, distance, soft_edge=10):
    "generate mask from consensus gaussian model with mask being 1 if voxel center closer than distance (in angstrom) from a gaussian center"
    device = consensus.device
    points = (consensus.pos*box_size)/ang_pix
    x = (torch.linspace(-0.5, 0.5, box_size)*(box_size-1))/ang_pix
    grid = torch.meshgrid(x, x, x)
    grid = torch.stack([grid[0].ravel(), grid[1].ravel(), grid[2].ravel()], 1)
    tree = KDTree(points.cpu().numpy())
    (dists, points) = tree.query(grid.cpu().numpy())
    ess_grid = grid[dists < distance]
    ess_grid_int = torch.round((ess_grid - torch.min(grid))*ang_pix).long()
    M = torch.zeros(box_size, box_size, box_size)
    M[ess_grid_int[:, 0], ess_grid_int[:, 1], ess_grid_int[:, 2]] = 1
    if soft_edge > 0:
        sm_bin_mask = torch.nn.functional.conv3d(M.unsqueeze(0).unsqueeze(0).to(device),
                                                 torch.ones(1, 1, soft_edge,
                                                            soft_edge,
                                                            soft_edge).to(
                                                     device) / (soft_edge ** 3),
                                                 padding='same')
        sm_bin_mask = sm_bin_mask.squeeze()
    else:
        sm_bin_mask = M.to(device)
    return sm_bin_mask.movedim(0, 1).movedim(1, 2).movedim(0, 1)
