#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 12:24:45 2021

@author: schwab
"""
from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.fft
from sklearn.decomposition import PCA
from tsnecuda import TSNE
from Bio.PDB import PDBParser, MMCIFParser, PDBIO
from Bio.PDB.mmcifio import MMCIFIO
from tqdm import tqdm
import mrcfile
from ..data.dataloaders.relion import RelionDataset
from scipy.special import comb
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.spatial import KDTree

'-----------------------------------------------------------------------------'
'CTF and Loss Functions'
'-----------------------------------------------------------------------------'


def apply_ctf(x, ctf):
    if x.is_complex():
        pass
    else:
        x = torch.fft.fft2(torch.fft.fftshift(x, dim=[-1, -2]), dim=[-1, -2])
    x0 = torch.multiply(x, ctf)
    x0 = torch.fft.ifft2(x0)
    return torch.real(x0)


def fourier_loss(x, y, ctf, W=None, sig=None):
    if x.is_complex():
        pass
    else:
        x = torch.fft.fft2(torch.fft.fftshift(
            x, dim=[-1, -2]), dim=[-1, -2], norm='ortho')
    y = torch.fft.fft2(torch.fft.fftshift(
        y, dim=[-1, -2]), dim=[-1, -2], norm='ortho')
    x = torch.multiply(x, ctf)
    # if W != None:
    #    y = torch.multiply(y, W)
    # else:
    #     x = torch.multiply(x,ctf)
    if W != None:
        l = torch.mean(torch.mean(
            torch.multiply(W, torch.abs(x - y) ** 2), dim=[-1, -2]))
    else:
        l = torch.mean(torch.mean(
            torch.abs(x - y) ** 2, dim=[-1, -2]))
    return l


'-----------------------------------------------------------------------------'
'projection stuff'
'-----------------------------------------------------------------------------'


class PointProjector(nn.Module):
    """Projects multi-class points from 3D to 2D."""

    # for angle ordering [TILT,ROT,PSI]

    def __init__(self, box_size):
        super(PointProjector, self).__init__()
        self.box_size = box_size

    def forward(self, p, rr):
        device = p.device
        # batch_size = rr.shape[0]
        # yaw = rr[:,1:2]+np.pi
        # pitch = -rr[:,0:1]
        # roll = rr[:,2:3]+np.pi
        if len(rr.shape) < 3:
            roll = rr[:, 0:1] + np.pi
            yaw = -rr[:, 2:3]
            pitch = rr[:, 1:2] + np.pi

            tensor_0 = torch.zeros_like(roll).to(device)
            tensor_1 = torch.ones_like(roll).to(device)

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
            # rotation of P
            Rp = torch.stack([
                torch.stack([tensor_1, tensor_0, tensor_0]),
                torch.stack([tensor_0, -tensor_1, tensor_0]),
                torch.stack([tensor_0, tensor_0, -tensor_1])])
            Rp = torch.movedim(torch.movedim(Rp, 3, 0), 3, 0)

            R = torch.matmul(RZ, RY)
            R = torch.matmul(R, RX)

        else:
            R = rr.unsqueeze(1)
            tensor_0 = torch.zeros_like(rr[:, :1, 0]).to(device)
            tensor_1 = torch.ones_like(rr[:, :1, 0]).to(device)
            Rp = torch.stack([
                torch.stack([tensor_1, tensor_0, tensor_0]),
                torch.stack([tensor_0, -tensor_1, tensor_0]),
                torch.stack([tensor_0, tensor_0, -tensor_1])])
            Rp = torch.movedim(torch.movedim(Rp, 3, 0), 3, 0)

        points3 = p  # -np.sqrt(2)*1/self.box_size*torch.ones_like(p)
        points2 = torch.matmul(R.squeeze(), torch.matmul(
            Rp.squeeze(), points3.movedim(1, 2)))
        ind = [-3, -2]
        return points2[:, ind, :]


class PointsToImages(nn.Module):
    """Renders a batch of images from a multi-class point cloud."""

    def __init__(self, box_size, n_classes, oversampling=1):
        super(PointsToImages, self).__init__()
        self.box_size = box_size
        self.n_classes = n_classes
        self.os = oversampling

    def forward(self, points, values):
        self.batch_size = points.shape[0]
        p = ((points + 0.5) * (self.box_size * self.os)).movedim(1, 2)
        device = p.device
        im = torch.zeros(self.batch_size, self.n_classes,
                         (self.box_size * self.os) ** 2).to(device)
        xypoints = p.floor().long()
        rxy = p - xypoints
        x, y = xypoints.split(1, dim=-1)
        rx, ry = rxy.split(1, dim=-1)

        for dx in (0, 1):
            x_ = x + dx
            wx = (1 - dx) + (2 * dx - 1) * rx
            for dy in (0, 1):
                y_ = y + dy
                wy = (1 - dy) + (2 * dy - 1) * ry

                w = wx * wy

                valid = ((0 <= x_) * (x_ < self.os * self.box_size) *
                         (0 <= y_) * (y_ < self.os * self.box_size)).long()

                idx = ((y_ * self.box_size * self.os + x_) * valid).squeeze()
                idx = torch.stack(self.n_classes * [idx], 1)
                w = (w * valid.type_as(w)).squeeze()
                w = torch.stack(self.n_classes * [w], 1)
                im.scatter_add_(2, idx, w * values)
        im = im.reshape(self.batch_size, self.n_classes,
                        self.os * self.box_size, self.os * self.box_size)

        return im


def initialize_consensus(model, ref, logdir, lr=0.001, n_epochs=300, mask=None):
    """deprecate"""
    device = model.device
    model_params = model.parameters()
    model_optimizer = torch.optim.Adam(model_params, lr=lr)
    z0 = torch.zeros(2, 2)
    r0 = torch.zeros(2, 3)
    t0 = torch.zeros(2, 2)
    print('Initializing gaussian positions from reference deformable_backprojection')
    for i in tqdm(range(n_epochs)):
        model_optimizer.zero_grad()
        V = model.generate_volume(r0.to(device), t0.to(device)).float()
        # fsc,res=FSC(ref,V[0],1,visualize = False)
        loss = torch.nn.functional.mse_loss(
            V[0], ref)  # +1e-7*f1(lay(model.pos))
        loss.backward()
        model_optimizer.step()
    print('Final error:', loss.item())
    with mrcfile.new(logdir + '/ini_volume.mrc', overwrite=True) as mrc:
        mrc.set_data((V[0] / torch.mean(V[0])).float().detach().cpu().numpy())


class FourierImageSmoother(nn.Module):
    """Smooths multiclass images in Fourier space.

    Smoothing is performed by multiplication with a gaussian.
    """

    def __init__(self, box_size, device, n_classes, oversampling=1, A=None, B=None):
        #  todo: A, B -> widths, amplitudes
        #  or else...
        super(FourierImageSmoother, self).__init__()
        self.box_size = box_size
        self.device = device
        self.n_classes = n_classes
        self.rad_inds, self.rad_mask = radial_index_mask(
            oversampling * box_size)

        if A == None and B == None:

            self.B = torch.nn.Parameter(torch.linspace(
                self.box_size/60, self.box_size/60, n_classes).to(device),
                requires_grad=True)
            self.A = torch.nn.Parameter(torch.linspace(
                0.1, 0.1, n_classes).to(device), requires_grad=True)

        else:
            self.B = torch.nn.Parameter(B.to(device), requires_grad=True)
            self.A = torch.nn.Parameter(A.to(device), requires_grad=False)

        self.os = oversampling
        self.crop = fourier_crop

    def forward(self, ims):
        R = torch.stack(self.n_classes * [self.rad_inds.to(self.device)], 0)
        # F = torch.exp(-(1/(self.B[:, None, None])**2) *
        #               R**2) * (self.A[0, None, None]**2)  # /self.B[:, None, None])
        # F = torch.exp(-(1/(self.B[:, None, None])**2) *
        #              R**2) * (torch.nn.functional.softmax(self.A[:, None, None], 0)**2)  # /self.B[:, None, None])
        F = torch.exp(-(1/(self.B[:, None, None])**2) *
                      R**2)  # * (torch.nn.functional.softmax(self.A[:, None, None], 0))
        # F = torch.exp(-(1/(self.B[:, None, None])**2) *
        #               R**2) * (self.A[:, None, None]**2)
        FF = torch.real(torch.fft.fft2(torch.fft.fftshift(
            F, dim=[-1, -2]), dim=[-1, -2], norm='ortho'))*(0.01+self.A[:, None, None]**2)/self.B[:, None, None]
        bs = ims.shape[0]
        Filts = torch.stack(bs * [FF], 0)
        # Filts = torch.fft.ifftshift(Filts, dim=[-2, -1])
        Fims = torch.fft.fft2(torch.fft.fftshift(
            ims, dim=[-2, -1]), norm='ortho')
        if self.n_classes > 1:
            out = torch.sum(Filts * Fims, dim=1)
        else:
            out = Filts * Fims
        if self.os > 1:
            out = self.crop(out, self.os)
        return out


def fourier_crop(img, oversampling):
    s = img.shape[-1]
    img = torch.fft.fftshift(img, [-1, -2])
    out = img[..., s // 2 - s // (2 * oversampling):s // 2 + s // (2 * oversampling),
              s // 2 - s // (2 * oversampling):s // 2 + s // (2 * oversampling)]
    out = torch.fft.fftshift(out, [-1, -2])
    return out


def fourier_crop3(img, oversampling):
    s = img.shape[-1]
    img = torch.fft.fftshift(img, [-1, -2, -3])
    out = img[..., s // 2 - s // (2 * oversampling):s // 2 + s // (2 * oversampling),
              s // 2 - s // (2 * oversampling):s // 2 + s // (2 * oversampling),
              s // 2 - s // (2 * oversampling):s // 2 + s // (2 * oversampling)]
    out = torch.fft.fftshift(out, [-1, -2, -3])
    return out


def radial_index_mask(box_size):

    x = torch.tensor(
        np.linspace(-box_size, box_size, box_size, endpoint=False))
    X, Y = torch.meshgrid(x, x, indexing='ij')
    R = torch.round(torch.sqrt(X ** 2 + Y ** 2))
    Mask = R < (x[-1])

    return R.long(), Mask


def radial_index_mask3(box_size, scale=None):
    if scale == None:
        x = torch.tensor(
            np.linspace(-box_size, box_size, box_size, endpoint=False))
    else:
        x = torch.tensor(
            np.linspace(-scale*box_size, scale*box_size, box_size, endpoint=False))
    X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
    R = torch.round(torch.sqrt(X ** 2 + Y ** 2 + Z ** 2))
    Mask = R < (x[-1])

    return R, Mask


def generate_form_factor(a, b, box_size):
    r = np.linspace(0, box_size, box_size, endpoint=False)
    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()
    R = np.stack(a.shape[0] * [r], 0)
    F = np.exp(-b[:, None] ** 2 * R) * a[:, None] ** 2
    return np.moveaxis(F, 0, 1)


def find_initialization_parameters(model, V):
    r0 = torch.zeros(2, 3).to(model.device)
    t0 = torch.zeros(2, 2).to(model.device)
    Vmodel = model.generate_volume(r0, t0)
    ratio = torch.sum(V ** 2) / torch.sum(Vmodel ** 2)
    # s = torch.linspace(0,0.5,100)
    # a = torch.linspace(0,2*ratio,100)
    # rad_inds, rad_mask = radial_index_mask3(oversampling*box_size)
    # R = torch.stack(1*[rad_inds.to(model.device)],0)
    # for i in range(100):
    #     for j in range(100):
    #         FF = torch.exp(-s[i,None,None]**2*R)*a[j,None,None]**2
    model.amp = torch.nn.Parameter(
        0.55 * torch.ones(1).to(model.device), requires_grad=True)


def initialize_dataset(
    refinement_star_file: Path,
    circular_mask_thickness: float,
    preload: bool,
    particle_diameter: Optional[float] = None
):
    dataset = RelionDataset(refinement_star_file)
    ### For Kyle's weird data ###
    # a = []
    # for i in dataset.image_file_paths:
    #     mrc = mrcfile.open(i, 'r')
    #     num_im = mrc.data.shape[0]
    #     a.append(np.arange(num_im))
    # dataset.part_stack_idx = np.concatenate(a)
    # dataset = dataset.make_particle_dataset()

    return dataset, diameter_ang, box_size, ang_pix, optics_group


class PointsToVolumes(nn.Module):
    """Renders points as multi-class volumes.

    Points are spread over 8 nearest voxels by trilinear interpolation.
    """

    def __init__(self, box_size, n_classes, oversampling):
        super(PointsToVolumes, self).__init__()
        self.box_size = box_size
        self.n_classes = n_classes
        self.os = oversampling

    def forward(self, positions, amplitudes):
        self.batch_size = positions.shape[0]
        device = positions.device
        p = ((positions + 0.5) * (self.box_size*self.os))
        vol = torch.zeros(self.batch_size, self.n_classes,
                          (self.box_size*self.os) ** 3).to(device)

        xyzpoints = p.floor().long()

        rxyz = p - xyzpoints

        x, y, z = xyzpoints.split(1, dim=-1)
        rx, ry, rz = rxyz.split(1, dim=-1)

        for dx in (0, 1):
            x_ = x + dx
            wx = (1 - dx) + (2 * dx - 1) * rx
            for dy in (0, 1):
                y_ = y + dy
                wy = (1 - dy) + (2 * dy - 1) * ry
                for dz in (0, 1):
                    z_ = z + dz
                    wz = (1 - dz) + (2 * dz - 1) * rz

                    w = wx * wy * wz
                    if self.batch_size > 1:
                        valid = ((0 <= x_) * (x_ < self.os*self.box_size) *
                                 (0 <= y_) * (y_ < self.os*self.box_size) *
                                 (0 <= z_) * (z_ < self.os*self.box_size)).long()
                        idx = ((((z_ * self.os*self.box_size + y_) *
                               self.box_size*self.os) + x_) * valid).squeeze()
                        idx = torch.stack(self.n_classes * [idx], 1)
                        w = (w * valid.type_as(w)).squeeze()
                        w = torch.stack(self.n_classes * [w], 1)
                    else:
                        valid = ((0 <= x_) * (x_ < self.os*self.box_size) *
                                 (0 <= y_) * (y_ < self.box_size*self.os) *
                                 (0 <= z_) * (z_ < self.box_size*self.os)).long()
                        idx = ((((z_ * self.box_size*self.os + y_) *
                               self.box_size*self.os) + x_) * valid).squeeze(2)
                        idx = torch.stack(self.n_classes * [idx], 1)
                        w = (w * valid.type_as(w)).squeeze(2)
                        w = torch.stack(self.n_classes * [w], 1)

                    vol.scatter_add_(2, idx, w * amplitudes)

        vol = vol.reshape(self.batch_size, self.n_classes,
                          self.os*self.box_size, self.os*self.box_size, self.os*self.box_size)
        return vol


def frc(x, y, ctf, batch_reduce='sum'):
    y = torch.fft.fft2(torch.fft.fftshift(
        y.squeeze(), dim=[-1, -2]), dim=[-1, -2])
    x = torch.multiply(x, ctf)
    N = x.shape[-1]
    device = x.device
    batch_size = x.shape[0]
    eps = 1e-8
    ind = torch.linspace(-(N - 1) / 2, (N - 1) / 2 - 1, N)

    X, Y = torch.meshgrid(ind, ind, indexing='ij')
    R = torch.cat(batch_size * [torch.fft.fftshift(torch.round(
        torch.pow(X ** 2 + Y ** 2, 0.5)).long()).unsqueeze(0)], 0).to(device)

    num = scatter_mean(torch.real(x * torch.conj(y)).flatten(start_dim=-2),
                       R.flatten(start_dim=-2))

    den = torch.pow(
        scatter_mean(torch.abs(x.flatten(start_dim=-2))
                     ** 2, R.flatten(start_dim=-2))
        * scatter_mean(torch.abs(y.flatten(start_dim=-2)) ** 2,
                       R.flatten(start_dim=-2)), 0.5)
    FRC = num / (den + eps)
    FRC = torch.sum(num / den, 0)

    return FRC


def maskpoints(points, ampvar, mask, box_size):
    bs = points.shape[0]
    indpoints = torch.round((points + 0.5) * (box_size - 1)).long()
    indpoints = torch.clip(indpoints, max=mask.shape[-1] - 1, min=0)
    if len(indpoints.shape) > 2:
        point_inds = mask[indpoints[:, :, 0],
                          indpoints[:, :, 1], indpoints[:, :, 2]] > 0
    else:
        point_inds = mask[indpoints[:, 0],
                          indpoints[:, 1], indpoints[:, 2]] > 0
    return points[point_inds, :], point_inds


def tensor_imshow(tensor, cmap='viridis'):
    x = tensor
    if len(x.shape) == 3:
        x = x[x.shape[0] // 2]

    if type(x).__module__ == 'torch':
        x = x.detach().data.cpu().numpy()

    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(x, cmap=cmap)
    plt.axis("off")
    plt.subplots_adjust(hspace=0, wspace=0)
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    matplotlib.use(backend)

    return fig


def tensor_plot(tensor, fix=False):
    x = tensor
    if type(x).__module__ == 'torch':
        x = x.detach().data.cpu().numpy()
    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x)
    if fix != False:
        ax.set_ylim([fix[0], fix[1]])

    matplotlib.use(backend)

    return fig


def tensor_scatter(x, y, c, s=0.1, alpha=0.5, cmap='inferno'):
    x = x.detach().cpu()
    y = y.detach().cpu()
    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')
    fig, ax = plt.subplots(figsize=(5, 5))
    if isinstance(c, str):
        ax.scatter(x, y, alpha=alpha, s=s, c=c)
    else:
        ax.scatter(x, y, alpha=alpha, s=s, c=c, cmap=cmap)
    plt.axis("off")
    plt.subplots_adjust(hspace=0, wspace=0)
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    matplotlib.use(backend)

    return fig


def tensor_hist(x, b):
    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')
    a = torch.max(torch.abs(x))
    h = torch.histc(x, b, min=-a, max=a)
    h = h.detach().cpu()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(np.arange(len(h)), h)

    matplotlib.use(backend)

    return fig


def visualize_latent(z, c, s=0.1, alpha=0.5, cmap='jet', method='umap'):
    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')
    if type(z).__module__ == 'torch':
        z = z.detach().data.cpu().numpy()
    if z.shape[1] == 2:
        embed = z
    elif method == 'pca':
        embed = PCA(n_components=2).fit_transform(z)
    elif method == 'tsne':
        tsne = TSNE(n_jobs=16)
        embed = tsne.fit_transform(z)
    elif method == 'umap':
        import umap
        embed = umap.UMAP(local_connectivity=1,
                          repulsion_strength=2, random_state=12).fit_transform(z)
    elif method == 'projection_last':
        embed = z[:-1]
    elif method == 'projection_first':
        embed = z[1:]

    fig, ax = plt.subplots(figsize=(5, 5))
    s = 40000 / z.shape[0]
    ax.scatter(embed[:, 0], embed[:, 1], alpha=alpha, s=s, c=c, cmap='jet')
    plt.axis("off")
    plt.subplots_adjust(hspace=0, wspace=0)
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.set_aspect('equal', adjustable='box')

    matplotlib.use(backend)

    return fig


def write_xyz(
    points: torch.Tensor,
    output_file: Path,
    box_size: int,
    ang_pix: float,
    class_id: torch.Tensor  # argmax of gaussian width
):
    # turn class IDs into atom specifiers for coloring
    points = box_size * ang_pix * (0.5 + points.detach().data.cpu().numpy())
    atom_spec = np.array(['C', 'O', 'N', 'H', 'S', 'Se', 'O1'] + ['C'] * 200)
    if torch.is_tensor(class_id) == True:
        atom_idx = class_id.detach().cpu().numpy().astype(int)
    else:
        atom_idx = class_id.astype(int)
    atoms = atom_spec[atom_idx]

    # write out file
    with open(output_file, mode='a') as f:
        f.write("%d\n%s\n" % (points.size / 3, output_file))
        for x, atom in zip(points.reshape(-1, 3), atoms):
            f.write("%s %.18g %.18g %.18g\n" % (atom, x[0], x[1], x[2]))


def graph2bild(points, edge_index, title, edge_thickness=None, color=5):
    points = points.detach().cpu().numpy()
    with open(title, 'a') as f:
        for k in range(points.shape[0]):
            f.write("%s %.18g %.18g %.18g %.18g\n" %
                    ('.sphere', points[k, 0], points[k, 1], points[k, 2], 0.02))
        y = np.concatenate([points[edge_index[0].cpu().numpy()],
                            points[edge_index[1].cpu().numpy()]], 1)
        f.write('%s %.18g\n' % ('.color', color))
        if edge_thickness != None:
            edge_thickness = edge_thickness.cpu()
        for k in range(y.shape[0]):
            if edge_thickness != None:
                t = edge_thickness[k]
            else:
                t = 0.01
            f.write('%s %.18g\n' % ('.color', t))
            f.write("%s %.18g %.18g %.18g %.18g %.18g %.18g %.18g\n" % (
                '.cylinder', y[k, 0], y[k, 1], y[k, 2], y[k, 3], y[k, 4], y[k, 5], 0.02))


def graphs2bild(total_points, points, edge_indices, amps, title, box_size, ang_pix):
    f = open(title + '.bild', 'a')
    color = 8
    total_points = total_points.detach().cpu().numpy()
    total_points = (total_points + 0.5) * box_size * ang_pix
    tk = 0
    for points, amps, edge_index in zip(points, amps, edge_indices):
        points = points.detach().cpu().numpy()
        points = (points + 0.5) * box_size * ang_pix
        points = points / (box_size * ang_pix) - 0.5
        f.write('%s %.18g\n' % ('.color', color))
        if edge_index != None:
            y = np.concatenate([total_points[edge_index[0].cpu(
            ).numpy()], total_points[edge_index[1].cpu().numpy()]], 1)
            for k in range(y.shape[0]):
                f.write("%s %.18g %.18g %.18g %.18g %.18g %.18g %.18g\n" % (
                    '.cylinder', y[k, 0], y[k, 1], y[k,
                        2], y[k, 3], y[k, 4], y[k, 5],
                    0.12))
        for k in range(points.shape[0]):
            f.write("%s %.18g %.18g %.18g %.18g\n" % (
                '.sphere', points[k, 0], points[k, 1], points[k, 2],
                0.04 * amps[tk + k]))
        color = color + 10
        tk += points.shape[0]
    f.close()


def field2bild(points, field, uncertainty, title, box_size, ang_pix):
    f = open(title + '.bild', 'a')
    color = 25
    cols = torch.linalg.norm(points - field, dim=1)
    points = points.detach().cpu().numpy()
    points = (points + 0.5) * box_size * ang_pix
    uncertainty = uncertainty*box_size*ang_pix
    cols = torch.round((uncertainty/20 * 45)).long()
    field = field.detach().cpu().numpy()
    field = (field + 0.5) * box_size * ang_pix
    y = np.concatenate([points, field], 1)
    #cols = torch.round(cols / torch.max(cols) * 65).long()
    for k in range(y.shape[0]):

        f.write('%s %.18g %.18g %.18g\n' % ('.color', 0.0, 0.0, 0.0))
        f.write("%s %.18g %.18g %.18g %.18g %.18g %.18g %.18g\n %.18g\n %.18g\n" % (
            '.arrow', y[k, 0], y[k, 1], y[k, 2], y[k, 3], y[k, 4], y[k, 5], 0.5, 1, 0.5))
    f.write('%s %.18g\n' % ('.transparency', 0.6))
    for k in range(y.shape[0]):
        f.write('%s %.18g\n' % ('.color', cols[k]))
        f.write("%s %.18g %.18g %.18g %.18g\n" % (
            '.sphere', y[k, 3], y[k, 4], y[k, 5], uncertainty[k]))

    f.close()


def points2bild(points, amps, title, box_size, ang_pix, color=19):
    f = open(title + '.bild', 'a')
    points = points.detach().cpu().numpy()
    points = (points + 0.5) * box_size * ang_pix
    #f.write('%s %.18g\n' % ('.color', color))
    for k in range(points.shape[0]):
        f.write('%s %.18g\n' % ('.color', color[k]))
        f.write("%s %.18g %.18g %.18g %.18g\n" % (
            '.sphere', points[k, 0], points[k, 1], points[k, 2], 2))  # * amps[k]))
    color = color + 20
    f.close()


def series2xyz(points, title, box_size, ang_pix):
    if type(points).__module__ == 'torch':
        points = box_size * ang_pix * \
            (0.5 + points.detach().data.cpu().numpy())
    atomtype = ("C",)
    for i in range(points.shape[0]):
        pp = points[i].squeeze()
        f = open(title + '.xyz', 'a')
        f.write("%d\n%s\n" % (points[i].size / 3, title))
        for x in points.reshape(-1, 3):
            f.write("%s %.18g %.18g %.18g\n" % (atomtype, x[0], x[1], x[2]))
    f.close()


def power_spec2(F1, batch_reduce=None):
    if F1.is_complex():
        pass
    else:
        F1 = torch.fft.fft2(torch.fft.fftshift(
            F1, dim=[-1, -2]), dim=[-2, -1], norm='ortho')
    device = F1.device
    N = F1.shape[-1]
    ind = torch.linspace(-(N - 1) / 2, (N - 1) / 2 - 1, N).to(device)
    end_ind = torch.round(torch.tensor(N / 2)).long()
    X, Y = torch.meshgrid(ind, ind, indexing='ij')
    R = torch.fft.fftshift(torch.round(torch.pow(X ** 2 + Y ** 2, 0.5)).long())
    p_s = scatter_mean(torch.abs(F1.flatten(start_dim=-2)) ** 2,
                       R.flatten().to(F1.device))
    if batch_reduce == 'mean':
        p_s = torch.mean(p_s, 0)
    p = p_s[R]
    return p, p_s


def radial_avg2(F1, batch_reduce=None):
    if F1.is_complex():
        pass
    else:
        F1 = torch.fft.fftn(F1, dim=[-2, -1])
    N = F1.shape[-1]
    ind = torch.linspace(-(N - 1) / 2, (N - 1) / 2 - 1, N)
    end_ind = torch.round(torch.tensor(N / 2)).long()
    X, Y = torch.meshgrid(ind, ind, indexing='ij')
    R = torch.fft.fftshift(torch.round(torch.pow(X ** 2 + Y ** 2, 0.5)).long())
    res = torch.arange(start=0, end=end_ind) ** 2,
    p_s = scatter_mean(torch.abs(F1.flatten(start_dim=-2)),
                       R.flatten().to(F1.device))
    if batch_reduce == 'mean':
        p_s = torch.mean(p_s, 0)
    p = p_s[R]
    return p, p_s


def prof2radim(w, out_value=0):
    N = w.shape[0]
    ind = torch.linspace(-N, N - 1, 2 * N)
    X, Y = torch.meshgrid(ind, ind, indexing='ij')
    R = torch.fft.fftshift(torch.round(torch.pow(X ** 2 + Y ** 2, 0.5)).long())
    R[R > N - 1] = N - 1
    W = w[R]
    W[R == N - 1] = out_value
    return W


def RadialAvg(F1, batch_reduce=None):
    if F1.is_complex():
        pass
    else:
        F1 = torch.fft.fftn(F1, dim=[-3, -2, -1])
    N = F1.shape[-1]
    ind = torch.linspace(-(N - 1) / 2, (N - 1) / 2 - 1, N)
    end_ind = torch.round(torch.tensor(N / 2)).long()
    X, Y, Z = torch.meshgrid(ind, ind, ind, indexing='ij')
    R = torch.fft.fftshift(torch.round(
        torch.pow(X ** 2 + Y ** 2 + Z ** 2, 0.5)).long())
    res = torch.arange(start=0, end=end_ind) ** 2,

    if len(F1.shape) == 3:
        p_s = scatter(torch.abs(F1.flatten(start_dim=-3)),
                      R.flatten(), reduce='mean')
    return p_s


def RadialAvgProfile(F1, batch_reduce=None):
    if F1.is_complex():
        pass
    else:
        F1 = torch.fft.fftn(F1, dim=[-3, -2, -1])
    device = F1.device
    N = F1.shape[-1]
    ind = torch.linspace(-(N - 1) / 2, (N - 1) / 2 - 1, N)
    end_ind = torch.round(torch.tensor(N / 2)).long()
    X, Y, Z = torch.meshgrid(ind, ind, ind, indexing='ij')
    R = torch.fft.fftshift(torch.round(
        torch.pow(X ** 2 + Y ** 2 + Z ** 2, 0.5)).long()).to(device)
    res = torch.arange(start=0, end=end_ind) ** 2,

    if len(F1.shape) == 3:
        p_s = scatter_mean(torch.abs(F1.flatten(start_dim=-3)),
                           R.flatten())
    Prof = torch.zeros_like(R).float().to(device)
    Prof[R] = p_s[R]

    return Prof


def fourier_shift_2d(
    grid_ft,
    xshift,
    yshift
):
    s = grid_ft.shape[-1]
    xshift = -xshift / float(s)
    yshift = -yshift / float(s)

    if torch.is_tensor(grid_ft):
        ls = torch.linspace(-s // 2, s // 2 - 1, s)
        y, x = torch.meshgrid(ls, ls, indexing='ij')
        x = x.to(grid_ft.device)
        y = y.to(grid_ft.device)
        dot_prod = 2 * np.pi * \
            (x[None, :, :] * xshift[:, None, None] +
             y[None, :, :] * yshift[:, None, None])
        dot_prod = torch.fft.fftshift(dot_prod, dim=[-1, -2])
        a = torch.cos(dot_prod)
        b = torch.sin(dot_prod)
    else:
        ls = np.linspace(-s // 2, s // 2 - 1, s),
        y, x = np.meshgrid(ls, ls, indexing="ij")
        dot_prod = 2 * np.pi * \
            (x[None, :, :] * xshift[:, None, None] +
             y[None, :, :] * yshift[:, None, None])
        dot_prod = torch.fft.fftshift(dot_prod, dim=[-1, -2])
        a = np.cos(dot_prod)
        b = np.sin(dot_prod)

    ar = a * grid_ft.real
    bi = b * grid_ft.imag
    ab_ri = (a + b) * (grid_ft.real + grid_ft.imag)

    return ar - bi + 1j * (ab_ri - ar - bi)


def FlipZ(F1):
    Fz = torch.flip(F1, [-3])
    return (Fz)


def PowerSpec(F1, batch_reduce=None):
    if F1.is_complex():
        pass
    else:
        F1 = torch.fft.fftn(F1, dim=[-3, -2, -1])
    N = F1.shape[-1]
    ind = torch.linspace(-(N - 1) / 2, (N - 1) / 2 - 1, N)
    end_ind = torch.round(torch.tensor(N / 2)).long()
    X, Y, Z = torch.meshgrid(ind, ind, ind, indexing='ij')
    R = torch.fft.fftshift(torch.round(
        torch.pow(X ** 2 + Y ** 2 + Z ** 2, 0.5)).long())
    res = torch.arange(start=0, end=end_ind) ** 2,

    if len(F1.shape) == 3:
        # p_s = scatter(torch.abs(F1.flatten(start_dim=-3))
        #              ** 2, R.flatten(), reduce='sum')
        p_s = scatter(torch.abs(F1.flatten(start_dim=-3))
                      ** 2, R.flatten())
    return p_s[:end_ind], res[0]


def FSC(F1, F2, ang_pix=1, visualize=False):
    device = F1.device
    if F1.is_complex():
        pass
    else:
        F1 = torch.fft.fftn(F1, dim=[-3, -2, -1])
    if F2.is_complex():
        pass
    else:
        F2 = torch.fft.fftn(F2, dim=[-3, -2, -1])

    if F1.shape != F2.shape:
        print('The volumes have to be the same size')

    N = F1.shape[-1]
    ind = torch.linspace(-(N - 1) / 2, (N - 1) / 2 - 1, N)
    end_ind = torch.round(torch.tensor(N / 2)).long()
    X, Y, Z = torch.meshgrid(ind, ind, ind, indexing='ij')
    R = torch.fft.fftshift(torch.round(
        torch.pow(X ** 2 + Y ** 2 + Z ** 2, 0.5)).long()).to(device)

    if len(F1.shape) == 3:
        num = torch.zeros(torch.max(R) + 1).to(device)
        den1 = torch.zeros(torch.max(R) + 1).to(device)
        den2 = torch.zeros(torch.max(R) + 1).to(device)
        num.scatter_add_(0, R.flatten(), torch.real(
            F1 * torch.conj(F2)).flatten())
        den = torch.pow(
            den1.scatter_add_(0, R.flatten(), torch.abs(
                F1.flatten(start_dim=-3)) ** 2)
            * den2.scatter_add_(0, R.flatten(),
                                torch.abs(F2.flatten(start_dim=-3)) ** 2), 0.5)
        FSC = num / den
    res = N * ang_pix / torch.arange(end_ind)
    FSC[0] = 1
    if visualize == True:
        plt.figure(figsize=(10, 10))
        plt.rcParams['axes.xmargin'] = 0
        plt.plot(FSC[:end_ind].cpu(), c='r')
        plt.plot(torch.ones(end_ind) * 0.5, c='black', linestyle='dashed')
        plt.plot(torch.ones(end_ind) * 0.143,
                 c='slategrey', linestyle='dotted')
        plt.xticks(torch.arange(start=0, end=end_ind, step=10), labels=np.round(
            res[torch.arange(start=0, end=end_ind, step=10)].numpy(), 1))
        plt.show()
    return FSC[0:end_ind], res


def make_color_map(p, mean_dist, kernel, box_size, device, n_classes):
    p2V = PointsToVolumes(box_size, n_classes)
    p = p.unsqueeze(0)
    mean_dist = mean_dist.unsqueeze(0).to(device)
    Cmap = p2V(p.expand(2, -1, -1), mean_dist.expand(2, -1)).to(device)
    return Cmap


def select_subset(starfile, subset_indices, outname):
    with open(starfile, 'r+') as f:
        with open(outname, 'w') as output:
            d = f.readlines()
            mode = 'handlers'
            count = 0
            startcount = 0
            for i in d:
                if i.startswith('data_particles'):
                    print('particles')
                    mode = 'particles'
                if mode == 'handlers':
                    output.write(i)

                if mode == 'particles':
                    output.write(i)
                    if i.startswith('loop_'):
                        print('particles_loop')
                        mode = 'particle_props'

                if mode == 'particle_props':
                    if i.startswith('_'):
                        output.write(i)
                    else:
                        if startcount == 0:
                            startcount = count
                        if count - startcount in subset_indices:
                            output.write(i)
                count = count + 1
            output.close()


def add_weight_decay_to_named_parameters(
    model: torch.nn.Module, weight_decay: float = 1e-5
) -> torch.nn.ParameterList:
    """Adding weight decay to weights, not biases"""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'weight' not in name and name not in ['image_smoother.A', 'image_smoother.B', 'amp', 'ampvar', 'model_positions']:
            no_decay.append(param)
        elif name not in ['image_smoother.A', 'image_smoother.B', 'amp', 'ampvar', 'model_positions']:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]


def pdb2points(name, random=False):
    pdb = PDBParser()
    cif = MMCIFParser()

    total_length = 0
    total_gr = []

    if name.endswith('.pdb'):
        model = pdb.get_structure('model', name)
    elif name.endswith('.cif'):
        model = cif.get_structure('model', name)

    coords = []
    for chain in model.get_chains():
        residues = chain.get_residues()
        for res in residues:
            for a in res.get_atoms():
                coords.append(a.get_coord())

    coords = torch.tensor(np.array(coords))

    return coords


def points2pdb(name, outname, points, random=False):
    if name.endswith('.pdb'):
        io = PDBIO()
    elif name.endswith('.cif'):
        io = MMCIFIO()
    pdb = PDBParser()
    cif = MMCIFParser()

    total_length = 0
    total_gr = []

    if name.endswith('.pdb'):
        model = pdb.get_structure('model', name)
    elif name.endswith('.cif'):
        model = cif.get_structure('model', name)
    i = 0
    coords = []
    for chain in model.get_chains():
        residues = chain.get_residues()
        for res in residues:
            for a in res.get_atoms():
                a.set_coord(points[i])
                i += 1

    io.set_structure(model)
    io.save(outname, preserve_atom_numbering=True)


def pdb2graph(name):
    pdb = PDBParser()
    cif = MMCIFParser()

    total_length = 0
    total_gr = []
    total_amp = []

    if name.endswith('.pdb'):
        model = pdb.get_structure('model', name)
    elif name.endswith('.cif'):
        model = cif.get_structure('model', name)

    for chain in model.get_chains():
        coords = []
        amp = []
        direction = []
        residues = chain.get_residues()
        for res in residues:
            cm = res.center_of_mass()
            try:
                coords.append(res['CA'].get_coord())
                direction.append(cm - res['CA'].get_coord())
                amp.append(len(res.child_list))
                amp.append(len(res.child_list))
            except:
                coords.append(res.center_of_mass())
                direction.append(np.zeros(3))
                amp.append(len(res.child_list))
                amp.append(len(res.child_list))

        coords = torch.tensor(np.array(coords))
        amp = torch.tensor(np.array(amp))
        direction = torch.tensor(np.array(direction))

        if total_length == 0:
            gr = torch.stack([torch.arange(
                start=0, end=coords.shape[0] - 1),
                torch.arange(start=1, end=coords.shape[0])], 0)
            gr = torch.cat(
                [gr,
                 torch.tensor([coords.shape[0] - 1, coords.shape[0] - 1]).unsqueeze(1)],
                1)
        else:
            gr_c = torch.stack([torch.arange(start=total_length,
                                             end=total_length + coords.shape[0] - 1),
                                torch.arange(start=total_length + 1,
                                             end=total_length + coords.shape[0])], 0)
            gr_c = torch.cat([gr_c, torch.tensor(
                [total_length + coords.shape[0] - 1,
                 total_length + coords.shape[0] - 1]).unsqueeze(1)], 1)
            gr = torch.cat([gr, gr_c], 1)
        if total_length == 0:
            total_coords = coords
            total_dirs = direction
            total_amp = amp
        else:

            total_coords = torch.cat([total_coords, coords], 0)
            total_dirs = torch.cat([total_dirs, direction], 0)
            total_amp = torch.cat([total_amp, amp], 0)
        total_length += coords.shape[0]

    diff1 = total_coords[gr[0]] - total_coords[gr[1]]
    diffnorm1 = torch.linalg.norm(diff1, dim=1)
    diff = total_coords[gr[0]] - total_coords[gr[1]]
    gr = gr[:, diffnorm1 < 7]
    randc = torch.randn_like(total_coords)
    randc /= torch.stack(3 * [torch.linalg.norm(randc, dim=1)], 1)
    zero_inds = torch.linalg.norm(diff, dim=1) == 0
    diff[zero_inds == True] = diff[torch.roll(zero_inds == True, -1)]
    # norms = torch.cross(direction,randc)
    norms = total_dirs.float()
    zinds = torch.where(torch.linalg.norm(norms, dim=1) == 0)
    norms[zinds[0]] = torch.cross(diff[zinds[0]], randc[zinds[0]]).float()
    norms = norms / torch.stack(3 * [torch.linalg.norm(norms, dim=1)], 1) * 3.7
    add_coords = total_coords + norms

    add1 = torch.arange(start=0, end=len(total_coords))
    add2 = torch.arange(start=len(total_coords),
                        end=len(total_coords) + len(add_coords))

    gr_add = torch.stack([add1, add2], 0)
    gr = torch.cat([gr, gr_add], 1)

    xyz = torch.cat([total_coords, add_coords], 0)
    gr = gr[:, gr[0] != gr[1]]
    gr = gr.long()

    return xyz, gr, total_amp


def pdb2allatoms(names, box_size, ang_pix):
    t_positions = []
    atoms = []
    for name in names:
        atompos = pdb2points(name)
        atoms.append(atompos)
    atom_positions = torch.cat(atoms, 0)
    atom_positions = atom_positions / (box_size * ang_pix)
    if torch.min(atom_positions) > 0:  # correct for normal pdbs
        atom_positions = atom_positions - 0.5

    return atom_positions


def initial_optimization(cons_model, atom_model, device, directory, angpix, N_epochs):
    z0 = torch.zeros(2, 2)
    r0 = torch.zeros(2, 3)
    t0 = torch.zeros(2, 2)
    V0 = atom_model.generate_volume(r0.to(device), t0.to(device))
    V0 = V0[0].float()
    box_size = V0.shape[0]
    with mrcfile.new(directory + '/optimization_volume.mrc', overwrite=True) as mrc:
        mrc.set_data(V0.detach().cpu().numpy())
    atom_model.requires_grad = False
    cons_model.pos.requires_grad = False
    coarse_params = cons_model.parameters()
    coarse_optimizer = torch.optim.Adam(coarse_params, lr=0.001)
    cons_model.amp.requires_grad = True
    V = cons_model.generate_volume(r0.to(device), t0.to(device)).float()
    fsc, res = FSC(V0.detach(), V[0], 1, visualize=False)

    for i in tqdm(range(N_epochs)):
        coarse_optimizer.zero_grad()
        V = cons_model.generate_volume(r0.to(device), t0.to(device)).float()
        fsc, res = FSC(V0.detach(), V[0], 1, visualize=False)
        loss = -torch.sum(fsc)  # 1e-2*f1(lay(coarse_model.pos))
        types = torch.argmax(torch.nn.functional.softmax(
            cons_model.ampvar, dim=0), dim=0)
        write_xyz(cons_model.pos, '/cephfs/schwab/approximation/positions' +
                  str(i).zfill(3), box_size, angpix, types)

        loss.backward()
        coarse_optimizer.step()

    with mrcfile.new(directory + '/coarse_initial_volume.mrc', overwrite=True) as mrc:
        mrc.set_data(V[0].detach().cpu().numpy())

    print('Total FSC value:', torch.sum(fsc))


def load_models(path, device, box_size, n_classes):
    cp = torch.load(path, map_location=device)
    encoder_half1 = cp['encoder_half1']
    encoder_half2 = cp['encoder_half2']
    # cons_model_l = cp['consensus']
    decoder_half1 = cp['decoder_half1']
    decoder_half1.p2i = PointsToImages(box_size, n_classes, 1)
    decoder_half1.image_smoother = FourierImageSmoother(
        box_size, device, n_classes, 1)
    decoder_half2 = cp['decoder_half2']
    decoder_half2.p2i = PointsToImages(box_size, n_classes, 1)
    decoder_half2.image_smoother = FourierImageSmoother(
        box_size, device, n_classes, 1)
    poses = cp['poses']
    encoder_half1.load_state_dict(cp['encoder_half1_state_dict'])
    encoder_half2.load_state_dict(cp['encoder_half2_state_dict'])
    decoder_half1.load_state_dict(cp['decoder_half1_state_dict'])
    decoder_half2.load_state_dict(cp['decoder_half2_state_dict'])
    poses.load_state_dict(cp['poses_state_dict'])
    decoder_half1.p2i.device = device
    decoder_half2.p2i.device = device
    decoder_half1.projector.device = device
    decoder_half2.projector.device = device
    decoder_half1.image_smoother.device = device
    decoder_half2.image_smoother.device = device
    decoder_half1.p2v.device = device
    decoder_half1.device = device
    decoder_half2.p2v.device = device
    decoder_half2.device = device
    decoder_half1.to(device)
    decoder_half2.to(device)

    return encoder_half1, encoder_half2, decoder_half1, decoder_half2


def reset_all_linear_layer_weights(model: nn.Module) -> nn.Module:
    """
    Resets all weights recursively for linear layers.

    ref:
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @ torch.no_grad()
    def init_weights(m):
        if type(m) == nn.Linear:
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(init_weights)


class spatial_grad(nn.Module):
    # For TV regularization
    def __init__(self, box_size):
        super(spatial_grad, self).__init__()
        self.box_size = box_size

    def forward(self, x):
        batch_size = x.size()[0]
        t_x = x.size()[2]
        h_x = x.size()[3]
        w_x = x.size()[4]

        t_grad = (x[:, :, 2:, :, :] - x[:, :, :h_x - 2, :, :])
        h_grad = (x[:, :, :, 2:, :] - x[:, :, :, :h_x - 2, :])
        w_grad = (x[:, :, :, :, 2:] - x[:, :, :, :, :w_x - 2])
        t_grad = torch.nn.functional.pad(t_grad[0, 0], [0, 0, 0, 0, 1, 1])
        h_grad = torch.nn.functional.pad(h_grad[0, 0], [0, 0, 1, 1, 0, 0])
        w_grad = torch.nn.functional.pad(w_grad[0, 0], [1, 1, 0, 0, 0, 0])
        grad = torch.stack([t_grad, h_grad, w_grad], 0)
        return grad

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def compute_threshold(V, percentage=98):
    th = np.percentile(V.flatten().cpu(), percentage)
    return th


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array(
        [bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def make_equidistant(x, y, N):
    N_p = x.shape[0]
    points = np.stack([x, y], 1)
    dp = points[:-1] - points[1:, :]
    n_points = []
    dists = np.linalg.norm(dp, axis=1)
    min_dist = np.min(dists)
    curve_length = np.sum(dists)
    seg_length = curve_length / (N_p - 1)
    print('Curve length:', curve_length)
    for i in range(N_p - 1):
        q = dists[i] / min_dist
        Nq = np.round(q)
        for j in range(int(Nq) - 1):
            n_points.append(points[i] + j * min_dist *
                            (dp[i] / np.linalg.norm(dp[i])))
    n_points = np.array(n_points)
    frac = np.maximum(int(np.round(n_points.shape[0] / N)), 1)
    return n_points[::frac, :], points


def knn_graph(X, k, workers):
    dev = X.device
    s_m = kneighbors_graph(X.cpu(), k, n_jobs=workers)
    s_m_coo = s_m.tocoo()
    gr0 = torch.tensor(s_m_coo.col)
    gr1 = torch.tensor(s_m_coo.row)
    # gr0 = torch.tensor(s_m.indices)
    # gr1 = torch.arange(0, X.shape[0]).repeat_interleave(2, dim=0)
    gr = torch.stack([gr0, gr1], 0).long()
    return gr.to(dev)


def radius_graph(X, r, workers):
    dev = X.device
    s_m = radius_neighbors_graph(X.cpu(), float(r.cpu()), n_jobs=workers)
    s_m_coo = s_m.tocoo()
    gr0 = torch.tensor(s_m_coo.col)
    gr1 = torch.tensor(s_m_coo.row)
    gr = torch.stack([gr0, gr1], 0).long()
    return gr.to(dev)


def scatter(src, inds, dim=-1):
    inds = inds.expand(src.size())
    size = list(src.size())
    size[dim] = int(inds.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    out.scatter_add_(dim, inds, src)
    return out.scatter_add_(dim, inds, src)


def scatter_mean(src, inds, dim=-1):
    out_sum = scatter(src, inds, dim)
    ones = torch.ones(inds.size(), dtype=src.dtype, device=src.device)
    count = scatter(ones, inds, dim)
    return out_sum / count


def calculate_grid_oversampling_factor(box_size: int) -> int:
    """Calculate a grid oversampling factor for rendering gaussians.

    Rendering gaussians on a grid is done by projecting the
    gaussian down into 2D and 'spreading' values which sum up to 1 over the
    nearest four pixels. Values for each pixel are calculated by
    bilinear interpolation.
    """
    if box_size < 100:
        grid_oversampling_factor = 1
    if box_size < 300 and box_size > 99:
        grid_oversampling_factor = 1
    if box_size > 299:
        grid_oversampling_factor = 1
    return grid_oversampling_factor


def generate_data_normalization_mask(box_size, dampening_factor, device):
    """Multiplies with exponential decay"""
    xx = torch.tensor(np.linspace(-1, 1, box_size, endpoint=False))
    XX, YY = torch.meshgrid(xx, xx, indexing='ij')
    BF = torch.fft.fftshift(
        torch.exp(-(dampening_factor * (XX ** 2 + YY ** 2))), dim=[-1, -2]).to(
        device)
    return BF


def remove_bidirectional_edges(gr):
    gr_sort1 = torch.sort(gr, dim=0)[0]
    gr_out = torch.unique(gr_sort1, dim=1)
    return gr_out


def graph_union(gr1, gr2):
    # undir_gr1 = torch.stack([gr1[1],gr1[0]],0)
    # undir_gr2 = torch.stack([gr2[1],gr2[0]],0)
    combined_gr = torch.cat([gr1, gr2], 1)
    union_gr = torch.unique(combined_gr, dim=1)
    union_gr = remove_bidirectional_edges(union_gr)
    return union_gr


def mask_from_positions(positions, box_size, ang_pix, distance):
    "generate mask from consensus gaussian model with mask being 1 if voxel center closer than distance (in angstrom) from a gaussian center"
    device = positions.device
    if box_size > 200:
        new_box = box_size//2
        new_ang_pix = ang_pix/2
        points = (positions*(box_size-1))/ang_pix
    else:
        new_box = box_size
        new_ang_pix = ang_pix
        points = (positions*(box_size-1))/ang_pix
    x = (torch.linspace(-0.5, 0.5, new_box)*(box_size-1))/ang_pix
    grid = torch.meshgrid(x, x, x, indexing='ij')
    grid = torch.stack([grid[0].ravel(), grid[1].ravel(), grid[2].ravel()], 1)
    tree = KDTree(points.detach().cpu().numpy())
    (dists, points) = tree.query(grid.cpu().numpy())
    ess_grid = grid[dists < distance]
    ess_grid_int = torch.clip(torch.round(
        (ess_grid - torch.min(grid))*new_ang_pix).long(), max=new_box-1)
    M = torch.zeros(new_box, new_box, new_box)
    M[ess_grid_int[:, 0], ess_grid_int[:, 1], ess_grid_int[:, 2]] = 1
    if box_size > 200:
        M = torch.nn.functional.upsample_nearest(M[None, None], scale_factor=2)
    return M[0, 0].to(device)


def count_interior_points(positions, mask, box_size, ang_pix):
    device = positions.device
    points = (positions+0.5)*(box_size-1)
    #MM = torch.zeros_like(mask)
    point_inds = torch.clip(torch.round(points).long(),
                            min=0, max=box_size - 1)
    # MM[point_inds[0, :, 2], point_inds[0, :, 1], point_inds[0, :, 0]] = 1
    # with mrcfile.new('/cephfs/schwab/bugtest' + 'points.mrc', overwrite=True) as mrc:
    #     mrc.set_data(MM.cpu().float().numpy())
    #     mrc.voxel_size = ang_pix
    count = torch.sum(
        mask[point_inds[:, :, 2], point_inds[:, :, 1], point_inds[:, :, 0]], 1)

    return count


def generate_latent_units(latent_dim):
    I = torch.eye(latent_dim)
    return I


def compute_input_latents(decoder, r, t, ctf, input_ims):
    bs = r.shape[0]
    E = torch.stack(
        bs*[generate_latent_units(decoder.latent_dim)]).to(decoder.device)
    ims, _, _ = decoder(E, r, t)
    input_ims = apply_ctf(input_ims, ctf)
    y = input_ims.flatten(start_dim=1)
    A = ims.flatten(start_dim=2)
    ATA = A.transpose(1, 2)
    lat = torch.linalg.solve(ATA, y)
    return lat
