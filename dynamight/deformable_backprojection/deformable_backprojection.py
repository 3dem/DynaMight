#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:11:04 2022

@author: schwab
"""

import argparse
from data.handlers.particle_image_preprocessor import ParticleImagePreprocessor
from torch.utils.data import DataLoader
from dynamight.utils.utils_new import *

from tqdm import tqdm
import scipy

parser = argparse.ArgumentParser(description='VAE evaluation')

parser.add_argument('particle_dir', type=str, metavar='log_dir',
                    help='input job (job directory or optimizer-file)')
parser.add_argument('--mask', help='directory of mask file', type=str)
parser.add_argument('--ctf', help='reconstruct with ctf', action='store_true')
parser.add_argument('--reference', type=str, help='reference model if available')
parser.add_argument('--gpu', dest='gpu', type=str, default="-1", help='gpu to use')
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--preload', action='store_true')
parser.add_argument('--out_dir', type=str, metavar='out_dir', help='output name')
parser.add_argument('--VAE_dir', type=str, help='VAE checkpoint directory')
parser.add_argument('--random_half', help='reconstruct half sets', action='store_true')
parser.add_argument('--latent_sampling', help='Number of samples in latent space per dimension',
                    type=int, default=50)
parser.add_argument('--ignore', help='Ignore tiles with fewer than this particles', type=int,
                    default=10)
parser.add_argument('--filter', help='Choice of filter to use in the filtered backprojection',
                    type=str, default='ctf')
parser.add_argument('--random_half_decoder',
                    help='reconstruct half sets with decoders trained on random half sets',
                    action='store_true')
parser.add_argument('--reconstruction_area', help='directory of mask file for local deformable_backprojection',
                    type=str)
parser.add_argument('--noise', help='add noise in training of the inverse deformation field',
                    type=str, default=False)
parser.add_argument('--particle_diameter', help='size of circular mask (ang)', type=int,
                    default=None)
parser.add_argument('--circular_mask_thickness', help='thickness of mask (ang)', type=int,
                    default=20)
args = parser.parse_args()

ds = 2
particle_diameter = None
circular_mask_thickness = 20
dataloader_threads = 4
batch_size = args.batch_size

a = torch.cuda.current_device()
device = 'cuda:' + str(a) if args.gpu == "-1" else 'cuda:' + str(args.gpu)

if args.reference:
    with mrcfile.open(args.reference) as mrc:
        Rvol = torch.tensor(mrc.data)

cp = torch.load(args.VAE_dir, map_location=device)

if args.random_half_decoder:
    encoder_half1 = cp['encoder_half1']
    encoder_half2 = cp['encoder_half2']
    cons_model = cp['consensus']
    decoder_half1 = cp['decoder_half1']
    decoder_half2 = cp['decoder_half2']
    try:
        inv_half1 = cp['inv_def_half1']
        inv_half2 = cp['inv_def_half2']
    except:
        pass
    poses = cp['poses']

    encoder_half1.load_state_dict(cp['encoder_half1_state_dict'])
    encoder_half2.load_state_dict(cp['encoder_half2_state_dict'])
    cons_model.load_state_dict(cp['consensus_state_dict'])
    decoder_half1.load_state_dict(cp['decoder_half1_state_dict'])
    decoder_half2.load_state_dict(cp['decoder_half2_state_dict'])
    try:
        inv_half1.load_state_dict(cp['inv_def_half1_state_dict'])
        inv_half2.load_state_dict(cp['inv_def_half2_state_dict'])
    except:
        pass
    poses.load_state_dict(cp['poses_state_dict'])

else:

    encoder = cp['encoder']
    cons_model = cp['consensus']
    decoder = cp['decoder']
    poses = cp['poses']

    encoder.load_state_dict(cp['encoder_state_dict'])
    cons_model.load_state_dict(cp['consensus_state_dict'])
    decoder.load_state_dict(cp['decoder_state_dict'])
    poses.load_state_dict(cp['poses_state_dict'])
    decoder_half1 = decoder
    decoder_half2 = decoder

n_classes = 2
N_points = cons_model.n_points

points = cons_model.pos.detach().cpu()
points = torch.tensor(points)
cons_model.pos = torch.nn.Parameter(points, requires_grad=False)
cons_model.n_points = len(points)
decoder_half1.n_points = cons_model.n_points
decoder_half2.n_points = cons_model.n_points
cons_model.ampvar = torch.nn.Parameter(torch.zeros(n_classes, decoder_half1.n_points).to(device),
                                       requires_grad=False)

cons_model.p2i.device = device
decoder_half1.p2i.device = device
decoder_half2.p2i.device = device
cons_model.proj.device = device
decoder_half1.proj.device = device
decoder_half2.proj.device = device
cons_model.i2F.device = device
decoder_half1.i2F.device = device
decoder_half2.i2F.device = device
cons_model.p2v.device = device
decoder_half1.p2v.device = device
decoder_half2.p2v.device = device
decoder_half1.device = device
decoder_half2.device = device
cons_model.device = device
decoder_half1.to(device)
decoder_half2.to(device)
cons_model.to(device)

encoder_half1.to(device)
encoder_half2.to(device)

latent_dim = encoder_half1.latent_dim

dataset, diameter_ang, box_size, ang_pix, optics_group = initialize_dataset(args.particle_dir,
                                                                            args.circular_mask_thickness,
                                                                            args.preload,
                                                                            args.particle_diameter)
# optics_groups = dataset.get_optics_group_stats()

# optics_group = optics_groups[0]


# box_size = optics_group['image_size']
# ang_pix = optics_group['pixel_size']
min_dist = 4.6

if args.mask:
    with mrcfile.open(args.mask) as mrc:
        mask = torch.tensor(mrc.data).to(device)

gv = torch.linspace(-0.5, 0.5, box_size)
G = torch.meshgrid(gv, gv, gv)
grid = torch.stack([G[0].flatten(), G[1].flatten(), G[2].flatten()], 1)

max_diameter_ang = box_size * optics_group['pixel_size'] - circular_mask_thickness

if particle_diameter is None:
    diameter_ang = box_size * 1 * optics_group['pixel_size'] - circular_mask_thickness
    print(f"Assigning a diameter of {round(diameter_ang)} angstrom")
else:
    if particle_diameter > max_diameter_ang:
        print(
            f"WARNING: Specified particle diameter {round(args.particle_diameter)} angstrom is too large\n"
            f" Assigning a diameter of {round(max_diameter_ang)} angstrom"
        )
        diameter_ang = max_diameter_ang
    else:
        diameter_ang = particle_diameter

if args.preload:
    dataset.preload_images()

if args.random_half:
    inds_half1 = cp['indices_half1'].cpu().numpy()
    inds_half2 = list(set(range(len(dataset))) - set(list(inds_half1)))
    print(len(inds_half1))
    print(len(inds_half2))
    dataset_half1 = torch.utils.data.Subset(dataset, inds_half1)
    dataset_half2 = torch.utils.data.Subset(dataset, inds_half2)

    # dataset_half1, dataset_half2 = torch.utils.data.dataset.random_split(dataset,[len(dataset)//2,len(dataset)-len(dataset)//2])

    data_loader_half1 = DataLoader(
        dataset=dataset_half1,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True
    )
    data_loader_half2 = DataLoader(
        dataset=dataset_half2,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True
    )
else:
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=8,
        num_workers=8,
        shuffle=True,
        pin_memory=True
    )

batch = next(iter(data_loader_half1))
data_preprocessor = ParticleImagePreprocessor()
data_preprocessor.initialize_from_stack(
    stack=batch["image"],
    circular_mask_radius=diameter_ang / (2 * optics_group['pixel_size']),
    circular_mask_thickness=circular_mask_thickness / optics_group['pixel_size']
)

box_size = optics_group['image_size']
ang_pix = optics_group['pixel_size']

pixel_distance = 1 / box_size
sm_width = 2

gv = torch.linspace(-0.5, 0.5, box_size)
G = torch.meshgrid(gv, gv, gv)
grid = torch.stack([G[0].ravel(), G[1].ravel(), G[2].ravel()], 1)


def get_ess_grid(grid, points, edge=600):
    tree = scipy.spatial.KDTree(points.cpu().numpy())
    (dists, points) = tree.query(grid.cpu().numpy())
    ess_grid = grid[dists < edge / box_size]
    out_grid = grid[dists >= edge / box_size]
    return ess_grid, out_grid


if args.reconstruction_area:
    with mrcfile.open(args.reconstruction_area) as mrc:
        rec_area = torch.tensor(mrc.data).to(device)
    print('generating smooth mask')
    bin_mask = (rec_area > 0).float()
    sm_bin_mask = torch.nn.functional.conv3d(bin_mask.unsqueeze(0).unsqueeze(0),
                                             torch.ones(1, 1, sm_width, sm_width, sm_width).to(
                                                 device) / (sm_width ** 3), padding='same')
    sm_bin_mask = sm_bin_mask.squeeze()
    print('smooth mask generated')
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
else:
    ess_grid, out_grid = get_ess_grid(grid, points)
    ess_grid = torch.tensor(ess_grid).to(device)
    out_grid = torch.tensor(out_grid).to(device)
    ess_grid_int = ((ess_grid + 0.5) * box_size).long()
    out_grid_int = ((out_grid + 0.5) * box_size).long()

decoder_half1.n_points = ess_grid.shape[0]
decoder_half2.n_points = ess_grid.shape[0]


class fwd_deformation_interpolator:
    def __init__(self, device, grid, points, box_size, neighbours, p):
        super(fwd_deformation_interpolator, self).__init__()
        self.grid = grid.cpu().numpy()
        self.int_grid = torch.tensor(np.round((self.grid + 0.5) * (box_size - 1))).long()
        self.neighbours = neighbours
        self.p = p
        self.box_size = box_size
        self.pointtree = scipy.spatial.KDTree(points.cpu().numpy())
        self.dists, self.pos = self.pointtree.query(self.grid, k=self.neighbours)
        self.coeff = torch.tensor(1 / self.dists ** p)
        self.coeff /= torch.sum(self.coeff, 1).unsqueeze(1)

    def compute_field(self, values):
        dispM = 2 * torch.ones(self.box_size + 1, self.box_size + 1, self.box_size + 1, 3)
        values = values.cpu()
        disp = self.coeff[:, 0, None] * values[self.pos[:, 0]] + self.coeff[:, 1, None] * values[
            self.pos[:, 1]] + self.coeff[:, 2, None] * values[self.pos[:, 2]]
        disp = torch.tensor(self.grid) - disp + torch.tensor(self.grid)
        values = torch.tensor(self.grid) - values + torch.tensor(self.grid)
        dispM[self.int_grid[:, 2], self.int_grid[:, 1], self.int_grid[:, 0], :] = values[:,
                                                                                  :].float()
        # dispM[self.int_grid[:,2],self.int_grid[:,1],self.int_grid[:,0],:] = disp[:,:].float()

        return dispM * 2


class fwd_deformation_interpolator_ds:
    def __init__(self, device, grid, points, box_size, ds, neighbours, p):
        super(fwd_deformation_interpolator_ds, self).__init__()
        self.grid = grid.cpu().numpy()
        self.int_grid = torch.tensor(np.round((self.grid + 0.5) * (box_size - 1)) // ds).long()
        self.neighbours = neighbours
        self.p = p
        self.box_size = box_size
        self.pointtree = scipy.spatial.KDTree(points.cpu().numpy())
        self.dists, self.pos = self.pointtree.query(self.grid, k=self.neighbours)
        self.coeff = torch.tensor(1 / self.dists ** p)
        self.coeff /= torch.sum(self.coeff, 1).unsqueeze(1)
        self.ds = ds

    def compute_field(self, values):
        size = self.box_size // ds
        dispM = 2 * torch.ones(1, 3, size, size, size)
        values = values.cpu()
        values = torch.tensor(self.grid) - values + torch.tensor(self.grid)
        dispM[0, :, self.int_grid[:, 2], self.int_grid[:, 1], self.int_grid[:, 0]] = values.movedim(
            0, 1).float()
        dispL = torch.nn.functional.upsample(dispM, scale_factor=self.ds, mode='trilinear')
        # dispM[self.int_grid[:,2],self.int_grid[:,1],self.int_grid[:,0],:] = disp[:,:].float()

        return dispL[0] * 2


class bwd_deformation_interpolator:
    def __init__(self, device, grid, box_size, neighbours, p):
        super(bwd_deformation_interpolator, self).__init__()
        self.grid = grid.cpu().numpy()
        self.int_grid = torch.tensor(np.round((self.grid + 0.5) * (box_size - 1))).long()
        self.neighbours = neighbours
        self.p = p
        self.box_size = box_size

    def compute_field(self, points, values):
        dispM = torch.zeros(self.box_size + 1, self.box_size + 1, self.box_size + 1, 3)
        pointtree = scipy.spatial.KDTree(points.cpu().numpy())
        dists, pos = pointtree.query(self.grid, k=self.neighbours)
        coeff = torch.tensor(1 / dists ** self.p)
        coeff /= torch.sum(coeff, 1).unsqueeze(1)
        values = values.cpu()
        disp = coeff[:, 0, None] * values[pos[:, 0]] + coeff[:, 1, None] * values[
            pos[:, 1]] + coeff[:, 2, None] * values[pos[:, 2]]
        dispM[self.int_grid[:, 2], self.int_grid[:, 1], self.int_grid[:, 0], :] = disp[:, :].float()

        return dispM * 2


def deform_volume(V, ess_grid, out_grid, ind_ess_grid, box_size, vol_type='volume'):
    Vnew = torch.zeros_like(V)
    V = V.movedim(1, 2).movedim(3, 2).movedim(2, 1)
    batch_size = V.shape[0]
    # ess_grid_int = torch.round((ess_grid+0.5)*box_size).long()
    ess_grid_int = ess_grid.long()
    print(ess_grid_int.dtype)
    out_grid_int = torch.round((out_grid + 0.5) * (box_size - 1)).long()
    ind_ess_grid = torch.round((ind_ess_grid + 0.5) * (box_size - 1)).long()
    # ind_ess_grid_flat = ind_ess_grid[:,0]*box_size**2+ind_ess_grid[:,1]*box_size+ind_ess_grid[:,2]
    # V_flat = V.flatten()
    # Vnew = V.scatter_add()
    # ind_ess_grid = torch.clip(torch.round((ess_grid+defo+0.5)*box_size),min = 0, max = box_size).long()
    # ind_ess_grid = torch.clip(torch.round((ess_grid+defo[0]+0.5)*box_size),min = 0, max = box_size).long()
    # for i in range(batch_size):
    #    Vnew[i,ess_grid_int[:,0],ess_grid_int[:,1],ess_grid_int[:,2]] = V[i,ind_ess_grid[:,0],ind_ess_grid[:,1],ind_ess_grid[:,2]]
    #    Vnew[i,out_grid_int[:,0],out_grid_int[:,1],out_grid_int[:,2]] = V[i,out_grid_int[:,0],out_grid_int[:,1],out_grid_int[:,2]]
    for i in range(batch_size):
        if vol_type == 'volume':
            # Vnew[:,out_grid_int[:,0],out_grid_int[:,1],out_grid_int[:,2]] = V[:,out_grid_int[:,0],out_grid_int[:,1],out_grid_int[:,2]] #set values uotside the masked region to the original values(noise)
            # Vnew[:,ind_ess_grid[:,0],ind_ess_grid[:,1],ind_ess_grid[:,2]] = 0 #V[:,ess_grid_int[:,0],ess_grid_int[:,1],ess_grid_int[:,2]] #set values for voxels where density is taken to zero or switch
            Vnew[i, ess_grid_int[:, 0], ess_grid_int[:, 1], ess_grid_int[:, 2]] += V[
                i, ind_ess_grid[:, 0], ind_ess_grid[:, 1], ind_ess_grid[:,
                                                           2]]  # substitute deformed values
            Vnew[i, out_grid_int[:, 0], out_grid_int[:, 1], out_grid_int[:, 2]] += V[
                i, out_grid_int[:, 0], out_grid_int[:, 1], out_grid_int[:, 2]]
        elif vol_type == 'ctf':
            Vnew[i, ess_grid_int[:, 0], ess_grid_int[:, 1], ess_grid_int[:, 2]] += V[
                i, ind_ess_grid[:, 0], ind_ess_grid[:, 1], ind_ess_grid[:, 2]]
            Vnew[i, out_grid_int[:, 0], out_grid_int[:, 1], out_grid_int[:, 2]] += V[
                i, out_grid_int[:, 0], out_grid_int[:, 1], out_grid_int[:, 2]]
    # Vnew[:,out_grid_int[:,0],out_grid_int[:,1],out_grid_int[:,2]] = V[:,out_grid_int[:,0],out_grid_int[:,1],out_grid_int[:,2]]

    return Vnew.movedim(1, 2).movedim(3, 2).movedim(2, 1)


spatgrad = spatial_grad(spatial_grad, box_size)


def get_grad(ess_grid, ind_ess_grid, box_size):
    device = ind_ess_grid.device
    gx = torch.zeros(box_size, box_size, box_size).to(device)
    gy = torch.zeros(box_size, box_size, box_size).to(device)
    gz = torch.zeros(box_size, box_size, box_size).to(device)
    ind_ess_grid = ind_ess_grid * (box_size - 1)
    ess_grid_int = ess_grid.long()

    gx[ess_grid_int[:, 0], ess_grid_int[:, 1], ess_grid_int[:, 2]] = ind_ess_grid[:, 0].float()
    gy[ess_grid_int[:, 0], ess_grid_int[:, 1], ess_grid_int[:, 2]] = ind_ess_grid[:, 1].float()
    gz[ess_grid_int[:, 0], ess_grid_int[:, 1], ess_grid_int[:, 2]] = ind_ess_grid[:, 2].float()

    gx = spatgrad(gx.unsqueeze(0).unsqueeze(0))
    gy = spatgrad(gy.unsqueeze(0).unsqueeze(0))
    gz = spatgrad(gz.unsqueeze(0).unsqueeze(0))

    det = gx[0] * gy[1] * gz[2] - gx[0] * gy[2] * gz[1] - gx[1] * gy[0] * gz[2] + gx[1] * gy[2] * \
          gz[0] + gx[2] * gy[0] * gz[1] - gx[2] * gy[1] * gz[0]
    # d = det[ess_grid_int[:,0],ess_grid_int[:,1],ess_grid_int[:,2]]
    return det  # torch.clip(torch.abs(det).movedim(0,1).movedim(2,1).movedim(1,0),max = 100)


def get_grad_from_field(disp, box_size):
    gx = spatgrad((box_size * disp[:, :, :, 0] / 2).unsqueeze(0).unsqueeze(0))
    gy = spatgrad((box_size * disp[:, :, :, 1] / 2).unsqueeze(0).unsqueeze(0))
    gz = spatgrad((box_size * disp[:, :, :, 2] / 2).unsqueeze(0).unsqueeze(0))

    det = gx[0] * gy[1] * gz[2] - gx[0] * gy[2] * gz[1] - gx[1] * gy[0] * gz[2] + gx[1] * gy[2] * \
          gz[0] + gx[2] * gy[0] * gz[1] - gx[2] * gy[1] * gz[0]
    # det[torch.abs(det)>5] = 1
    return torch.abs(det)


def deform_volume_new(V, ess_grid, out_grid, defo, box_size):
    # input Volume, integer grid points for deformable_backprojection, outer grid points which stay the same, raw deformation vector
    Vnew = torch.zeros_like(V)
    V = V.movedim(1, 2).movedim(3, 2).movedim(2, 1)
    batch_size = defo.shape[0]
    # ess_grid_int = torch.round((ess_grid+0.5)*box_size).long()
    ess_grid_int = ess_grid_int.long()
    out_grid_int = torch.round((out_grid + 0.5) * box_size).long()
    # ind_ess_grid = torch.clip(torch.round((ess_grid+defo+0.5)*box_size),min = 0, max = box_size).long()

    for i in range(batch_size):
        def_ess_grid = (ess_grid + defo[i] + 0.5) * box_size
        ind_ess_grid = torch.floor(def_ess_grid).long()
        vol = torch.zeros_like(def_ess_grid[:, 0]).unsqueeze(1)
        x, y, z = ind_ess_grid.split(1, dim=-1)
        rxyz = def_ess_grid - ind_ess_grid
        rx, ry, rz = rxyz.split(1, dim=-1)
        values = torch.zeros_like(def_ess_grid)
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

                    valid = ((0 <= x_) * (x_ < box_size) * (0 <= y_) * (y_ < box_size) * (
                            0 <= z_) * (z_ < box_size)).long()
                    w = (w * valid.type_as(w))
                    values = V[i, x_, y_, z_]
                    vol += w * values

        # ind_ess_grid = torch.clip(torch.round((ess_grid+defo[i]+0.5)*box_size),min = 0, max = box_size).long()
        Vnew[i, ess_grid_int[:, 0], ess_grid_int[:, 1], ess_grid_int[:, 2]] = vol.squeeze()
        Vnew[i, out_grid_int[:, 0], out_grid_int[:, 1], out_grid_int[:, 2]] = V[
            i, out_grid_int[:, 0], out_grid_int[:, 1], out_grid_int[:, 2]]

    return Vnew.movedim(1, 2).movedim(3, 2).movedim(2, 1)


class rotate_volume(nn.Module):

    def __init__(self, box_size, device):
        super(rotate_volume, self).__init__()
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
        R = torch.cat([R, torch.zeros([batch_size, 1, 3, 1]).to(self.device)], dim=3).squeeze()
        if R.dim() == 2:
            R = R.unsqueeze(0)
        G = torch.nn.functional.affine_grid(R, (
        batch_size, 1, self.box_size + 1, self.box_size + 1, self.box_size + 1),
                                            align_corners=False).float()
        VV = F.grid_sample(input=V, grid=G.to(self.device), mode='bilinear', align_corners=False)

        return VV


# def get_inverse_deformation(dis,grid,p2v):
#     new_grid = grid+dis
#     new_dis = -dis
#     grid_dis = p2v(new_grid,new_dis)
#     gradx = torch.gradient(grid_dis,)


def generate_filter(box_size, filter_type='ramlak', dimension=2):
    gv = torch.linspace(-0.5, 0.5, box_size)
    G = torch.meshgrid(gv, gv, gv)
    grid = torch.stack([G[0].flatten(), G[1].flatten(), G[2].flatten()], 1)
    if dimension == 2:
        kx = torch.linspace(-box_size / 2, box_size / 2 - 1, box_size)
        GG = torch.meshgrid(kx, kx)
        R = torch.fft.fftshift(torch.pow(GG[0] ** 2 + GG[1] ** 2, 0.5), dim=[-1, -2]).to(device)
        mask = R > box_size / 2
        if filter_type == 'ramlak':
            filt = R
        elif filter_type == 'cosine':
            filt = torch.fft.fftshift(R * torch.cos(np.pi * R / box_size), dim=[-1, -2]).to(device)
        elif filter_type == 'shepplogan':
            filt = torch.fft.fftshift(R * torch.sinc(R / box_size), dim=[-1, -2]).to(device)
        filt[R > box_size / 2] = 0
    elif dimension == 3:
        kx = torch.linspace(-box_size / 2, box_size / 2 - 1, box_size)
        GG = torch.meshgrid(kx, kx, kx)
        R = torch.fft.fftshift(torch.pow(GG[0] ** 2 + GG[1] ** 2 + GG[2] ** 2, 0.5),
                               dim=[-1, -2, -3]).to(device)
        mask = R > box_size / 2
        if filter_type == 'ramlak':
            filt = R
        elif filter_type == 'cosine':
            filt = torch.fft.fftshift(R * torch.cos(np.pi * R / box_size), dim=[-1, -2, -3]).to(
                device)
        elif filter_type == 'shepplogan':
            filt = torch.fft.fftshift(R * torch.sinc(R / box_size), dim=[-1, -2, -3]).to(device)
        filt[R > box_size / 2] = 0

    return filt, mask


# learning the inverse deformation:


inv_half1 = inverse_displacement(device, latent_dim, N_points, 6, 96, lin_block, 6, box_size).to(
    device)
inv_half2 = inverse_displacement(device, latent_dim, N_points, 6, 96, lin_block, 6, box_size).to(
    device)
inv_half1_params = inv_half1.parameters()
inv_half2_params = inv_half2.parameters()
inv_half1_params = add_weight_decay(inv_half1, weight_decay=0.3)
inv_half2_params = add_weight_decay(inv_half2, weight_decay=0.3)
inv_half1_optimizer = torch.optim.Adam(inv_half1_params, lr=5e-4)
inv_half2_optimizer = torch.optim.Adam(inv_half2_params, lr=5e-4)

N_inv = 100
inv_loss_h1 = 0
inv_loss_h2 = 0

losslist = torch.zeros(N_inv, 2)

if args.random_half:
    inds_half1 = cp['indices_half1'].cpu().numpy()
    inds_half2 = list(set(range(len(dataset))) - set(list(inds_half1)))
    print(len(inds_half1))
    print(len(inds_half2))
    dataset_half1 = torch.utils.data.Subset(dataset, inds_half1)
    dataset_half2 = torch.utils.data.Subset(dataset, inds_half2)

    # dataset_half1, dataset_half2 = torch.utils.data.dataset.random_split(dataset,[len(dataset)//2,len(dataset)-len(dataset)//2])

    data_loader_half1 = DataLoader(
        dataset=dataset_half1,
        batch_size=100,
        num_workers=8,
        shuffle=True,
        pin_memory=True
    )
    data_loader_half2 = DataLoader(
        dataset=dataset_half2,
        batch_size=100,
        num_workers=8,
        shuffle=True,
        pin_memory=True
    )

mu_in_tot = torch.zeros(len(dataset), latent_dim)
pos_tot = torch.zeros(len(dataset), N_points, 3)
gdist = gv[1] - gv[0]
gs = torch.linspace(-0.5, 0.5, box_size // ds)
Gs = torch.meshgrid(gs, gs, gs)
smallgrid = torch.stack([Gs[0].ravel(), Gs[1].ravel(), Gs[2].ravel()], 1)
_, outpos = get_ess_grid(smallgrid, cons_model.pos, edge=10)

for epochs in range(N_inv):
    print('Inversion loss for half 1 at iteration', epochs, inv_loss_h1)
    inv_loss_h1 = 0
    print('Inversion loss for half 2 at iteration', epochs, inv_loss_h2)
    inv_loss_h2 = 0
    for batch_ndx, sample in enumerate(data_loader_half1):
        inv_half1_optimizer.zero_grad()
        if epochs > 0 and args.noise == 'False':
            # print('FUCK YOU!')
            idx = sample['idx']
            c_pos = inv_half1([mu_in_tot[idx].to(device)], pos_tot[idx].to(device))
        else:
            with torch.no_grad():
                r, y, ctf, t_in = sample["rotation"], sample["image"], sample["ctf"], -sample[
                    'translation']
                idx = sample['idx']
                r, t = poses(idx)
                batch_size = y.shape[0]
                ctfs_l = torch.nn.functional.pad(ctf, (
                box_size // 2, box_size // 2, box_size // 2, box_size // 2, 0, 0)).to(device)
                ctfs = torch.fft.fftshift(ctf, dim=[-1, -2])
                y_in = data_preprocessor.apply_square_mask(y)
                y_in = data_preprocessor.apply_translation(y_in, -t[:, 0], -t[:, 1])
                y_in = data_preprocessor.apply_circular_mask(y_in)
                mu, sig = encoder_half1(y_in.to(device), ctfs.to(device))
                # mu_in = [mu]
                mu_in = [mu + 0 * torch.randn_like(mu)]
                if args.noise == 'True':
                    noise_real = (5 / (ang_pix * box_size)) * torch.randn_like(cons_model.pos).to(
                        device)
                    # out_points = outpos[torch.randint_like(cons_model.pos[:,0],outpos.shape[0]).long()].to(device)
                    # evalpos = torch.cat([cons_model.pos+noise_real,out_points])
                    evalpos = cons_model.pos + noise_real
                    proj, proj_im, proj_pos, pos, dis = decoder_half1.forward(mu_in, r.to(device),
                                                                              evalpos.to(device),
                                                                              torch.ones_like(
                                                                                  evalpos[:, 0]),
                                                                              torch.ones(n_classes,
                                                                                         evalpos.shape[
                                                                                             0]).to(
                                                                                  device),
                                                                              t.to(device))
                else:
                    proj, proj_im, proj_pos, pos, dis = decoder_half1.forward(mu_in, r.to(device),
                                                                              cons_model.pos.to(
                                                                                  device),
                                                                              cons_model.amp.to(
                                                                                  device),
                                                                              cons_model.ampvar.to(
                                                                                  device),
                                                                              t.to(device))
                    mu_in_tot[idx] = mu.cpu()
                    pos_tot[idx] = pos.cpu()
            c_pos = inv_half1(mu_in, pos)

        if args.noise == 'True':
            loss = torch.sum((c_pos - cons_model.pos.to(device) + noise_real) ** 2)
        else:
            loss = torch.sum((c_pos - cons_model.pos.to(device)) ** 2)
        loss.backward()
        inv_half1_optimizer.step()
        inv_loss_h1 += loss.item()

    losslist[epochs, 0] = inv_loss_h1

    for batch_ndx, sample in enumerate(data_loader_half2):
        inv_half2_optimizer.zero_grad()
        if epochs > 0 and args.noise == 'False':
            idx = sample['idx']
            c_pos = inv_half2([mu_in_tot[idx].to(device)], pos_tot[idx].to(device))
        else:
            with torch.no_grad():
                r, y, ctf, t_in = sample["rotation"], sample["image"], sample["ctf"], -sample[
                    'translation']
                idx = sample['idx']
                r, t = poses(idx)
                batch_size = y.shape[0]
                ctfs_l = torch.nn.functional.pad(ctf, (
                box_size // 2, box_size // 2, box_size // 2, box_size // 2, 0, 0)).to(device)
                ctfs = torch.fft.fftshift(ctf, dim=[-1, -2])
                y_in = data_preprocessor.apply_square_mask(y)
                y_in = data_preprocessor.apply_translation(y_in, -t[:, 0], -t[:, 1])
                y_in = data_preprocessor.apply_circular_mask(y_in)
                mu, sig = encoder_half2(y_in.to(device), ctfs.to(device))
                # mu_in = [mu]
                mu_in = [mu + 0 * torch.randn_like(mu)]
                if args.noise == 'True':
                    noise_real = (5 / (ang_pix * box_size)) * torch.randn_like(cons_model.pos).to(
                        device)
                    # out_points = outpos[torch.randint_like(cons_model.pos[:,0],outpos.shape[0]).long()].to(device)
                    # evalpos = torch.cat([cons_model.pos+noise_real,out_points])
                    evalpos = cons_model.pos + noise_real
                    proj, proj_im, proj_pos, pos, dis = decoder_half1.forward(mu_in, r.to(device),
                                                                              evalpos.to(device),
                                                                              torch.ones_like(
                                                                                  evalpos[:, 0]),
                                                                              torch.ones(n_classes,
                                                                                         evalpos.shape[
                                                                                             0]).to(
                                                                                  device),
                                                                              t.to(device))
                else:
                    proj, proj_im, proj_pos, pos, dis = decoder_half2.forward(mu_in, r.to(device),
                                                                              cons_model.pos.to(
                                                                                  device),
                                                                              cons_model.amp.to(
                                                                                  device),
                                                                              cons_model.ampvar.to(
                                                                                  device),
                                                                              t.to(device))
                    mu_in_tot[idx] = mu.cpu()
                    pos_tot[idx] = pos.cpu()
            c_pos = inv_half2(mu_in, pos)
        if args.noise == 'True':
            # loss = torch.sum((c_pos-torch.cat([(cons_model.pos.to(device)+noise_real),out_points],0))**2)
            loss = torch.sum((c_pos - cons_model.pos.to(device) + noise_real) ** 2)
        else:
            loss = torch.sum((c_pos - cons_model.pos.to(device)) ** 2)
        loss.backward()
        inv_half2_optimizer.step()
        inv_loss_h2 += loss.item()
    losslist[epochs, 1] = inv_loss_h2


del mu_in_tot, pos_tot

checkpoint = {'inv_half1': inv_half1, 'inv_half2': inv_half2,
              'inv_half1_state_dict': inv_half1.state_dict(),
              'inv_half2_state_dict': inv_half2.state_dict()}
torch.save(checkpoint, '/cephfs/schwab/inv_checkpoint_noise_noreg_molly.pth')

if args.random_half:
    inds_half1 = cp['indices_half1'].cpu().numpy()
    inds_half2 = list(set(range(len(dataset))) - set(list(inds_half1)))
    print(len(inds_half1))
    print(len(inds_half2))
    dataset_half1 = torch.utils.data.Subset(dataset, inds_half1)
    dataset_half2 = torch.utils.data.Subset(dataset, inds_half2)

    # dataset_half1, dataset_half2 = torch.utils.data.dataset.random_split(dataset,[len(dataset)//2,len(dataset)-len(dataset)//2])

    data_loader_half1 = DataLoader(
        dataset=dataset_half1,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True
    )
    data_loader_half2 = DataLoader(
        dataset=dataset_half2,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True
    )

if args.random_half:
    print('Computing latent_space and indices for half 1')
    latent_space = torch.zeros(len(dataset), latent_dim)
    latent_indices_half1 = []

    for batch_ndx, sample in enumerate(tqdm(data_loader_half1)):
        with torch.no_grad():
            r, y, ctf, t_in = sample["rotation"], sample["image"], sample["ctf"], -sample[
                'translation']
            idx = sample['idx']
            r, t = poses(idx)
            batch_size = y.shape[0]
            ctfs_l = torch.nn.functional.pad(ctf, (
            box_size // 2, box_size // 2, box_size // 2, box_size // 2, 0, 0)).to(device)
            ctfs = torch.fft.fftshift(ctf, dim=[-1, -2])
            y_in = data_preprocessor.apply_square_mask(y)
            y_in = data_preprocessor.apply_translation(y_in, -t[:, 0], -t[:, 1])
            y_in = data_preprocessor.apply_circular_mask(y_in)
            mu, _ = encoder_half1(y_in.to(device), ctfs.to(device))
            mu_in = [mu]
            latent_space[sample["idx"].cpu().numpy()] = mu.detach().cpu()
            latent_indices_half1.append(sample['idx'])

    print('Computing latent_space and indices for half2')
    latent_indices_half2 = []

    for batch_ndx, sample in enumerate(tqdm(data_loader_half2)):
        with torch.no_grad():
            r, y, ctf, t_in = sample["rotation"], sample["image"], sample["ctf"], -sample[
                'translation']
            idx = sample['idx']
            r, t = poses(idx)
            batch_size = y.shape[0]
            ctfs_l = torch.nn.functional.pad(ctf, (
            box_size // 2, box_size // 2, box_size // 2, box_size // 2, 0, 0)).to(device)
            ctfs = torch.fft.fftshift(ctf, dim=[-1, -2])
            y_in = data_preprocessor.apply_square_mask(y)
            y_in = data_preprocessor.apply_translation(y_in, -t[:, 0], -t[:, 1])
            y_in = data_preprocessor.apply_circular_mask(y_in)
            mu, _ = encoder_half2(y_in.to(device), ctfs.to(device))
            mu_in = [mu]
            latent_space[sample["idx"].cpu().numpy()] = mu.detach().cpu()
            latent_indices_half2.append(sample['idx'])

    diam = torch.zeros(1)
    xmin = torch.zeros(latent_dim)

    for i in range(latent_dim):
        xmin[i] = torch.min(latent_space[:, i])

        xmax = torch.max(latent_space[:, i])
        diam = torch.maximum(xmax - xmin[i], diam)

    max_side = diam
    xx = []
    for i in range(latent_dim):
        xx.append(torch.linspace(xmin[i], xmin[i] + max_side[0], args.latent_sampling))

    latent_indices_half1 = torch.cat(latent_indices_half1, 0)
    latent_indices_half2 = torch.cat(latent_indices_half2, 0)

    latent_space_half1 = latent_space[latent_indices_half1]
    latent_space_half2 = latent_space[latent_indices_half2]

    XY = torch.meshgrid(xx)
    gxy = torch.stack([X.ravel() for X in XY], 1)

    tree = scipy.spatial.KDTree(gxy.cpu().numpy())
    (dists_half1, latent_points_half1) = tree.query(latent_space_half1.cpu().numpy(), p=1)
    (dists_half2, latent_points_half2) = tree.query(latent_space_half2.cpu().numpy(), p=1)


else:

    print('Computing latent space and indices for the complete dataset')

    latent_space = torch.zeros(len(dataset), latent_dim)
    latent_indices = torch.zeros(len(dataset)).long()

    for batch_ndx, sample in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            r, y, ctf, t_in = sample["rotation"], sample["image"], sample["ctf"], -sample[
                'translation']
            idx = sample['idx']
            r, t = poses(idx)
            batch_size = y.shape[0]
            ctfs_l = torch.nn.functional.pad(ctf, (
            box_size // 2, box_size // 2, box_size // 2, box_size // 2, 0, 0)).to(device)
            ctfs = torch.fft.fftshift(ctf, dim=[-1, -2])
            y_in = data_preprocessor.apply_square_mask(y)
            y_in = data_preprocessor.apply_translation(y_in, -t[:, 0], -t[:, 1])
            y_in = data_preprocessor.apply_circular_mask(y_in)
            mu, _ = encoder(y_in.to(device), ctfs.to(device))
            mu_in = [mu]
            latent_space[sample["idx"].cpu().numpy()] = mu.detach().cpu()
            latent_indices[sample['idx'].cpu().numpy()] = sample['idx']

    print('Tiling latent space')

    xmin = torch.min(latent_space[:, 0])
    xmax = torch.max(latent_space[:, 0])
    ymin = torch.min(latent_space[:, 1])
    ymax = torch.max(latent_space[:, 1])

    max_side = torch.maximum(xmax - xmin, ymax - ymin)

    xx = torch.linspace(xmin, xmin + max_side, args.latent_sampling)
    yy = torch.linspace(ymin, ymin + max_side, args.latent_sampling)

    dxy = (xx[1] - xx[0]) / 2

    XY = torch.meshgrid(xx, yy)

    gxy = torch.stack([XY[0].ravel(), XY[1].ravel()], 1)

    tree = scipy.spatial.KDTree(gxy.cpu().numpy())
    (dists, latent_points) = tree.query(latent_space.cpu().numpy(), p=1)

lam_thres = box_size ** 3 / 250 ** 3

V = torch.zeros(box_size, box_size, box_size).to(device)
gv = torch.linspace(-0.5, 0.5, box_size)
G = torch.meshgrid(gv, gv, gv)
grid = torch.stack([G[0].flatten(), G[1].flatten(), G[2].flatten()], 1)
kx = torch.linspace(-box_size / 2, box_size / 2 - 1, box_size)
GG = torch.meshgrid(kx, kx)

fx = torch.linspace(-5, 5, 10)
[FX, FY, FZ] = torch.meshgrid(fx, fx, fx)
fsig = 1
ker = torch.exp(-(FX ** 2 + FY ** 2 + FZ ** 2) / fsig)
ker /= torch.linalg.norm(ker)
# ker[ker>0.1] = 1
ker /= torch.sum(ker)
ker = ker.unsqueeze(0).unsqueeze(0)

filt, mask2 = generate_filter(box_size, filter_type='ramlak')

GG3 = torch.meshgrid(gv, gv, gv)
mask3 = (GG3[0] ** 2 + GG3[1] ** 2 + GG3[2] ** 2) < 0.47

if args.filter != 'ctf':
    GG3 = torch.meshgrid(kx, kx, kx)
    filt3, mask3 = generate_filter(box_size, filter_type='ramlak', dimension=3)

i = 0
cons_model.amp = torch.nn.Parameter(torch.ones(1), requires_grad=False)

smallgrid, outsmallgrid = get_ess_grid(smallgrid, cons_model.pos)

gss = torch.linspace(-0.5, 0.5, box_size // 8)
Gss = torch.meshgrid(gss, gss, gss)
supersmallgrid = torch.stack([Gss[0].ravel(), Gss[1].ravel(), Gss[2].ravel()], 1)

fwd_int = fwd_deformation_interpolator_ds(device, smallgrid, smallgrid, box_size, ds, 3, 2)
# bwd_int = bwd_deformation_interpolator(device, ess_grid, box_size, 3, 2)
id_trans = torch.eye(3, 4).unsqueeze(0)
id_grid = torch.nn.functional.affine_grid(id_trans, (1, 1, box_size, box_size, box_size),
                                          align_corners=True)

rotation = rotate_volume(box_size, device)
gridlen = ess_grid.shape[0]
CTF = torch.zeros(box_size, box_size, box_size).to(device)
tdet = torch.zeros(box_size + 1, box_size + 1, box_size + 1).to(device)
om_imgs = torch.zeros(1)
bp_imgs = torch.zeros(1)
if args.random_half:
    his = []
    rel_inds = []
    print('start deformable_backprojection of half 1')
    print('computing relevant tiles')
    for j in tqdm(range(gxy.shape[0])):
        tile_indices = latent_indices_half1[latent_points_half1 == j]
        if len(tile_indices) > 0:
            his.append(len(tile_indices))
            rel_inds.append(j)
    # plt.hist(his,bins = 100)
    # plt.show()
    print(np.sum(np.array(his) == 1), 'tiles with single particles')
    print('tile with the most images has ', np.max(np.array(his)), 'particles')

    for j in rel_inds:
        tile_indices = latent_indices_half1[latent_points_half1 == j]
        current_data = torch.utils.data.Subset(dataset, tile_indices)
        current_data_loader = DataLoader(
            dataset=current_data,
            batch_size=8,
            num_workers=8,
            shuffle=False,
            pin_memory=False
        )
        z_tile = [torch.stack(2 * [gxy[j]], 0).to(device)]
        print(j, len(tile_indices))
        r = torch.zeros(2, 3)
        t = torch.zeros(2, 2)

        if len(tile_indices) > args.ignore:
            with torch.no_grad():
                _, _, _, n_pos, deformation = decoder_half1(z_tile, r.to(device),
                                                            supersmallgrid.to(device),
                                                            cons_model.amp.to(device),
                                                            torch.ones(n_classes,
                                                                       supersmallgrid.shape[0]).to(
                                                                device), t.to(device))
                tile_deformation = inv_half1(z_tile, torch.stack(2 * [smallgrid.to(device)], 0))

                disp = fwd_int.compute_field(tile_deformation[0])
                # disp0 = torch.nn.functional.conv3d(disp[...,0].unsqueeze(0).unsqueeze(0),ker,padding = 'same')
                # disp1 = torch.nn.functional.conv3d(disp[...,1].unsqueeze(0).unsqueeze(0),ker,padding = 'same')
                # disp2 = torch.nn.functional.conv3d(disp[...,2].unsqueeze(0).unsqueeze(0),ker,padding = 'same')
                disp0 = disp[0, ...]
                disp1 = disp[1, ...]
                disp2 = disp[2, ...]
                disp = torch.stack([disp0.squeeze(), disp1.squeeze(), disp2.squeeze()], 3)
                # detjac = get_grad_from_field(disp,box_size)
                dis_grid = disp[None, :, :, :]
                if i % 100 == 0:
                    im_deformation = inv_half1(z_tile, n_pos)
                    field2bild(n_pos[0], im_deformation[0],
                               '/cephfs/schwab/fields_mon/Ninversefield' + str(i).zfill(3),
                               box_size, ang_pix)
                    field2bild(supersmallgrid.to(device), n_pos[0],
                               '/cephfs/schwab/fields_mon/Nforwardfield' + str(i).zfill(3),
                               box_size, ang_pix)
                    field2bild(supersmallgrid.to(device), im_deformation[0],
                               '/cephfs/schwab/fields_mon/Nfwdbwdfield' + str(i).zfill(3), box_size,
                               ang_pix)
                    print('saved field')

                ind_ess_grid = torch.clip(torch.round((tile_deformation[0] + 0.5) * (box_size - 1)),
                                          min=0, max=box_size - 1).long()
                in_grid = torch.clip((ess_grid + 0.5) * (box_size - 1), min=0,
                                     max=box_size - 1).long()

                print(ind_ess_grid.shape, ind_ess_grid.dtype)
                # det = torch.nn.functional.conv3d(det.unsqueeze(0).unsqueeze(0),ker.to(device),padding = 'same')
                # det[det>5] = 1
                # det = det.unsqueeze(0).unsqueeze(0)

                # det[det == 0] = 1
                # det = torch.clip(det,min = 0.5, max = 10)
                # _,counts = torch.unique(ind_ess_grid,return_counts = True, dim = 1)
                print('Backprojected', bp_imgs.cpu().numpy(), 'particles from', len(dataset),
                      'and omitted backprojection of', om_imgs.cpu().numpy(), 'particles')
                # print('Duplicate indices:', torch.sum(counts-1).cpu().numpy())
                bp_imgs += len(tile_indices)

                # del tile_deformation
        else:
            om_imgs += len(tile_indices)
        if len(tile_indices) > args.ignore:
            print('Starting backprojection of tile', j, 'from', gxy.shape[0], '::',
                  len(tile_indices), 'images')
            i = i + 1
            if i % 50 == 0:
                try:
                    VV = V[:, :, :] * sm_bin_mask
                except:
                    VV = V[:, :, :]
                VV = torch.fft.fftn(VV, dim=[-1, -2, -3])
                if args.filter == 'ctf':
                    VV2 = torch.real(
                        torch.fft.ifftn(VV / torch.maximum(CTF, lam_thres * torch.ones_like(CTF)),
                                        dim=[-1, -2, -3]))

                with mrcfile.new(
                    '/cephfs/schwab/backprojection/volumes14/reconstruction_half1_' + str(
                        i // 50).zfill(3) + '.mrc', overwrite=True) as mrc:
                    mrc.set_data((VV2 / torch.max(VV2)).float().cpu().numpy())

            with torch.no_grad():
                for batch_ndx, sample in enumerate(tqdm(current_data_loader)):
                    r, y, ctf, t_in = sample["rotation"], sample["image"], sample["ctf"], -sample[
                        'translation']
                    idx = sample['idx']
                    r, t = poses(idx)
                    batch_size = y.shape[0]
                    ctfs_l = torch.nn.functional.pad(ctf, (
                    box_size // 2, box_size // 2, box_size // 2, box_size // 2, 0, 0)).to(device)
                    ctfs = torch.fft.fftshift(ctf, dim=[-1, -2])
                    y_in = data_preprocessor.apply_square_mask(y)
                    y_in = data_preprocessor.apply_translation(y_in, -t[:, 0], -t[:, 1])
                    y_in = data_preprocessor.apply_circular_mask(y_in)
                    y_in, r, t, ctfs, ctf = y_in.to(device), r.to(device), t.to(device), ctfs.to(
                        device), ctf.to(device)
                    if args.ctf:
                        y = torch.fft.fft2(y_in, dim=[-1, -2])
                        y = y * ctfs
                        yr = torch.real(torch.fft.ifft2(y, dim=[-1, -2])).unsqueeze(1)
                    else:
                        yr = y_in.unsqueeze(1)
                    if args.filter == 'ctf':
                        ctfr2_l = torch.fft.fftshift(torch.real(
                            torch.fft.ifft2(torch.fft.fftshift(ctfs_l, dim=[-1, -2]),
                                            dim=[-1, -2], )).unsqueeze(1), dim=[-1, -2])
                        ctfr2_lc = torch.nn.functional.avg_pool2d(ctfr2_l, kernel_size=3, stride=2,
                                                                  padding=1)
                        CTFy = ctfr2_lc.expand(batch_size, box_size, box_size, box_size)
                        CTFy = torch.nn.functional.pad(CTFy, (0, 1, 0, 1, 0, 1, 0, 0))
                        CTFy = rotation(CTFy.unsqueeze(1), r).squeeze()

                        if len(CTFy.shape) < 4:
                            CTFy = CTFy.unsqueeze(0)

                        # CTFy = deform_volume(CTFy,ess_grid,out_grid,ind_ess_grid,box_size)
                        CTFy = CTFy[:, :-1, :-1, :-1]
                        CTF += (1 / len(dataset)) * torch.real(
                            torch.sum(torch.fft.fftn(CTFy.unsqueeze(1), dim=[-1, -2, -3]) ** 2,
                                      0).squeeze())

                    Vy = yr.expand(batch_size, box_size, box_size, box_size)
                    Vy = torch.nn.functional.pad(Vy, (0, 1, 0, 1, 0, 1, 0, 0))
                    Vy = rotation(Vy.unsqueeze(1), r).squeeze()
                    if len(Vy.shape) < 4:
                        Vy = Vy.unsqueeze(0)
                    # Vy = torch.sum(Vy.unsqueeze(1),0).squeeze()
                    if len(Vy.shape) < 4:
                        Vy = Vy.unsqueeze(0)
                    # Vy = deform_volume(Vy,ind_ess_grid,out_grid,ess_grid,box_size)

                    Vy = torch.sum(Vy, 0)
                    Vy = F.grid_sample(input=Vy.unsqueeze(0).unsqueeze(0), grid=dis_grid.to(device),
                                       mode='bilinear', align_corners=False)
                    V += (1 / len(dataset)) * Vy.squeeze()

    try:
        V = V * sm_bin_mask
    except:
        V = V
    V = torch.fft.fftn(V, dim=[-1, -2, -3])
    if args.filter == 'ctf':
        V = torch.real(torch.fft.ifftn(V / torch.maximum(CTF, lam_thres * torch.ones_like(CTF)),
                                       dim=[-1, -2, -3]))

    else:
        V = torch.real(torch.fft.ifftn(VV * filt3.to(device), dim=[-1, -2, -3]))

    with mrcfile.new(args.out_dir.replace('.mrc', '_half1.mrc'), overwrite=True) as mrc:
        mrc.set_data((V / torch.max(V)).float().detach().cpu().numpy())

    del V, CTF
    V = torch.zeros(box_size, box_size, box_size).to(device)
    CTF = torch.zeros(box_size, box_size, box_size).to(device)
    i = 0
    his = []
    rel_inds = []

    print('start deformable_backprojection of half 2')
    print('computing relevant tiles')
    for j in tqdm(range(gxy.shape[0])):
        tile_indices = latent_indices_half2[latent_points_half2 == j]
        if len(tile_indices) > 0:
            his.append(len(tile_indices))
            rel_inds.append(j)
    # plt.hist(his,bins = 100)
    # plt.show()
    print(np.sum(np.array(his) == 1), 'tiles with single particles')
    print('tile with the most images has ', np.max(np.array(his)), 'particles')

    for j in rel_inds:
        tile_indices = latent_indices_half2[latent_points_half2 == j]
        current_data = torch.utils.data.Subset(dataset, tile_indices)
        current_data_loader = DataLoader(
            dataset=current_data,
            batch_size=8,
            num_workers=8,
            shuffle=False,
            pin_memory=False
        )
        z_tile = [torch.stack(2 * [gxy[j]], 0).to(device)]
        print(j, len(tile_indices))
        r = torch.zeros(2, 3)
        t = torch.zeros(2, 2)

        if len(tile_indices) > args.ignore:
            with torch.no_grad():
                # _,_,_,n_pos,deformation = decoder_half1(z_tile,r.to(device),cons_model.pos.to(device),cons_model.amp.to(device),cons_model.ampvar.to(device),t.to(device))
                tile_deformation = inv_half2(z_tile, torch.stack(2 * [smallgrid.to(device)], 0))

                disp = fwd_int.compute_field(tile_deformation[0])
                # disp0 = torch.nn.functional.conv3d(disp[...,0].unsqueeze(0).unsqueeze(0),ker,padding = 'same')
                # disp1 = torch.nn.functional.conv3d(disp[...,1].unsqueeze(0).unsqueeze(0),ker,padding = 'same')
                # disp2 = torch.nn.functional.conv3d(disp[...,2].unsqueeze(0).unsqueeze(0),ker,padding = 'same')

                disp0 = disp[0, ...]
                disp1 = disp[1, ...]
                disp2 = disp[2, ...]
                disp = torch.stack([disp0.squeeze(), disp1.squeeze(), disp2.squeeze()], 3)
                # detjac = get_grad_from_field(disp,box_size)
                dis_grid = disp[None, :, :, :]

                ind_ess_grid = torch.clip(torch.round((tile_deformation[0] + 0.5) * (box_size - 1)),
                                          min=0, max=box_size - 1).long()
                in_grid = torch.clip((ess_grid + 0.5) * (box_size - 1), min=0,
                                     max=box_size - 1).long()

                print(ind_ess_grid.shape, ind_ess_grid.dtype)
                # det = torch.nn.functional.conv3d(det.unsqueeze(0).unsqueeze(0),ker.to(device),padding = 'same')
                # det[det>5] = 1
                # det = det.unsqueeze(0).unsqueeze(0)

                # det[det == 0] = 1
                # det = torch.clip(det,min = 0.5, max = 10)
                # _,counts = torch.unique(ind_ess_grid,return_counts = True, dim = 1)
                print('Backprojected', bp_imgs.cpu().numpy(), 'particles from', len(dataset),
                      'and omitted backprojection of', om_imgs.cpu().numpy(), 'particles')
                # print('Duplicate indices:', torch.sum(counts-1).cpu().numpy())
                bp_imgs += len(tile_indices)

                del tile_deformation
        else:
            om_imgs += len(tile_indices)
        if len(tile_indices) > args.ignore:
            print('Starting backprojection of tile', j, 'from', gxy.shape[0], '::',
                  len(tile_indices), 'images')
            i = i + 1
            if i % 50 == 0:
                try:
                    VV = V[:, :, :] * sm_bin_mask
                except:
                    VV = V[:, :, :]
                VV = torch.fft.fftn(VV, dim=[-1, -2, -3])
                if args.filter == 'ctf':
                    VV2 = torch.real(
                        torch.fft.ifftn(VV / torch.maximum(CTF, lam_thres * torch.ones_like(CTF)),
                                        dim=[-1, -2, -3]))

                with mrcfile.new(
                    '/cephfs/schwab/backprojection/volumes14/reconstruction_half2_' + str(
                        i // 50).zfill(3) + '.mrc', overwrite=True) as mrc:
                    mrc.set_data((VV2 / torch.max(VV2)).float().cpu().numpy())

            with torch.no_grad():
                for batch_ndx, sample in enumerate(tqdm(current_data_loader)):
                    r, y, ctf, t_in = sample["rotation"], sample["image"], sample["ctf"], -sample[
                        'translation']
                    idx = sample['idx']
                    r, t = poses(idx)
                    batch_size = y.shape[0]
                    ctfs_l = torch.nn.functional.pad(ctf, (
                    box_size // 2, box_size // 2, box_size // 2, box_size // 2, 0, 0)).to(device)
                    ctfs = torch.fft.fftshift(ctf, dim=[-1, -2])
                    y_in = data_preprocessor.apply_square_mask(y)
                    y_in = data_preprocessor.apply_translation(y_in, -t[:, 0], -t[:, 1])
                    y_in = data_preprocessor.apply_circular_mask(y_in)
                    y_in, r, t, ctfs, ctf = y_in.to(device), r.to(device), t.to(device), ctfs.to(
                        device), ctf.to(device)
                    if args.ctf:
                        y = torch.fft.fft2(y_in, dim=[-1, -2])
                        y = y * ctfs
                        yr = torch.real(torch.fft.ifft2(y, dim=[-1, -2])).unsqueeze(1)
                    else:
                        yr = y_in.unsqueeze(1)
                    if args.filter == 'ctf':
                        ctfr2_l = torch.fft.fftshift(torch.real(
                            torch.fft.ifft2(torch.fft.fftshift(ctfs_l, dim=[-1, -2]),
                                            dim=[-1, -2], )).unsqueeze(1), dim=[-1, -2])
                        ctfr2_lc = torch.nn.functional.avg_pool2d(ctfr2_l, kernel_size=3, stride=2,
                                                                  padding=1)
                        CTFy = ctfr2_lc.expand(batch_size, box_size, box_size, box_size)
                        CTFy = torch.nn.functional.pad(CTFy, (0, 1, 0, 1, 0, 1, 0, 0))
                        CTFy = rotation(CTFy.unsqueeze(1), r).squeeze()

                        if len(CTFy.shape) < 4:
                            CTFy = CTFy.unsqueeze(0)

                        # CTFy = deform_volume(CTFy,ess_grid,out_grid,ind_ess_grid,box_size)
                        CTFy = CTFy[:, :-1, :-1, :-1]
                        CTF += (1 / len(dataset)) * torch.real(
                            torch.sum(torch.fft.fftn(CTFy.unsqueeze(1), dim=[-1, -2, -3]) ** 2,
                                      0).squeeze())

                    Vy = yr.expand(batch_size, box_size, box_size, box_size)
                    Vy = torch.nn.functional.pad(Vy, (0, 1, 0, 1, 0, 1, 0, 0))
                    Vy = rotation(Vy.unsqueeze(1), r).squeeze()
                    if len(Vy.shape) < 4:
                        Vy = Vy.unsqueeze(0)
                    # Vy = torch.sum(Vy.unsqueeze(1),0).squeeze()
                    if len(Vy.shape) < 4:
                        Vy = Vy.unsqueeze(0)
                    # Vy = deform_volume(Vy,ind_ess_grid,out_grid,ess_grid,box_size)

                    Vy = torch.sum(Vy, 0)
                    Vy = F.grid_sample(input=Vy.unsqueeze(0).unsqueeze(0), grid=dis_grid.to(device),
                                       mode='bilinear', align_corners=False)
                    V += (1 / len(dataset)) * Vy.squeeze()

    try:
        V = V * sm_bin_mask
    except:
        V = V
    V = torch.fft.fftn(V, dim=[-1, -2, -3])
    if args.filter == 'ctf':
        V = torch.real(torch.fft.ifftn(V / torch.maximum(CTF, lam_thres * torch.ones_like(CTF)),
                                       dim=[-1, -2, -3]))

    else:
        V = torch.real(torch.fft.ifftn(VV * filt3.to(device), dim=[-1, -2, -3]))

    with mrcfile.new(args.out_dir.replace('.mrc', '_half2.mrc'), overwrite=True) as mrc:
        mrc.set_data((V / torch.max(V)).float().detach().cpu().numpy())

    del V, CTF
