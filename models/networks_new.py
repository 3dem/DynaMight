#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 12:43:55 2021

@author: schwab
"""

import numpy as np
import torch
import torch.nn as nn
import torch.fft
from ..utils.utils_new import *


def positional_encoding_geom2(coords, enc_dim, DD):
    D2 = DD // 2
    freqs = torch.arange(enc_dim, dtype=torch.float).to(coords.device)
    freqs = D2 * (1. / D2) ** (freqs / (enc_dim - 1))  # 2/D*2pi to 2pi
    freqs = freqs.view(*[1] * len(coords.shape), -1)
    coords = coords.unsqueeze(-1)
    k = coords * freqs
    s = torch.sin(k)
    c = torch.cos(k)
    x = torch.cat([s, c], -1)
    return x.flatten(-2)


class gaussian_decoder(torch.nn.Module):
    def __init__(self, box_size, device, kernel, latent_dim, n_points, n_layers, n_neurons, block):
        super(gaussian_decoder, self).__init__()
        self.acth = nn.ReLU()
        self.latent_dim = latent_dim
        self.box_size = box_size
        self.n_points = n_points
        self.kernel_size = 2 * int(3 * kernel + 0.5) + 1  # truncate is 3, maybe must be higher
        self.k2 = make_kernel2(self.kernel_size, kernel)
        self.k3 = make_kernel(self.kernel_size, kernel)
        self.down = torch.nn.AvgPool2d(4, 4)
        self.ini = .5 * torch.ones(3)
        self.pos = torch.nn.Parameter(0.01 * (torch.rand(n_points, 3) - self.ini),
                                      requires_grad=True)
        # self.pos = torch.nn.Parameter(0.03*(torch.rand(n_points,3)-0.5),requires_grad=True)
        self.act = torch.nn.Tanh()
        self.amp = torch.nn.Parameter(0.1 * torch.ones(1), requires_grad=True)
        self.ampvar = torch.nn.Parameter(torch.rand(n_points), requires_grad=True)
        self.p2v = points2volume(self.box_size, device, self.k3)
        self.proj = point_projection(device, self.box_size)
        self.p2i = points2image(self.box_size, device, self.k2)
        self.res_block = block
        self.deform1 = self.make_layers(n_layers, n_neurons)
        self.lin1b = nn.Linear(3, 3, bias=False)
        self.lin1a = nn.Linear(n_neurons, 3, bias=False)
        self.lin0 = nn.Linear(3 + latent_dim, n_neurons, bias=False)
        self.device = device

    def forward(self, z, r, d):
        self.batch_size = z.shape[0]
        posi = torch.stack(self.batch_size * [self.pos], 0)
        res = torch.zeros_like(posi)

        if d > 0:
            conf_feat = torch.stack(self.n_points * [z], 0).squeeze().movedim(0, 1)
            res = self.lin0(torch.cat([posi, conf_feat], 2))
            res = self.deform1(res)
            res = self.act(self.lin1a(res))
            res = self.lin1b(res)
            pos = posi + res
        else:
            pos = posi

        Proj_pos = self.proj(pos, r)
        # if a>0:
        #     Proj_im = self.p2i(Proj_pos,torch.stack(self.batch_size*[self.ampvar]))
        # else:
        Proj_im = self.p2i(Proj_pos,
                           self.amp * torch.stack(self.batch_size * [torch.ones(self.n_points)]).to(
                               self.device))
        Proj = Proj_im
        return Proj, Proj_im, Proj_pos, pos, res

    def make_layers(self, n_layers, n_neurons):
        layers = []
        for j in range(n_layers):
            layers += [self.res_block(n_neurons, n_neurons)]

        return nn.Sequential(*layers)

    def volume(self, z, r):
        # for evaluation
        bs = z.shape[0]
        _, _, _, pos, _ = self.forward(z, r, 1)
        V = self.p2v(pos, self.amp * torch.ones(bs, self.n_points).to(self.device))
        return V

    def update_kernel(self, kernel):
        self.kernel_size = int(np.ceil(kernel * 6))
        if self.kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.k2 = make_kernel2(self.kernel_size, kernel)
        self.k3 = make_kernel(self.kernel_size, kernel)
        self.p2v = points2volume(self.box_size, self.device, self.k3)
        self.p2i = points2image(self.box_size, self.device, self.k2)

    def initialize_points(self, ini, th):
        ps = []
        n_box_size = ini.shape[0]
        while len(ps) < self.n_points:
            points = torch.rand(self.n_points, 3)
            indpoints = torch.round((n_box_size - 1) * points).long()
            point_inds = ini[indpoints[:, 0], indpoints[:, 1], indpoints[:, 2]] > th
            if len(ps) > 0:
                ps = torch.cat([ps, points[point_inds] - 0.5], 0)
            else:
                ps = points[point_inds] - 0.5
        self.pos = torch.nn.Parameter(ps[:self.n_points].to(self.device), requires_grad=True)

    def state(self, state):
        if state == 'initial':
            self.pos = torch.nn.Parameter(self.pos, requires_grad=True)
            # dec_optimizer.add_param_group({'params': decoder.pos})
            self.pos.requires_grad = True
            self.i2F.A.requires_grad = True
            self.i2F.B.requires_grad = True

        if state == 'refine_consensus':
            points = self.pos
            self.pos = torch.nn.Parameter(points, requires_grad=True)
            # dec_optimizer.add_param_group({'params': decoder.pos})
            self.requires_grad = False
            self.amp.requires_grad = True
            self.pos.requires_grad = True
            self.i2F.A.requires_grad = True
            self.i2F.B.requires_grad = True
        if state == 'refine_displacement':
            self.requires_grad = True
            self.i2F.A.requires_grad = False
            self.i2F.B.requires_grad = False
            self.pos.requires_grad = False

    def add_points(self):
        # useless
        perm = torch.randperm(self.pos.shape)
        np = (self.pos - self.pos[perm]) / 2
        self.pos = torch.nn.Parameter(torch.cat([self.pos, np], 0), requires_grad=True)
        self.ampvar = torch.nn.Parameter(torch.cat([self.ampvar, self.ampvar]), requires_grad=True)
        self.n_points = self.n_points * 2


class multi_gaussian_decoder(torch.nn.Module):
    def __init__(self, box_size, device, kernel, latent_dim, n_points, n_classes, n_layers,
                 n_neurons, block):
        super(multi_gaussian_decoder, self).__init__()
        self.acth = nn.ReLU()
        self.latent_dim = latent_dim
        self.box_size = box_size
        self.n_points = n_points
        self.kernel_size = 2 * int(3 * kernel + 0.5) + 1  # truncate is 3, maybe must be higher
        self.k2 = make_kernel2(self.kernel_size, kernel)
        self.k3 = make_kernel(self.kernel_size, kernel)
        self.down = torch.nn.AvgPool2d(4, 4)
        self.ini = .5 * torch.ones(3)
        self.pos = torch.nn.Parameter(0.015 * (torch.rand(n_points, 3) - self.ini),
                                      requires_grad=True)
        # self.pos = torch.nn.Parameter(0.03*(torch.rand(n_points,3)-0.5),requires_grad=True)
        self.act = torch.nn.Tanh()
        self.amp = torch.nn.Parameter(0.05 * torch.ones(1), requires_grad=True)
        self.ampvar = torch.nn.Parameter(torch.rand(n_classes, n_points), requires_grad=True)
        self.p2v = points2mult_volume(self.box_size, device, n_classes)
        self.proj = point_projection(device, self.box_size)
        self.p2i = points2mult_image(self.box_size, device, n_classes)
        # self.i2F = ims2Fim(self.box_size,device,n_classes)
        self.i2F = ims2F_form(self.box_size, device, n_classes)
        self.res_block = block
        self.deform1 = self.make_layers(n_layers, n_neurons)
        self.lin1b = nn.Linear(3, 3, bias=False)
        self.lin1a = nn.Linear(n_neurons, 3, bias=False)
        self.lin0 = nn.Linear(3 + latent_dim, n_neurons, bias=False)
        self.device = device

    def forward(self, z, r, d):
        self.batch_size = z.shape[0]
        posi = torch.stack(self.batch_size * [self.pos], 0)
        res = torch.zeros_like(posi)

        if d > 0:
            conf_feat = torch.stack(self.n_points * [z], 0).squeeze().movedim(0, 1)
            res = self.lin0(torch.cat([posi, conf_feat], 2))
            res = self.deform1(res)
            res = self.act(self.lin1a(res))
            res = self.lin1b(res)
            pos = posi + res
        else:
            pos = posi

        Proj_pos = self.proj(pos, r)
        # if a>0:
        #     Proj_im = self.p2i(Proj_pos,torch.stack(self.batch_size*[self.ampvar]))
        # else:

        Proj_im = self.p2i(Proj_pos, torch.stack(
            self.batch_size * [self.amp * torch.nn.functional.softmax(self.ampvar, dim=0)],
            dim=0).to(self.device))
        Proj = self.i2F(Proj_im)
        return Proj, Proj_im, Proj_pos, pos, res

    def make_layers(self, n_layers, n_neurons):
        layers = []
        for j in range(n_layers):
            layers += [self.res_block(n_neurons, n_neurons)]

        return nn.Sequential(*layers)

    def volume(self, z, r, defo=1):
        # for evaluation NEEDS TO BE REWRITTEN FOR MULTI-GAUSSIAN
        bs = z.shape[0]
        _, _, _, pos, _ = self.forward(z, r, defo)
        V = self.p2v(pos, torch.stack(bs * [torch.nn.functional.softmax(self.ampvar, dim=0)],
                                      0) * self.amp.to(self.device))
        # V = self.p2v(pos,torch.stack(bs*[self.ampvar],0)*self.amp.to(self.device))
        V = torch.fft.fftn(V, dim=[-3, -2, -1])
        R, M = radial_index_mask3(self.box_size)
        R = torch.stack(self.i2F.n_classes * [R.to(self.device)], 0)
        FF = torch.exp(-self.i2F.B[:, None, None, None] ** 2 * R) * self.i2F.A[:, None, None,
                                                                    None] ** 2
        bs = V.shape[0]
        Filts = torch.stack(bs * [FF], 0)
        Filts = torch.fft.ifftshift(Filts, dim=[-3, -2, -1])
        V = torch.real(torch.fft.ifftn(torch.sum(Filts * V, 1), dim=[-3, -2, -1]))
        return V

    def initialize_points(self, ini, th):
        ps = []
        while len(ps) < self.n_points:
            points = torch.rand(self.n_points, 3)
            indpoints = torch.round((self.box_size - 1) * points).long()
            point_inds = ini[indpoints[:, 0], indpoints[:, 1], indpoints[:, 2]] > th
            if len(ps) > 0:
                ps = torch.cat([ps, points[point_inds] - 0.5], 0)
            else:
                ps = points[point_inds] - 0.5
        self.pos = torch.nn.Parameter(ps[:self.n_points].to(self.device), requires_grad=True)

    def add_points(self, th, ang_pix):
        # useless
        fatclass = torch.argmin(self.i2F.B)
        probs = torch.nn.functional.softmax(self.ampvar, dim=0)
        fatclass_points = self.pos[probs[fatclass, :] > th]
        eps = 0.5 / (self.box_size * ang_pix) * (torch.rand_like(fatclass_points) - 0.5)
        np = fatclass_points + eps
        self.pos = torch.nn.Parameter(torch.cat([self.pos, np], 0), requires_grad=True)
        self.ampvar = torch.nn.Parameter(
            torch.cat([self.ampvar, torch.rand_like(self.ampvar[:, probs[fatclass, :] > th])], 1),
            requires_grad=True)
        self.n_points = self.pos.shape[0]

    def state(self, state):
        if state == 'initial':
            # self.pos = torch.nn.Parameter(self.pos,requires_grad = True)
            # dec_optimizer.add_param_group({'params': decoder.pos})
            self.pos.requires_grad = True
            self.i2F.A.requires_grad = True
            self.i2F.B.requires_grad = True

        if state == 'refine_consensus':
            # points = self.pos
            # self.pos = torch.nn.Parameter(points,requires_grad = True)
            # dec_optimizer.add_param_group({'params': decoder.pos})
            self.requires_grad = False
            self.amp.requires_grad = True
            self.pos.requires_grad = True
            self.i2F.A.requires_grad = True
            self.i2F.B.requires_grad = True
        if state == 'refine_displacement':
            self.requires_grad = True
            self.i2F.A.requires_grad = False
            self.i2F.B.requires_grad = False
            self.pos.requires_grad = False


class gaussian_decoder_het(
    torch.nn.Module):  # Decoder with the possibility to add and remove density
    def __init__(self, box_size, device, kernel, latent_dim, n_points, n_layers, n_neurons, block):
        super(gaussian_decoder_het, self).__init__()
        self.acth = nn.ReLU()
        self.latent_dim = latent_dim
        self.box_size = box_size
        self.n_points = n_points
        self.kernel_size = int(np.ceil(kernel * 6))
        if self.kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.k2 = make_kernel2(self.kernel_size, kernel)
        self.k3 = make_kernel(self.kernel_size, kernel)
        self.down = torch.nn.AvgPool2d(4, 4)
        self.ini = .5 * torch.ones(3)
        self.ini[2] = 1
        self.pos = torch.nn.Parameter(0.1 * (torch.rand(n_points, 3) - self.ini),
                                      requires_grad=True)
        # self.pos = torch.nn.Parameter(0.03*(torch.rand(n_points,3)-0.5),requires_grad=True)
        self.act = torch.nn.Tanh()
        self.amp = torch.nn.Parameter(0.1 * torch.ones(1), requires_grad=True)
        self.ampvar = torch.nn.Parameter(torch.rand(n_points), requires_grad=True)
        self.p2v = points2volume(self.box_size, device, self.k3)
        self.proj = point_projection(device, self.box_size)
        self.p2i = points2image(self.box_size, device, self.k2)
        self.res_block = block
        self.deform1 = self.make_layers(n_layers, n_neurons)
        self.lin1b = nn.Linear(3, 3, bias=False)
        self.lin1a = nn.Linear(n_neurons, 3, bias=False)
        self.lin0 = nn.Linear(3 + latent_dim - 1, n_neurons, bias=False)
        self.device = device
        self.linamp = nn.Linear(1, n_points)
        self.actamp = nn.Sigmoid()

    def forward(self, z, r, d):
        self.batch_size = z.shape[0]
        posi = torch.stack(self.batch_size * [self.pos], 0)
        res = torch.zeros_like(posi)

        if d > 0:
            conf_feat = torch.stack(self.n_points * [z[:, :-1]], 0).squeeze().movedim(0, 1)
            res = self.lin0(torch.cat([posi, conf_feat], 2))
            res = self.deform1(res)
            res = self.act(self.lin1a(res))
            res = self.lin1b(res)
            pos = posi + res
        else:
            pos = posi
        self.amp_corr = self.actamp(self.linamp(z[:, -1:]))
        Proj_pos = self.proj(pos, r)
        Proj_im = self.p2i(Proj_pos, self.amp_corr * self.amp * torch.stack(
            self.batch_size * [torch.ones(self.n_points)]).to(self.device))
        Proj = Proj_im
        return Proj, Proj_im, Proj_pos, pos, res

    def make_layers(self, n_layers, n_neurons):
        layers = []
        for j in range(n_layers):
            layers += [self.res_block(n_neurons, n_neurons)]

        return nn.Sequential(*layers)

    def volume(self, z, r):
        # for evaluation
        bs = z.shape[0]
        _, _, _, pos, _ = self.forward(z, r, 1)
        V = self.p2v(pos, self.amp * self.amp_corr)
        return V

    def update_kernel(self, kernel):
        self.kernel_size = int(np.ceil(kernel * 6))
        if self.kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.k2 = make_kernel2(self.kernel_size, kernel)
        self.k3 = make_kernel(self.kernel_size, kernel)
        self.p2v = points2volume(self.box_size, self.device, self.k3)
        self.p2i = points2image(self.box_size, self.device, self.k2)

    def initialize_points(self, ini, th):
        ps = []
        while len(ps) < self.n_points:
            points = 0.6 * torch.rand(self.n_points, 3) + 0.2
            indpoints = torch.round((self.box_size - 1) * points).long()
            point_inds = ini[indpoints[:, 0], indpoints[:, 1], indpoints[:, 2]] > th
            if len(ps) > 0:
                ps = torch.cat([ps, points[point_inds] - 0.5], 0)
            else:
                ps = points[point_inds] - 0.5
        self.pos = torch.nn.Parameter(ps[:self.n_points], requires_grad=True)

    def add_points(self):
        # useless
        perm = torch.randperm(self.pos.shape)
        np = (self.pos - self.pos[perm]) / 2
        self.pos = torch.nn.Parameter(torch.cat([self.pos, np], 0), requires_grad=True)
        self.ampvar = torch.nn.Parameter(torch.cat([self.ampvar, self.ampvar]), requires_grad=True)
        self.n_points = self.n_points * 2


class reshet_encoder(torch.nn.Module):

    def __init__(self, box_size, latent_dim):
        super().__init__()  # call of torch.nn.Module
        # self.down = nn.AvgPool2d(2,stride = 2)
        self.lin1 = nn.Linear(in_features=box_size ** 2, out_features=256)
        self.lin1a = nn.Linear(in_features=256, out_features=256)
        self.lin1b = nn.Linear(in_features=256, out_features=128)
        self.lin1c = nn.Linear(in_features=128, out_features=128)
        self.lin2 = nn.Linear(in_features=128, out_features=latent_dim)
        self.lin3 = nn.Linear(in_features=128, out_features=latent_dim)
        self.flat = nn.Flatten()
        self.act1 = nn.Tanh()
        self.actr = nn.ReLU()
        # self.enc_optimizer = []

    def forward(self, x):
        # inp = self.down(x)
        inp = self.flat(x)
        x2 = self.lin1(inp)
        x2 = x2 + self.actr(self.lin1a(x2))
        x2 = self.actr(x2)
        x2 = self.act1(self.lin1b(x2))
        x2 = x2 + self.act1(self.lin1c(x2))
        x3 = self.lin3(x2)
        x2 = self.lin2(x2)
        return x2, x3


class wavelet_encoder(torch.nn.Module):

    def __init__(self, box_size, latent_dim, level=3):
        super().__init__()  # call of torch.nn.Module
        self.wave_dec = DTCWTForward(J=level, biort='near_sym_b', qshift='qshift_b')
        # self.down = nn.AvgPool2d(2,stride = 2)
        self.lin1 = nn.Linear(in_features=(box_size // 4) ** 2, out_features=256)
        self.lin1a = nn.Linear(in_features=256, out_features=64)
        self.lin2 = nn.Linear(in_features=2 * 6 * (box_size // 2) ** 2, out_features=256)
        self.lin2a = nn.Linear(in_features=256, out_features=64)
        self.lin3 = nn.Linear(in_features=2 * 6 * (box_size // 4) ** 2, out_features=256)
        self.lin3a = nn.Linear(in_features=256, out_features=64)
        self.lin4 = nn.Linear(in_features=2 * 6 * (box_size // 8) ** 2, out_features=256)
        self.lin4a = nn.Linear(in_features=256, out_features=64)
        self.lin5 = nn.Linear(in_features=256, out_features=128)
        self.linmu = nn.Linear(in_features=128, out_features=latent_dim)
        self.linsig = nn.Linear(in_features=128, out_features=latent_dim)
        self.flat = nn.Flatten()
        self.act1 = nn.Tanh()
        self.actr = nn.ReLU()
        # self.enc_optimizer = []

    def forward(self, x, ctf):
        inp = torch.fft.fft2(x, dim=[-1, -2])
        inp = x * ctf
        inp = torch.real(torch.fft.ifft2(inp, dim=[-1, -2])).float()
        wdec_1, wdec_h = self.wave_dec(inp.unsqueeze(1))
        inp_1 = self.flat(wdec_1)
        inp_1 = self.actr(self.lin1(inp_1))
        inp_1 = self.act1(self.lin1a(inp_1))
        inp_2 = wdec_h[0].squeeze()
        inp_2 = self.flat(inp_2)
        inp_2 = self.actr(self.lin2(inp_2))
        inp_2 = self.act1(self.lin2a(inp_2))
        inp_3 = wdec_h[1].squeeze()
        inp_3 = self.flat(inp_3)
        inp_3 = self.actr(self.lin3(inp_3))
        inp_3 = self.act1(self.lin3a(inp_3))
        inp_4 = wdec_h[2].squeeze()
        inp_4 = self.flat(inp_4)
        inp_4 = self.actr(self.lin4(inp_4))
        inp_4 = self.act1(self.lin4a(inp_4))
        inp_end = torch.cat([inp_1, inp_2, inp_3, inp_4], 0)
        inp_end = self.act1(self.lin5(inp_end))
        mu = self.linmu(inp_end)
        sig = self.linsig(inp_end)

        return mu, sig


class res_ctf_encoder(torch.nn.Module):

    def __init__(self, box_size, latent_dim):
        super().__init__()  # call of torch.nn.Module
        # self.down = nn.AvgPool2d(2,stride = 2)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding='same')
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding='same')
        self.lin1 = nn.Linear(in_features=box_size ** 2, out_features=256)
        self.lin1a = nn.Linear(in_features=256, out_features=256)
        self.lin1b = nn.Linear(in_features=256, out_features=128)
        self.lin1c = nn.Linear(in_features=128, out_features=128)
        self.lin2 = nn.Linear(in_features=128, out_features=latent_dim)
        self.lin3 = nn.Linear(in_features=128, out_features=latent_dim)
        self.flat = nn.Flatten()
        self.act1 = nn.Tanh()
        self.actr = nn.ReLU()
        # self.enc_optimizer = []

    def forward(self, x, ctf):
        # inp = self.down(x)
        inp = torch.stack([x, ctf], 1)
        xx = self.actr(self.conv1(inp))
        xx = xx + self.actr(self.conv2(xx))
        xx = xx + self.actr(self.conv3(xx))
        xx = xx + self.actr(self.conv4(xx))
        xx = self.conv5(xx)
        x_in = self.flat(xx)
        x2 = self.lin1(x_in)
        x2 = x2 + self.actr(self.lin1a(x2))
        x2 = self.actr(x2)
        x2 = self.act1(self.lin1b(x2))
        x2 = x2 + self.act1(self.lin1c(x2))
        x3 = self.lin3(x2)
        x2 = self.lin2(x2)
        return x2, x3, xx


class het_encoder(torch.nn.Module):

    def __init__(self, box_size, latent_dim, down_sample):
        super().__init__()  # call of torch.nn.Module
        self.down_sample = down_sample
        self.down = nn.AvgPool2d(2, stride=2)
        self.lin1 = nn.Linear(in_features=int((box_size / (2 ** down_sample)) ** 2),
                              out_features=256)
        # self.lin1a = nn.Linear(in_features = 256, out_features = 256)
        # self.lin1aa = nn.Linear(in_features = 256, out_features = 256)
        self.lin1b = nn.Linear(in_features=256, out_features=128)
        self.lin2 = nn.Linear(in_features=128, out_features=latent_dim)
        self.lin3 = nn.Linear(in_features=128, out_features=latent_dim)
        self.flat = nn.Flatten()
        self.act1 = nn.Tanh()
        self.actr = nn.ReLU()
        self.latent_dim = latent_dim
        # self.enc_optimizer = []

    def forward(self, x, ctf):
        # inp = self.down(x)
        # inp = torch.fft.fft2(x,dim=[-1,-2],norm = 'ortho')
        # inp = x*ctf
        # inp = torch.real(torch.fft.ifft2(inp,dim=[-1,-2],norm = 'ortho')).float()
        inp = x
        for i in range(self.down_sample):
            inp = self.down(inp)
        inp = self.flat(inp)
        x2 = self.lin1(inp)
        x2 = self.actr(x2)
        # x2 = self.actr(self.lin1a(x2))
        # x2 = self.actr(self.lin1aa(x2))
        x2 = self.act1(self.lin1b(x2))
        x3 = self.lin3(x2)
        x2 = self.lin2(x2)
        return x2, x3  #


class complete_encoder(torch.nn.Module):

    def __init__(self, box_size, latent_dim, n_gauss):
        super().__init__()  # call of torch.nn.Module
        self.down = nn.AvgPool2d(2, stride=2)
        self.lin1 = nn.Linear(in_features=box_size ** 2, out_features=256)
        self.lin1b = nn.Linear(in_features=256, out_features=128)
        self.lin2 = nn.Linear(in_features=128, out_features=latent_dim)
        self.lin3 = nn.Linear(in_features=128, out_features=latent_dim)
        self.flat = nn.Flatten()
        self.act1 = nn.Tanh()
        self.actr = nn.ReLU()
        self.latent_dim = latent_dim
        self.box_size = box_size
        self.N_points = n_gauss
        self.lin0tot = nn.Linear(in_features=3 * n_gauss, out_features=(box_size // 2) ** 2)
        self.lin1tot = nn.Linear(in_features=2 * (box_size // 2) ** 2, out_features=256)
        self.lin2tot = nn.Linear(in_features=256, out_features=128)
        self.lin3tot = nn.Linear(in_features=128, out_features=128)
        self.lin4tot = nn.Linear(in_features=128, out_features=128)
        self.lin5tot = nn.Linear(in_features=128, out_features=3)
        self.lin6tot = nn.Linear(in_features=128, out_features=3)
        # self.enc_optimizer = []

    def forward(self, x, ctf, pos):
        # inp = self.down(x)
        # inp = torch.fft.fft2(x,dim=[-1,-2])
        # inp = x*ctf
        # inp = torch.real(torch.fft.ifft2(inp,dim=[-1,-2])).float()
        inp = x
        inp1 = self.down(inp)
        inp = self.flat(x)
        x2 = self.lin1(inp)
        x2 = self.actr(x2)
        x2 = self.act1(self.lin1b(x2))
        x3 = self.lin3(x2)
        x2 = self.lin2(x2)
        inp1 = self.flat(inp1)
        inp2 = torch.stack(inp.shape[0] * [torch.flatten(pos)])
        inp2 = self.actr(self.lin0tot(inp2))
        inp_tot = torch.cat([inp1, inp2], 1)
        r = self.actr(self.lin1tot(inp_tot))
        r = self.actr(self.lin2tot(r))
        r = self.actr(self.lin3tot(r))
        r = self.actr(self.lin4tot(r))
        u = self.lin5tot(r)
        v = self.lin6tot(r)
        u = torch.nn.functional.normalize(u, dim=1)
        v = torch.nn.functional.normalize(v, dim=1)
        w = torch.nn.functional.normalize(v - torch.stack(3 * [torch.sum(u * v, 1)], 1) * u)
        w2 = torch.cross(u, w, dim=1)
        R = torch.stack([u, w, w2], 2)

        return x2, x3, R


class mixed_encoder(torch.nn.Module):

    def __init__(self, box_size, latent_dim):
        super().__init__()  # call of torch.nn.Module
        self.down = nn.AvgPool2d(2, stride=2)
        self.conv1a = nn.Conv2d(1, 8, 3, padding='same')
        self.conv1b = nn.Conv2d(8, 8, 3, padding='same')
        self.conv2a = nn.Conv2d(8, 8, 3, padding='same')
        self.conv2b = nn.Conv2d(8, 8, 3, padding='same')
        self.lin1 = nn.Linear(in_features=(box_size // 4) ** 2 * 8, out_features=256)
        self.lin1b = nn.Linear(in_features=256, out_features=128)
        self.lin2 = nn.Linear(in_features=128, out_features=latent_dim)
        self.lin3 = nn.Linear(in_features=128, out_features=latent_dim)
        self.flat = nn.Flatten()
        self.act1 = nn.Tanh()
        self.actr = nn.ReLU()
        # self.enc_optimizer = []

    def forward(self, x, ctf):
        inp = torch.fft.fft2(x, dim=[-1, -2])
        inp = x * ctf
        inp = torch.real(torch.fft.ifft2(inp, dim=[-1, -2])).float()
        inp = self.actr(self.conv1a(inp))
        inp = self.actr(self.conv1b(inp))
        inp = self.down(inp)
        inp = self.actr(self.conv2a(inp))
        inp = self.actr(self.conv2b(inp))
        inp = self.down(inp)
        inp = self.flat(inp)
        x2 = self.lin1(inp)
        x2 = self.actr(x2)
        x2 = self.act1(self.lin1b(x2))
        x3 = self.lin3(x2)
        x2 = self.lin2(x2)
        return x2, x3  #


class gaussian_decoder_het2(
    torch.nn.Module):  # coordinate based density value (needs extra latent dimension for density)
    def __init__(self, box_size, device, kernel, latent_dim, n_points, n_layers, n_neurons, block,
                 diff_conf=False):
        super(gaussian_decoder_het2, self).__init__()
        self.diff_conf = diff_conf
        self.acth = nn.ReLU()
        self.latent_dim = latent_dim
        self.box_size = box_size
        self.n_points = n_points
        self.kernel_size = int(np.ceil(kernel * 6))
        if self.kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.k2 = make_kernel2(self.kernel_size, kernel)
        self.k3 = make_kernel(self.kernel_size, kernel)
        self.down = torch.nn.AvgPool2d(4, 4)
        self.ini = .5 * torch.ones(3)
        # self.ini[2] = 1
        self.pos = torch.nn.Parameter(0.1 * (torch.rand(n_points, 3) - self.ini),
                                      requires_grad=True)
        # self.pos = torch.nn.Parameter(0.03*(torch.rand(n_points,3)-0.5),requires_grad=True)
        self.act = torch.nn.Tanh()
        self.amp = torch.nn.Parameter(0.1 * torch.ones(1), requires_grad=True)
        self.p2v = points2volume(self.box_size, device, self.k3)
        self.proj = point_projection(device, self.box_size)
        self.p2i = points2image(self.box_size, device, self.k2)
        self.res_block = block
        self.deform1 = self.make_layers(n_layers, n_neurons)
        self.lin1b = nn.Linear(3, 3, bias=False)
        self.lin1a = nn.Linear(n_neurons, 3, bias=False)
        self.lin0 = nn.Linear(3 + latent_dim, n_neurons, bias=False)
        self.device = device
        if diff_conf == True:
            self.lin0 = nn.Linear(3 + latent_dim - 1, n_neurons, bias=False)
            self.linamp1 = nn.Linear(3 + 1, n_neurons, bias=False)
            self.ampblock = self.make_layers(n_layers, n_neurons)
            self.linamp2 = nn.Linear(n_neurons, 1)
            self.actamp = nn.Sigmoid()

    def forward(self, z, r, d):
        self.batch_size = z.shape[0]
        posi = torch.stack(self.batch_size * [self.pos], 0)
        res = torch.zeros_like(posi)
        if self.diff_conf == True:
            conf_feat_a = torch.stack(self.n_points * [z[:, -1:]], 0).movedim(0, 1)
            a = self.linamp1(torch.cat([posi, conf_feat_a], 2))
            a = self.ampblock(a)
            amp_corr = self.actamp(self.linamp2(a))

            if d > 0:
                conf_feat = torch.stack(self.n_points * [z[:, :-1]], 0).squeeze().movedim(0, 1)
                res = self.lin0(torch.cat([posi, conf_feat], 2))
                res = self.deform1(res)
                res = self.act(self.lin1a(res))
                res = self.lin1b(res)
                pos = posi + res
            else:
                pos = posi
            Proj_pos = self.proj(pos, r)
            Proj_im = self.p2i(Proj_pos, self.amp * amp_corr.squeeze().to(self.device))
        else:
            if d > 0:
                conf_feat = torch.stack(self.n_points * [z], 0).squeeze().movedim(0, 1)
                res = self.lin0(torch.cat([posi, conf_feat], 2))
                res = self.deform1(res)
                res = self.act(self.lin1a(res))
                res = self.lin1b(res)
                pos = posi + res
            else:
                pos = posi

            Proj_pos = self.proj(pos, r)
            Proj_im = self.p2i(Proj_pos, self.amp * torch.stack(
                self.batch_size * [torch.ones(self.n_points)]).to(self.device))
        Proj = Proj_im
        return Proj, pos, res, amp_corr

    def make_layers(self, n_layers, n_neurons):
        layers = []
        for j in range(n_layers):
            layers += [self.res_block(n_neurons, n_neurons)]

        return nn.Sequential(*layers)

    def volume(self, z, r):
        # for evaluation
        bs = z.shape[0]
        _, pos, _, amp_corr = self.forward(z, r, 1)
        if self.diff_conf == True:
            V = self.p2v(pos, self.amp * amp_corr.squeeze())
        else:
            V = self.p2v(pos, torch.stack(bs * [self.ampvar]))
        return V

    def update_kernel(self, kernel):
        self.kernel_size = int(np.ceil(kernel * 6))
        if self.kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.k2 = make_kernel2(self.kernel_size, kernel)
        self.k3 = make_kernel(self.kernel_size, kernel)
        self.p2v = points2volume(self.box_size, self.device, self.k3)
        self.p2i = points2image(self.box_size, self.device, self.k2)

    def add_points(self):
        # useless
        perm = torch.randperm(self.pos.shape)
        np = (self.pos - self.pos[perm]) / 2
        self.pos = torch.nn.Parameter(torch.cat([self.pos, np], 0), requires_grad=True)
        self.ampvar = torch.nn.Parameter(torch.cat([self.ampvar, self.ampvar]), requires_grad=True)
        self.n_points = self.n_points * 2


class implicit_pointnet(torch.nn.Module):
    def __init__(self, box_size, device, kernel_high, latent_dim, n_points):
        super(implicit_pointnet, self).__init__()
        self.acth = nn.ReLU()
        self.n_points = n_points
        # self.linw = nn.Linear(in_features = 3, out_features = box_size**2 )
        self.box_size = box_size
        self.project = relion_projection(box_size, device)
        self.pos = torch.nn.Parameter(0.35 * (torch.rand(n_points, 3) - 0.5), requires_grad=True)
        self.W = torch.nn.Parameter(0.1 * torch.rand(1, 96, 96), requires_grad=True)
        self.x = torch.nn.Parameter(torch.ones(n_points), requires_grad=False)
        self.amp = torch.nn.Parameter(-torch.ones(1), requires_grad=True)
        self.p2v_high = points2volume(box_size, device, kernel_high)
        self.kernel_high = kernel_high
        self.deform1 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 10, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 30, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 150, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(150, 600, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(600, int((self.n_points * 3) / 8), bias=False),
        )
        self.up1 = torch.nn.Upsample(scale_factor=2)
        self.conv0 = torch.nn.Conv1d(3, 6, 3, padding=1, bias=False)
        self.conv1 = torch.nn.Conv1d(6, 3, 3, padding=1, bias=False)
        self.conv2 = torch.nn.Conv1d(3, 3, 3, padding=1, bias=False)

    def forward(self, z, r, d):
        self.batch_size = z.shape[0]
        # gr = self.eg(self.cons_vol)
        # W = self.linw(r)
        W = torch.stack(self.batch_size * [self.W], 0)
        # W = self.linw(torch.cat([z,r],1))
        # W = W.reshape(self.batch_size,self.box_size,self.box_size)
        posi = torch.stack(self.batch_size * [self.pos], 0)
        res = posi

        if d > 0:
            res = self.deform1(z)
            res = res.reshape(self.batch_size, 3, int(self.n_points / 8))
            res = self.up1(res)
            res = self.conv0(res)
            res = self.up1(res)
            res = self.conv1(res)
            res = self.acth(res)
            res = self.up1(res)
            res = self.conv2(res)
            res = res.movedim(1, 2)
            pos = posi + res
        else:
            pos = posi

        V_high = self.p2v_high(pos,
                               self.amp * torch.stack(self.batch_size * [torch.ones_like(self.x)],
                                                      0))
        Proj_high = self.project(V_high.unsqueeze(1), r)

        return Proj_high, V_high, pos, W, res


class pose_model(torch.nn.Module):
    def __init__(self, box_size, device, orientations, translations):
        super(pose_model, self).__init__()
        self.orientations = torch.nn.Parameter(orientations, requires_grad=True)
        self.translations = torch.nn.Parameter(translations, requires_grad=True)
        self.device = device
        self.box_size = box_size

    def forward(self, ind):
        r = self.orientations[ind]
        t = self.translations[ind]
        return r, t


class simple_scale_model(torch.nn.Module):
    def __init(self, box_size, device, scales):
        super(simple_scale_model, self).__init__()
        self.scales = torch.nn.Parameter(scales, requires_grad=True)
        self.device = device
        self.box_size = box_size

    def forward(self, ind):
        s = self.scale[ind]
        return s


class consensus_model(torch.nn.Module):
    def __init__(self, box_size, device, n_points, n_classes, oversampling=1):
        super(consensus_model, self).__init__()
        self.box_size = box_size
        self.n_points = n_points
        self.ini = .5 * torch.ones(3)
        self.pos = torch.nn.Parameter(0.035 * (torch.rand(n_points, 3) - self.ini),
                                      requires_grad=True)
        self.amp = torch.nn.Parameter(30 * torch.ones(n_classes, n_points), requires_grad=True)
        # self.ampvar = torch.nn.Parameter(torch.rand(n_classes,n_points),requires_grad=True)
        self.ampvar = torch.nn.Parameter(0.5 * torch.randn(n_classes, n_points), requires_grad=True)
        # self.ampvar = torch.nn.Parameter(torch.rand(n_classes,n_points),requires_grad=True)
        self.proj = point_projection(self.box_size)
        self.p2i = points2mult_image(self.box_size, n_classes, oversampling)
        self.i2F = ims2F_form(self.box_size, device, n_classes, oversampling)
        self.W = torch.nn.Parameter(torch.ones(box_size // 2), requires_grad=True)
        self.device = device
        if box_size > 360:
            self.vol_box = box_size // 2
        else:
            self.vol_box = box_size
        self.p2v = points2mult_volume(self.vol_box, n_classes)

    def forward(self, r, shift):
        self.batch_size = r.shape[0]
        posi = torch.stack(self.batch_size * [self.pos], 0)
        res = torch.zeros_like(posi)
        Proj_pos = self.proj(posi, r)  # -torch.stack(self.n_points*[shift],2)/(2*self.box_size)
        # if a>0:+
        #     Proj_im = self.p2i(Proj_pos,torch.stack(self.batch_size*[self.ampvar]))
        # else:

        Proj_im = self.p2i(Proj_pos, torch.stack(self.batch_size * [
            torch.clip(self.amp, min=1) * torch.nn.functional.softmax(self.ampvar, dim=0)],
                                                 dim=0).to(self.device))
        Proj = self.i2F(Proj_im)
        Proj = fourier_shift_2d(Proj.squeeze(), shift[:, 0], shift[:, 1])
        return Proj, Proj_im, Proj_pos, posi

    def volume(self, r, shift):
        # for evaluation NEEDS TO BE REWRITTEN FOR MULTI-GAUSSIAN
        bs = r.shape[0]
        _, _, _, pos = self.forward(r, shift)
        V = self.p2v(pos, torch.stack(bs * [torch.nn.functional.softmax(self.ampvar, dim=0)],
                                      0) * torch.clip(self.amp, min=1).to(self.device))
        # V = self.p2v(pos,torch.stack(bs*[self.ampvar],0)*self.amp.to(self.device))
        V = torch.fft.fftn(V, dim=[-3, -2, -1], norm='ortho')
        R, M = radial_index_mask3(self.vol_box)
        R = torch.stack(self.i2F.n_classes * [R.to(self.device)], 0)
        A = self.i2F.A
        B = self.i2F.B
        FF = torch.exp(-B[:, None, None, None] ** 2 * R) * A[:, None, None, None] ** 2
        Filts = torch.stack(bs * [FF], 0)
        Filts = torch.fft.ifftshift(Filts, dim=[-3, -2, -1])
        V = torch.real(torch.fft.ifftn(torch.sum(Filts * V, 1), dim=[-3, -2, -1], norm='ortho'))
        return V

    # def volume2(self):
    #     r = torch.zeros(2,3)
    #     shift = torch.zeros(2,2)
    #     _,_,_,pos = self.forward(r,shift)
    #     V = p2volume(pos,torch.stack(2*[torch.ones_like(self.ampvar)],0)*self.amp

    def initialize_points(self, ini, th):
        ps = []
        n_box_size = ini.shape[0]
        while len(ps) < self.n_points:
            points = torch.rand(self.n_points, 3)
            indpoints = torch.round((n_box_size - 1) * points).long()
            point_inds = ini[indpoints[:, 0], indpoints[:, 1], indpoints[:, 2]] > th
            if len(ps) > 0:
                ps = torch.cat([ps, points[point_inds] - 0.5], 0)
            else:
                ps = points[point_inds] - 0.5
        self.pos = torch.nn.Parameter(ps[:self.n_points].to(self.device), requires_grad=True)

    def add_points(self, th, ang_pix):
        # useless
        fatclass = torch.argmin(self.i2F.B)
        probs = torch.nn.functional.softmax(self.ampvar, dim=0)
        fatclass_points = self.pos[probs[fatclass, :] > th]
        eps = 0.5 / (self.box_size * ang_pix) * (torch.rand_like(fatclass_points) - 0.5)
        np = fatclass_points + eps
        self.pos = torch.nn.Parameter(torch.cat([self.pos, np], 0), requires_grad=True)
        self.ampvar = torch.nn.Parameter(
            torch.cat([self.ampvar, torch.rand_like(self.ampvar[:, probs[fatclass, :] > th])], 1),
            requires_grad=True)
        self.amp = torch.nn.Parameter(0.5 * self.amp, requires_grad=True)
        self.n_points = self.pos.shape[0]

    def double_points(self, ang_pix, dist):
        theta = torch.rand(self.pos.shape[0]).to(self.device)
        phi = torch.rand(self.pos.shape[0]).to(self.device)
        eps = (dist / (self.box_size * ang_pix)) * torch.stack(
            [torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)],
            1)
        np = self.pos + eps
        self.pos = torch.nn.Parameter(torch.cat([self.pos, np], 0), requires_grad=True)
        self.amp = torch.nn.Parameter(0.8 * self.amp, requires_grad=True)
        self.ampvar = torch.nn.Parameter(torch.cat([self.ampvar, torch.rand_like(self.ampvar)], 1),
                                         requires_grad=True)
        self.n_points = self.pos.shape[0]


class displacement_decoder(torch.nn.Module):
    def __init__(self, box_size, device, latent_dim, n_points, n_classes, n_layers, n_neurons,
                 block, pos_enc_dim, oversampling=1, mask=None):
        super(displacement_decoder, self).__init__()
        self.acth = nn.ReLU()
        self.device = device
        self.latent_dim = latent_dim
        self.box_size = box_size
        self.n_points = n_points
        self.ini = .5 * torch.ones(3)
        self.act = torch.nn.ELU()
        # self.p2v = points2mult_volume(self.box_size,n_classes)
        self.proj = point_projection(self.box_size)
        self.p2i = points2mult_image(self.box_size, n_classes, oversampling)
        self.i2F = ims2F_form(self.box_size, device, n_classes, oversampling)
        self.res_block = lin_block
        self.deform1 = self.make_layers(pos_enc_dim, latent_dim, n_neurons, n_layers, box_size)
        self.lin1b = nn.Linear(3, 3, bias=False)
        self.lin1a = nn.Linear(n_neurons, 3, bias=False)
        if pos_enc_dim == 0:
            self.lin0 = nn.Linear(3 + latent_dim, n_neurons, bias=False)
        else:
            self.lin0 = nn.Linear(3 * pos_enc_dim * 2 + latent_dim, n_neurons, bias=False)
        self.pos_enc_dim = pos_enc_dim
        self.mask = mask
        self.lin1b.weight.data.fill_(0.0)
        if box_size > 360:
            self.vol_box = box_size // 2
        else:
            self.vol_box = box_size
        self.p2v = points2mult_volume(self.vol_box, n_classes)

    def forward(self, z, r, cons, amp, ampvar, shift):
        self.batch_size = z[0].shape[0]

        if self.mask == None:

            if self.pos_enc_dim == 0:
                cons_pos = cons
            else:
                cons_pos = positional_encoding_geom2(cons, self.pos_enc_dim, self.box_size)
            # print(cons_pos.shape)
            # print(consn.shape)
            posi = cons.expand(self.batch_size, cons.shape[0], 3)
            posi_n = cons_pos.expand(self.batch_size, cons_pos.shape[0], cons_pos.shape[1])

            conf_feat = z[0].unsqueeze(1).expand(-1, cons.shape[0], -1)

            res = self.lin0(torch.cat([posi_n, conf_feat], 2))
            res = self.deform1(res)
            res = self.act(self.lin1a(res))
            res = self.lin1b(res)
            pos = posi + res
            resn = res

        else:
            posi = cons.expand(self.batch_size, cons.shape[0], 3)
            for i in range(len(self.mask)):
                consn, inds = maskpoints(cons, ampvar, self.mask[i], self.box_size)
                cons_pos = positional_encoding_geom2(consn, self.pos_enc_dim, self.box_size)

                posi_n = torch.stack(self.batch_size * [cons_pos], 0)

                conf_feat = torch.stack(consn.shape[0] * [z[i]], 0).squeeze().movedim(0, 1)
                res = self.lin0(torch.cat([posi_n, conf_feat], 2))
                res = self.deform1(res)
                res = self.act(self.lin1a(res))
                res = self.lin1b(res)
                resn = torch.zeros_like(posi)
                resn[:, inds, :] = res
                poss, posinds = maskpoints(posi + resn, ampvar, self.mask[i], self.box_size)
                # resn = res
                posi[posinds] = poss
            pos = posi

        Proj_pos = self.proj(pos, r)

        Proj_im = self.p2i(Proj_pos, torch.stack(
            self.batch_size * [amp * torch.nn.functional.softmax(ampvar, dim=0)], dim=0).to(
            self.device))
        Proj = self.i2F(Proj_im)
        Proj = fourier_shift_2d(Proj.squeeze(), shift[:, 0], shift[:, 1])

        return Proj, Proj_im, Proj_pos, pos, resn

    def make_layers(self, pos_enc_dim, latent_dim, n_neurons, n_layers, box_size):
        layers = []
        for j in range(n_layers):
            layers += [self.res_block(n_neurons, n_neurons)]

        return nn.Sequential(*layers)

    def generate_deformation(self, z, points):
        grid_pos = positional_encoding_geom2(grid, self.pos_enc_dim, self.box_size)
        posi = torch.stack(self.batch_size * [grid], 0)
        posi_n = torch.stack(self.batch_size * [grid_pos], 0)

        conf_feat = torch.stack(grid.shape[0] * [z], 0).squeeze().movedim(0, 1)
        res = self.lin0(torch.cat([posi_n, conf_feat], 2))
        res = self.deform1(res)
        res = self.act(self.lin1a(res))
        res = self.lin1b(res)
        pos = posi + res

        Vd = torch.nn.functional.grid_sample(V, pos, align_corners=False)

    def volume(self, z, r, cons, amp, ampvar, shift):
        bs = z[0].shape[0]
        _, _, _, pos, _ = self.forward(z, r, cons, amp, ampvar, shift)
        V = self.p2v(pos,
                     torch.stack(bs * [torch.nn.functional.softmax(ampvar, dim=0)], 0) * torch.clip(
                         amp, min=1).to(self.device))
        # V = self.p2v(pos,torch.stack(bs*[ampvar],0)*amp.to(self.device))
        V = torch.fft.fftn(V, dim=[-3, -2, -1])
        R, M = radial_index_mask3(self.vol_box)
        R = torch.stack(self.i2F.n_classes * [R.to(self.device)], 0)
        FF = torch.exp(-self.i2F.B[:, None, None, None] ** 2 * R) * self.i2F.A[:, None, None,
                                                                    None] ** 2
        bs = V.shape[0]
        Filts = torch.stack(bs * [FF], 0)
        Filts = torch.fft.ifftshift(Filts, dim=[-3, -2, -1])
        V = torch.real(torch.fft.ifftn(torch.sum(Filts * V, 1), dim=[-3, -2, -1]))
        return V


class inverse_displacement(torch.nn.Module):
    def __init__(self, device, latent_dim, n_points, n_layers, n_neurons, block, pos_enc_dim,
                 box_size, mask=None):
        super(inverse_displacement, self).__init__()
        self.acth = nn.ReLU()
        self.device = device
        self.latent_dim = latent_dim
        self.n_points = n_points
        self.act = torch.nn.ELU()
        self.box_size = box_size
        self.res_block = lin_block
        self.deform1 = self.make_layers(pos_enc_dim, latent_dim, n_neurons, n_layers)
        self.lin1b = nn.Linear(3, 3, bias=False)
        self.lin1a = nn.Linear(n_neurons, 3, bias=False)
        if pos_enc_dim == 0:
            self.lin0 = nn.Linear(3 + latent_dim, n_neurons, bias=False)
        else:
            self.lin0 = nn.Linear(3 * pos_enc_dim * 2 + latent_dim, n_neurons, bias=False)
        self.pos_enc_dim = pos_enc_dim
        self.mask = mask
        self.lin1b.weight.data.fill_(0.0)

    def make_layers(self, pos_enc_dim, latent_dim, n_neurons, n_layers):
        layers = []
        for j in range(n_layers):
            layers += [self.res_block(n_neurons, n_neurons)]

        return nn.Sequential(*layers)

    def forward(self, z, pos):
        self.batch_size = z[0].shape[0]

        if self.mask == None:
            posn = pos
            if self.pos_enc_dim == 0:
                enc_pos = posn
            else:
                enc_pos = positional_encoding_geom2(posn, self.pos_enc_dim, self.box_size)

            conf_feat = torch.stack(posn.shape[1] * [z[0]], 0).squeeze().movedim(0, 1)

            res = self.lin0(torch.cat([enc_pos, conf_feat], 2))
            res = self.deform1(res)
            res = self.act(self.lin1a(res))
            res = self.lin1b(res)
            c_pos = posn + res

        return c_pos


# class displacement_decoder(torch.nn.Module):
#     def __init__(self,box_size,device,latent_dim,n_points,n_classes,n_layers,n_neurons,block,pos_enc_dim,oversampling = 1, mask = None):
#         super(displacement_decoder,self).__init__()
#         self.acth = nn.ReLU()
#         self.latent_dim = latent_dim
#         self.box_size = box_size
#         self.n_points = n_points
#         self.ini = .5*torch.ones(3)
#         self.act = torch.nn.Tanh()
#         self.p2v = points2mult_volume(self.box_size, device,n_classes)
#         self.proj = point_projection(device,self.box_size)
#         self.p2i = points2mult_image(self.box_size,device,n_classes,oversampling)
#         #self.i2F = ims2Fim(self.box_size,device,n_classes)
#         self.i2F = ims2F_form(self.box_size,device,n_classes,oversampling)
#         self.res_block = block
#         self.deform1 = self.make_layers(n_layers,n_neurons)
#         self.lin1b = nn.Linear(3,3,bias = False)
#         self.lin1a = nn.Linear(n_neurons,3,bias = False)
#         self.lin0 = nn.Linear(3*pos_enc_dim*2+latent_dim,n_neurons,bias = False)
#         self.device = device
#         self.pos_enc_dim = pos_enc_dim
#         self.mask = mask
#         self.lin1b.weight.data.fill_(0.0)
#         self.linvar1 = nn.Linear(box_size+latent_dim,box_size)
#         self.linvar2 = nn.Linear(box_size,box_size)
#         self.linvar3 = nn.Linear(box_size,box_size)
#         self.radial,_ = radial_index_mask(self.box_size)
#         self.act1 = nn.Tanh()
#         self.actamp = nn.Sigmoid()
#         self.freq = torch.linspace(0,1,self.box_size).to(device)


#     def forward(self,z,r,cons,amp, ampvar,shift):
#         self.batch_size = z[0].shape[0]

#         if self.mask == None:
#             consn = cons
#             cons_pos = positional_encoding_geom2(consn,self.pos_enc_dim,self.box_size)
#             posi = torch.stack(self.batch_size * [cons], 0)
#             posi_n = torch.stack(self.batch_size*[cons_pos],0)


#             conf_feat = torch.stack(consn.shape[0]*[z[0]],0).squeeze().movedim(0,1)
#             res = self.lin0(torch.cat([posi_n,conf_feat],2))
#             res = self.deform1(res)
#             res = self.act(self.lin1a(res))
#             res = self.lin1b(res)
#             pos = posi + res
#             resn = res

#         else:
#             posi = torch.stack(self.batch_size * [cons], 0)
#             for i in range(len(self.mask)):
#                 consn, inds = maskpoints(cons,ampvar,self.mask[i],self.box_size)
#                 cons_pos = positional_encoding_geom2(consn,self.pos_enc_dim,self.box_size)

#                 posi_n = torch.stack(self.batch_size*[cons_pos],0)

#                 conf_feat = torch.stack(consn.shape[0]*[z[i]],0).squeeze().movedim(0,1)
#                 res = self.lin0(torch.cat([posi_n,conf_feat],2))
#                 res = self.deform1(res)
#                 res = self.act(self.lin1a(res))
#                 res = self.lin1b(res)
#                 resn = torch.zeros_like(posi)
#                 resn[:,inds,:] = res
#                 poss, posinds = maskpoints(posi+resn,ampvar,self.mask[i],self.box_size)
#                 #resn = res
#                 posi[posinds] = poss
#             pos = posi


#         Proj_pos = self.proj(pos,r)#-torch.stack(self.n_points*[shift],2)/(2*self.box_size)
#         # if a>0:
#         #     Proj_im = self.p2i(Proj_pos,torch.stack(self.batch_size*[self.ampvar]))
#         # else:

#         Proj_im = self.p2i(Proj_pos,torch.stack(self.batch_size*[amp*torch.nn.functional.softmax(ampvar,dim = 0)],dim=0).to(self.device))
#         Proj = self.i2F(Proj_im)
#         Proj = fourier_shift_2d(Proj.squeeze(),shift[:,0],shift[:,1])

#         freq=torch.stack(self.batch_size*[self.freq])
#         v = torch.cat([freq,z[0]],1)
#         v = self.act1(self.linvar1(v))
#         v = self.act1(self.linvar2(v))
#         v = self.linvar3(v)

#         m = v[:,torch.clip(self.radial,max = self.box_size-1)]


#         return Proj, Proj_im, Proj_pos, pos, resn, m

#     def make_layers(self, n_layers,n_neurons):
#         layers = []
#         for j in range(n_layers):
#             layers += [self.res_block(n_neurons,n_neurons)]

#         return nn.Sequential(*layers)


#     def generate_deformation(self,z,points):
#         grid_pos = positional_encoding_geom2(grid,self.pos_enc_dim,self.box_size)
#         posi = torch.stack(self.batch_size * [grid], 0)
#         posi_n = torch.stack(self.batch_size*[grid_pos],0)


#         conf_feat = torch.stack(grid.shape[0]*[z],0).squeeze().movedim(0,1)
#         res = self.lin0(torch.cat([posi_n,conf_feat],2))
#         res = self.deform1(res)
#         res = self.act(self.lin1a(res))
#         res = self.lin1b(res)
#         pos = posi + res

#         Vd = torch.nn.functional.grid_sample(V,pos,align_corners = False)


#     def volume(self,z,r,cons,amp, ampvar,shift):
#         bs = z[0].shape[0]
#         _,_,_,pos,_ = self.forward(z,r,cons,amp,ampvar,shift)
#         V = self.p2v(pos,torch.stack(bs*[torch.nn.functional.softmax(ampvar,dim=0)],0)*amp.to(self.device))
#         V = torch.fft.fftn(V,dim = [-3,-2,-1])
#         R, M = radial_index_mask3(self.box_size)
#         R = torch.stack(self.i2F.n_classes*[R.to(self.device)],0)
#         FF = torch.exp(-self.i2F.B[:,None,None,None]**2*R)*self.i2F.A[:,None,None,None]**2
#         bs = V.shape[0]
#         Filts = torch.stack(bs*[FF],0)
#         Filts = torch.fft.ifftshift(Filts,dim = [-3,-2,-1])
#         V = torch.real(torch.fft.ifftn(torch.sum(Filts*V,1),dim = [-3,-2,-1]))
#         return V

# class scale_correction(torch.nn.Module):
#     def __init__(self,scale_latent_dim):
#         super(scale_correction,self).__init__()
#         self.lin0 = nn.Linear(scale_latent_dim,10)
#         self.lin1 = nn.Linear(10,10)
#         self.lin2 = nn.Linear(10,1)
#         self.act = nn.ReLU()
#         self.actend = nn.Sigmoid()

#     def forward(self,z):
#         z = self.act(self.lin0(z))
#         z = self.act(self.lin1(z))
#         z = self.actend(self.lin2(z))

#         return z   

class displacement_decoder_amp(torch.nn.Module):
    def __init__(self, box_size, device, latent_dim, n_points, n_classes, n_layers, n_neurons,
                 block, pos_enc_dim, oversampling=1, mask=None):
        super(displacement_decoder_amp, self).__init__()
        self.acth = nn.ReLU()
        self.latent_dim = latent_dim
        self.box_size = box_size
        self.n_points = n_points
        self.ini = .5 * torch.ones(3)
        self.act = torch.nn.Tanh()
        self.p2v = points2mult_volume(self.box_size, device, n_classes)
        self.proj = point_projection(device, self.box_size)
        self.p2i = points2mult_image(self.box_size, device, n_classes, oversampling)
        # self.i2F = ims2Fim(self.box_size,device,n_classes)
        self.i2F = ims2F_form(self.box_size, device, n_classes, oversampling)
        self.res_block = block
        self.deform1 = self.make_layers(n_layers, n_neurons)
        self.lin1b = nn.Linear(3, 3, bias=False)
        self.lin1a = nn.Linear(n_neurons, 3, bias=False)
        self.lin0 = nn.Linear(3 * pos_enc_dim * 2 + latent_dim, n_neurons, bias=False)
        self.device = device
        self.pos_enc_dim = pos_enc_dim
        self.mask = mask
        self.linamp = nn.Linear(1, 8)
        self.linamp1 = nn.Linear(8, 8)
        self.linamp2 = nn.Linear(8, n_points)
        self.actamp = nn.Sigmoid()
        self.linvar1 = nn.Linear(latent_dim, box_size)
        self.linvar2 = nn.Linear(box_size, box_size)
        self.linvar3 = nn.Linear(box_size, box_size)
        self.radial = radial_index_mask(self.box_size)

    def forward(self, z, r, cons, amp, ampvar, shift, z_amp=None):
        self.batch_size = z[0].shape[0]

        if z_amp != None:
            ac = self.acth(self.linamp(z_amp))
            ac = self.acth(self.linamp1(ac))
            amp_corr = self.actamp(self.linamp2(ac))
            amp_corr = amp_corr.unsqueeze(1)

        else:
            amp_corr = torch.ones_like(ampvar)

        if self.mask == None:
            consn = cons
            cons_pos = positional_encoding_geom2(consn, self.pos_enc_dim, self.box_size)
            posi = torch.stack(self.batch_size * [cons], 0)
            posi_n = torch.stack(self.batch_size * [cons_pos], 0)

            conf_feat = torch.stack(consn.shape[0] * [z[0]], 0).squeeze().movedim(0, 1)
            res = self.lin0(torch.cat([posi_n, conf_feat], 2))
            res = self.deform1(res)
            res = self.act(self.lin1a(res))
            res = self.lin1b(res)
            pos = posi + res
            resn = res

        else:
            posi = torch.stack(self.batch_size * [cons], 0)
            for i in range(len(self.mask)):
                consn, inds = maskpoints(cons, ampvar, self.mask[i], self.box_size)
                cons_pos = positional_encoding_geom2(consn, self.pos_enc_dim, self.box_size)

                posi_n = torch.stack(self.batch_size * [cons_pos], 0)

                conf_feat = torch.stack(consn.shape[0] * [z[i]], 0).squeeze().movedim(0, 1)
                res = self.lin0(torch.cat([posi_n, conf_feat], 2))
                res = self.deform1(res)
                res = self.act(self.lin1a(res))
                res = self.lin1b(res)
                resn = torch.zeros_like(posi)
                resn[:, inds, :] = res
                poss, posinds = maskpoints(posi + resn, ampvar, self.mask[i], self.box_size)
                # resn = res
                posi[posinds] = poss
            pos = posi

        Proj_pos = self.proj(pos, r)  # -torch.stack(self.n_points*[shift],2)/(2*self.box_size)
        # if a>0:
        #     Proj_im = self.p2i(Proj_pos,torch.stack(self.batch_size*[self.ampvar]))
        # else:

        Proj_im = self.p2i(Proj_pos, amp_corr * torch.stack(
            self.batch_size * [amp * torch.nn.functional.softmax(ampvar, dim=0)], dim=0).to(
            self.device))
        Proj = self.i2F(Proj_im)
        Proj = fourier_shift_2d(Proj.squeeze(), shift[:, 0], shift[:, 1])

        v = self.acth(self.linvar1(z))
        v = self.acth(self.linvar2(v))
        v = self.actamp(self.linvar3(v))
        m = v[:, self.radial]

        return Proj, Proj_im, Proj_pos, pos, resn, m

    def make_layers(self, n_layers, n_neurons):
        layers = []
        for j in range(n_layers):
            layers += [self.res_block(n_neurons, n_neurons)]

        return nn.Sequential(*layers)

    def generate_deformation(self, z, points):
        grid_pos = positional_encoding_geom2(grid, self.pos_enc_dim, self.box_size)
        posi = torch.stack(self.batch_size * [grid], 0)
        posi_n = torch.stack(self.batch_size * [grid_pos], 0)

        conf_feat = torch.stack(grid.shape[0] * [z], 0).squeeze().movedim(0, 1)
        res = self.lin0(torch.cat([posi_n, conf_feat], 2))
        res = self.deform1(res)
        res = self.act(self.lin1a(res))
        res = self.lin1b(res)
        pos = posi + res

        Vd = torch.nn.functional.grid_sample(V, pos, align_corners=False)

    def volume(self, z, r, cons, amp, ampvar, shift, z_amp):
        bs = z[0].shape[0]
        _, _, _, pos, _, amp_corr = self.forward(z, r, cons, amp, ampvar, shift, z_amp)
        V = self.p2v(pos, amp_corr * torch.stack(bs * [torch.nn.functional.softmax(ampvar, dim=0)],
                                                 0) * amp.to(self.device))
        V = torch.fft.fftn(V, dim=[-3, -2, -1])
        R, M = radial_index_mask3(self.box_size)
        R = torch.stack(self.i2F.n_classes * [R.to(self.device)], 0)
        FF = torch.exp(-self.i2F.B[:, None, None, None] ** 2 * R) * self.i2F.A[:, None, None,
                                                                    None] ** 2
        bs = V.shape[0]
        Filts = torch.stack(bs * [FF], 0)
        Filts = torch.fft.ifftshift(Filts, dim=[-3, -2, -1])
        V = torch.real(torch.fft.ifftn(torch.sum(Filts * V, 1), dim=[-3, -2, -1]))
        return V


# class displacement_decoder_amp(torch.nn.Module):
#     def __init__(self, box_size, device, latent_dim, n_points, n_classes, n_layers, n_neurons, block, pos_enc_dim):
#         super(displacement_decoder_amp, self).__init__()
#         self.acth = nn.ReLU()
#         self.latent_dim = latent_dim
#         self.box_size = box_size
#         self.n_points = n_points
#         self.ini = .5 * torch.ones(3)
#         self.pos = torch.nn.Parameter(0.015 * (torch.rand(n_points, 3) - self.ini), requires_grad=True)
#         self.act = torch.nn.Tanh()
#         self.p2v = points2mult_volume(self.box_size, device, n_classes)
#         self.proj = point_projection(device, self.box_size)
#         self.p2i = points2mult_image(self.box_size, device, n_classes)
#         # self.i2F = ims2Fim(self.box_size,device,n_classes)
#         self.i2F = ims2F_form(self.box_size, device, n_classes)
#         self.res_block = block
#         self.deform1 = self.make_layers(n_layers, n_neurons)
#         self.lin1b = nn.Linear(3, 3, bias=False)
#         self.lin1a = nn.Linear(n_neurons, 3, bias=False)
#         self.lin0 = nn.Linear(3 * pos_enc_dim * 2 + latent_dim-1, n_neurons, bias=False)
#         self.device = device
#         self.pos_enc_dim = pos_enc_dim
#         self.linamp = nn.Linear(1, 8)
#         self.linamp1 = nn.Linear(8,8)
#         self.linamp2 = nn.Linear(8,n_points)
#         self.actamp = nn.Sigmoid()

#     def forward(self, z, r, cons, amp, ampvar, shift):
#         self.batch_size = z.shape[0]
#         cons_pos = positional_encoding_geom2(cons, self.pos_enc_dim, self.box_size)
#         posi = torch.stack(self.batch_size * [cons], 0)
#         posi_n = torch.stack(self.batch_size * [cons_pos], 0)

#         conf_feat = torch.stack(self.n_points * [z[:,:-1]], 0).squeeze().movedim(0, 1)
#         res = self.lin0(torch.cat([posi_n, conf_feat], 2))
#         res = self.deform1(res)
#         res = self.act(self.lin1a(res))
#         res = self.lin1b(res)
#         pos = posi + res

#         ac = self.acth(self.linamp(z[:,-1:]))
#         ac = self.acth(self.linamp1(ac))
#         self.amp_corr = self.actamp(self.linamp2(ac))
#         Proj_pos = self.proj(pos, r)-shift.unsqueeze(2)
#         # if a>0:
#         #     Proj_im = self.p2i(Proj_pos,torch.stack(self.batch_size*[self.ampvar]))
#         # else:

#         Proj_im = self.p2i(Proj_pos,amp * self.amp_corr.unsqueeze(1).to(self.device))
#         Proj = self.i2F(Proj_im)
#         return Proj, Proj_im, Proj_pos, pos, res

#     def make_layers(self, n_layers, n_neurons):
#         layers = []
#         for j in range(n_layers):
#             layers += [self.res_block(n_neurons, n_neurons)]

#         return nn.Sequential(*layers)

#     def volume(self, z, r, cons, amp, ampvar,shift):
#         bs = z.shape[0]
#         _, _, _, pos, _ = self.forward(z, r, cons, amp, ampvar,shift)
#         V = self.p2v(pos, self.amp_corr.unsqueeze(1) * amp.to(self.device))
#         V = torch.fft.fftn(V, dim=[-3, -2, -1])
#         R, M = radial_index_mask3(self.box_size)
#         R = torch.stack(self.i2F.n_classes * [R.to(self.device)], 0)
#         FF = torch.exp(-self.i2F.B[:, None, None, None] ** 2 * R) * self.i2F.A[:, None, None, None] ** 2
#         bs = V.shape[0]
#         Filts = torch.stack(bs * [FF], 0)
#         Filts = torch.fft.ifftshift(Filts, dim=[-3, -2, -1])
#         V = torch.real(torch.fft.ifftn(torch.sum(Filts * V, 1), dim=[-3, -2, -1]))
#         return V


class image_decoder(torch.nn.Module):
    def __init__(self, box_size, device, latent_dim, n_layers, n_neurons, block):
        super(image_decoder, self).__init__()
        self.acth = nn.ReLU()
        self.latent_dim = latent_dim
        self.box_size = box_size
        self.res_block = block
        self.deform1 = self.make_layers(n_layers, n_neurons)
        self.lin1 = nn.Linear(n_neurons, box_size ** 2, bias=False)
        self.lin0 = nn.Linear(latent_dim + 3, n_neurons, bias=False)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding='same')
        self.device = device

    def forward(self, z, r, y):
        self.batch_size = z.shape[0]
        feat = torch.cat([z, r], 1)
        feat = self.acth(self.lin0(feat))
        feat = self.deform1(feat)
        feat = self.lin1(feat)
        feat = feat.reshape(self.batch_size, self.box_size, self.box_size)
        feat = torch.stack([feat, y], 1)
        feat = self.acth(self.conv1(feat))
        feat = self.acth(self.conv2(feat))
        def_field = self.conv3(feat)
        def_field = def_field.reshape(self.batch_size, self.box_size, self.box_size, 2)
        out = torch.nn.functional.grid_sample(y.unsqeeze(1), def_field, mode='bilinear',
                                              align_corners=False)
        return out

    def make_layers(self, n_layers, n_neurons):
        layers = []
        for j in range(n_layers):
            layers += [self.res_block(n_neurons, n_neurons)]

        return nn.Sequential(*layers)


class res_block(torch.nn.Module):
    def __init__(self, pos_enc_dim, latent_dim, n_neurons, n_layers, box_size):
        super(res_block, self).__init__()
        self.lin0 = nn.Linear(3 + latent_dim, n_neurons, bias=False)
        self.lin1 = nn.Linear(n_neurons, n_neurons, bias=False)
        self.lin1a = nn.Linear(n_neurons, 3, bias=False)
        self.act = nn.ELU()
        self.n_layers = n_layers
        self.pos_enc_dim = pos_enc_dim
        self.box_size = box_size
        self.lin1a.weight.data.fill_(0.0)
        # self.act = nn.ReLU()

    def forward(self, input_list):
        z = input_list[1]
        consn = input_list[0]
        batch_size = z[0].shape[0]
        cons_pos = consn
        # cons_pos = positional_encoding_geom2(consn,self.pos_enc_dim,self.box_size)
        if len(cons_pos.shape) < 3:
            posi_n = torch.stack(batch_size * [cons_pos], 0)
        else:
            posi_n = cons_pos
        conf_feat = torch.stack(posi_n.shape[1] * [z[0]], 1).squeeze()
        res = self.act(self.lin0(torch.cat([posi_n, conf_feat], 2)))
        res = self.act(self.lin1(res))
        res = self.lin1a(res)
        res = consn + 1 / self.n_layers * res
        return [res, z]


class lin_block(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(lin_block, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim, bias=False)
        self.act = nn.ELU()

    def forward(self, x):
        res = self.act(self.lin(x))
        return res
