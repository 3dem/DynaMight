#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 12:43:55 2021

@author: schwab
"""

import torch
import torch.nn as nn
import torch.fft
from ..utils.utils_new import point_projection, ims2F_form, points2mult_image, points2mult_volume, fourier_shift_2d, radial_index_mask3, maskpoints, radial_index_mask


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
        Proj_pos = self.proj(posi, r)  


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

    def generate_deformation(self, z, points, grid,V):
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
        return Vd

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

    def generate_deformation(self, z, points,grid,V):
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
        return Vd

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
