import torch
import torch.nn
from torch import nn as nn

from dynamight.models.blocks import LinearBlock
from dynamight.models.utils import positional_encoding
from dynamight.utils.utils_new import point_projection, points2mult_image, ims2F_form, points2mult_volume, \
    maskpoints, fourier_shift_2d, radial_index_mask3, radial_index_mask


class DisplacementDecoder(torch.nn.Module):
    def __init__(self, box_size, device, latent_dim, n_points, n_classes, n_layers, n_neurons,
                 block, pos_enc_dim, oversampling=1, mask=None):
        super(DisplacementDecoder, self).__init__()
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
        self.res_block = LinearBlock
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
                cons_pos = positional_encoding(cons, self.pos_enc_dim, self.box_size)
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
                cons_pos = positional_encoding(consn, self.pos_enc_dim, self.box_size)

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
        grid_pos = positional_encoding(grid, self.pos_enc_dim, self.box_size)
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


class InverseDisplacementDecoder(torch.nn.Module):
    def __init__(self, device, latent_dim, n_points, n_layers, n_neurons, block, pos_enc_dim,
                 box_size, mask=None):
        super(InverseDisplacementDecoder, self).__init__()
        self.acth = nn.ReLU()
        self.device = device
        self.latent_dim = latent_dim
        self.n_points = n_points
        self.act = torch.nn.ELU()
        self.box_size = box_size
        self.res_block = LinearBlock
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
                enc_pos = positional_encoding(posn, self.pos_enc_dim, self.box_size)

            conf_feat = torch.stack(posn.shape[1] * [z[0]], 0).squeeze().movedim(0, 1)

            res = self.lin0(torch.cat([enc_pos, conf_feat], 2))
            res = self.deform1(res)
            res = self.act(self.lin1a(res))
            res = self.lin1b(res)
            c_pos = posn + res

        return c_pos


class DisplacementDecoderAmplitude(torch.nn.Module):
    def __init__(self, box_size, device, latent_dim, n_points, n_classes, n_layers, n_neurons,
                 block, pos_enc_dim, oversampling=1, mask=None):
        super(DisplacementDecoderAmplitude, self).__init__()
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
            cons_pos = positional_encoding(consn, self.pos_enc_dim, self.box_size)
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
                cons_pos = positional_encoding(consn, self.pos_enc_dim, self.box_size)

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
        grid_pos = positional_encoding(grid, self.pos_enc_dim, self.box_size)
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
