import torch
import torch.nn

from dynamight.utils.utils_new import point_projection, points2mult_image, ims2F_form, points2mult_volume, \
    fourier_shift_2d, radial_index_mask3


class ConsensusModel(torch.nn.Module):
    """Starting point for modelling deformations.

    """
    def __init__(
        self,
        box_size: int,
        device: torch.device,
        n_points: int,
        n_classes: int,
        grid_oversampling: int = 1
    ):
        super(ConsensusModel, self).__init__()
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
        self.p2i = points2mult_image(self.box_size, n_classes, grid_oversampling)
        self.i2F = ims2F_form(self.box_size, device, n_classes, grid_oversampling)
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
