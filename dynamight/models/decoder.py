from typing import Tuple, Optional, List

import torch
import torch.nn
import torch.nn.functional as F
from torch import nn as nn
from tqdm import tqdm
import mrcfile
from dynamight.models.blocks import LinearBlock
from dynamight.models.utils import positional_encoding, initialize_points_from_volume
from dynamight.utils.utils_new import PointProjector, PointsToImages, \
    FourierImageSmoother, PointsToVolumes, \
    maskpoints, fourier_shift_2d, radial_index_mask3, radial_index_mask, knn_graph, radius_graph, fourier_crop3, compute_threshold, add_weight_decay_to_named_parameters, FSC, graph_union
import numpy as np


# decoder turns set(s) of points into 2D image(s)


class DisplacementDecoder(torch.nn.Module):
    def __init__(
        self,
        box_size,
        ang_pix,
        device,
        n_latent_dims,  # input dimensionality
        n_points,  # number of points to decode to
        n_classes,  # number of 'types of point'
        n_layers,
        n_neurons_per_layer,
        block=LinearBlock,
        pos_enc_dim=10,
        grid_oversampling_factor=1,
        mask=None,
        model_positions: Optional[torch.Tensor] = None
    ):
        super(DisplacementDecoder, self).__init__()
        self.device = device
        self.latent_dim = n_latent_dims
        self.box_size = box_size
        self.ang_pix = ang_pix
        self.vol_box = box_size
        self.n_points = n_points
        self.pos_enc_dim = pos_enc_dim
        self.mask = mask
        self.block = block  # block type can be LinearBlock or ResBlock
        self.activation = torch.nn.ELU()
        self.n_classes = n_classes
        n_input_neurons = n_latent_dims + \
            3 if pos_enc_dim == 0 else 6 * pos_enc_dim + n_latent_dims

        # fully connected network stuff
        self.input = nn.Linear(
            n_input_neurons, n_neurons_per_layer, bias=False)
        self.layers = self.make_layers(n_neurons_per_layer, n_layers)
        self.output = torch.nn.Sequential(
            nn.Linear(n_neurons_per_layer, 3, bias=False),
            self.activation,
            nn.Linear(3, 3, bias=False),
        )

        # coordinate model to image stuff
        self.projector = PointProjector(self.box_size)

        # todo: fuse Points2Images and FourierImageSmoothier into Imager module
        # ideally two methods, points_to_grid and grid_to_image
        self.p2i = PointsToImages(
            self.box_size, n_classes, grid_oversampling_factor)
        self.image_smoother = FourierImageSmoother(
            self.box_size, device, n_classes, grid_oversampling_factor
        )
        self.p2v = PointsToVolumes(
            box_size, n_classes, grid_oversampling_factor)

        # parameter initialisation
        self.output[-1].weight.data.fill_(0.0)
        self.model_positions = torch.nn.Parameter(
            model_positions, requires_grad=True)
        # self.amp = torch.nn.Parameter(
        #    10 * torch.ones(n_classes, n_points), requires_grad=True
        # )
        self.amp = torch.nn.Parameter(
            50 * torch.ones(n_classes, n_points), requires_grad=False
        )
        self.ampvar = torch.nn.Parameter(
            torch.randn(n_classes, n_points), requires_grad=True
        )
        # graphs and distances
        self.neighbour_graph = []
        self.radius_graph = []
        self.model_distances = []
        self.mean_neighbour_distance = []
        self.mean_graph_distance = []
        self.loss_mode = []
        self.warmup = True
        self.grid_oversampling_factor = grid_oversampling_factor
        self.masked_positions = []
        self.unmasked_positions = []
        self.n_active_points = n_points
        self.active_indices = []

    def _update_positions_unmasked(
        self, z, positions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the positions with a forward pass through the fully
        connected part of the network.

        No mask is used.
        Returns updated positions and residuals.
        """
        # do positional encoding of points
        if self.pos_enc_dim == 0:
            encoded_positions = positions
        else:
            encoded_positions = positional_encoding(
                positions, self.pos_enc_dim, self.box_size
            )

        # expand original positions to match batch side (used as residual)
        expanded_positions = positions.expand(
            self.batch_size, positions.shape[0], 3
        )

        # expand encoded positions to match batch size
        expanded_encoded_positions = encoded_positions.expand(
            self.batch_size,
            encoded_positions.shape[0],
            encoded_positions.shape[1],
        )

        # expand latent code to (b, n_positions, n_latent_dimensions)
        conf_feat = z.unsqueeze(1).expand(-1, positions.shape[0], -1)

        # concatenate positional encoded positions with expanded latent code
        x = torch.cat([expanded_encoded_positions, conf_feat], dim=2)

        # do forward pass to calculate change in position
        displacements = self.input(x)
        displacements = self.layers(displacements)
        displacements = self.output(displacements)

        # calculate final positions
        final_positions = expanded_positions + \
            displacements  # (b, n_gaussians, 3)
        return final_positions, displacements

    def forward(
        self,
        z: torch.Tensor,  # (b, d_latent_dims)
        orientation: torch.Tensor,  # (b, 3) euler angles
        shift: torch.Tensor,  # (b, 2)
        positions: Optional[torch.Tensor] = None,  # (n_points, 3)
    ):
        """Decode latent variable into coordinate model and make a projection image."""
        if positions is None:
            posin = False
            positions = self.model_positions
            amp = self.amp
            ampvar = self.ampvar
        else:
            posin = True
            # amp = torch.tensor([1.0]).to(self.device)
            amp = torch.ones(
                self.n_classes, positions.shape[0]).to(self.device)
            # ampvar = torch.tensor([1.0]).to(self.device)
            ampvar = torch.ones(
                self.n_classes, positions.shape[0]).to(self.device)

        self.batch_size = z.shape[0]

        if self.warmup == True:
            updated_positions = positions.expand(
                self.batch_size, positions.shape[0], 3
            )
            displacements = torch.zeros_like(updated_positions)
        else:
            if self.mask is None or posin == True:
                updated_positions, displacements = self._update_positions_unmasked(
                    z, positions
                )

            else:  # TODO: schwab, currently not working
                updated_positions_in_mask, displacements_in_mask = self._update_positions_unmasked(
                    z, self.unmasked_positions
                )
                updated_positions = torch.cat(
                    [updated_positions_in_mask, self.masked_positions.expand(self.batch_size, self.masked_positions.shape[0], 3)], 1)
                displacements = torch.cat(
                    [displacements_in_mask, torch.zeros_like(self.masked_positions).expand(self.batch_size, self.masked_positions.shape[0], 3)], 1)

        # turn points into images
        projected_positions = self.projector(updated_positions, orientation)
        weighted_amplitudes = torch.stack(
            self.batch_size * [amp*F.softmax(ampvar, dim=0)], dim=0
        )  # (b, n_points, n_gaussian_widths)
        # weighted_amplitudes = torch.stack(
        #    self.batch_size * [amp], dim=0
        # )
        weighted_amplitudes = weighted_amplitudes.to(self.device)
        projection_images = self.p2i(projected_positions, weighted_amplitudes)
        projection_images = self.image_smoother(projection_images)
        projection_images = fourier_shift_2d(
            projection_images.squeeze(), shift[:, 0], shift[:, 1]
        )
        if self.mask is None or self.warmup == True or posin == True:
            return projection_images, updated_positions, displacements
        else:
            return projection_images, updated_positions_in_mask, displacements_in_mask

    def initialize_physical_parameters(
        self,
        reference_volume: torch.Tensor,
        # lr: float = 0.001,
        lr: float = 0.001,
        n_epochs: int = 50,
    ):
        reference_norm = torch.sum(reference_volume**2)
        ref_amps = reference_norm/self.n_points
        # self.image_smoother.A = torch.nn.Parameter(torch.linspace(
        #    0.5*ref_amps, ref_amps, self.n_classes).to(self.device), requires_grad=True)
        print('Optimizing scale only')
        optimizer = torch.optim.Adam(
            [self.image_smoother.A], lr=100*lr)
        if reference_volume.shape[-1] > 360:
            reference_volume = torch.nn.functional.avg_pool3d(
                reference_volume.unsqueeze(0).unsqueeze(0), 2)
            reference_volume = reference_volume.squeeze()

        for i in range(n_epochs):
            optimizer.zero_grad()
            V = self.generate_consensus_volume()
            loss = torch.nn.functional.mse_loss(
                V[0].float(), reference_volume.to(V.device))
            loss.backward()
            optimizer.step()

        optimizer = torch.optim.Adam(self.physical_parameters, lr=0.1*lr)
        # self.image_smoother.B.requires_grad = False
        print(
            'Initializing gaussian positions from reference')
        for i in tqdm(range(n_epochs)):
            optimizer.zero_grad()
            V = self.generate_consensus_volume()
            loss = torch.mean((V[0].float()-reference_volume.to(V.device))**2)
            loss.backward()
            optimizer.step()

        print('Final error:', loss.item())
        self.image_smoother.A = torch.nn.Parameter(
            self.image_smoother.A*(np.pi/np.sqrt(self.box_size)), requires_grad=True)
        # self.amp.requires_grad = False
        self.amp = torch.nn.Parameter(
            self.amp*(np.pi/np.sqrt(self.box_size)), requires_grad=True)
        self.image_smoother.B.requires_grad = True
        if self.mask is None:
            self.n_active_points = self.model_positions.shape[0]
        else:
            pass

    def compute_neighbour_graph(self):
        if self.mask == None:
            positions_ang = self.model_positions.detach() * self.box_size * self.ang_pix
            knn = knn_graph(positions_ang, 2, workers=8)
            differences = positions_ang[knn[0]] - positions_ang[knn[1]]
            neighbour_distances = torch.pow(
                torch.sum(differences**2, dim=1), 0.5)
            self.neighbour_graph = knn
            self.mean_neighbour_distance = torch.mean(neighbour_distances)
        else:
            positions_ang = self.unmasked_positions.detach()*self.box_size * self.ang_pix
            knn = knn_graph(positions_ang, 2, workers=8)
            differences = positions_ang[knn[0]] - positions_ang[knn[1]]
            neighbour_distances = torch.pow(
                torch.sum(differences**2, dim=1), 0.5)
            self.neighbour_graph = knn
            self.mean_neighbour_distance = torch.mean(neighbour_distances)

    def compute_radius_graph(self):
        if self.mask == None:
            positions_ang = self.model_positions.detach() * self.box_size * self.ang_pix
            self.radius_graph = radius_graph(
                positions_ang, r=1.5*self.mean_neighbour_distance, workers=8
            )
            differences = positions_ang[self.radius_graph[0]
                                        ] - positions_ang[self.radius_graph[1]]
            # differences = positions_ang[self.neighbour_graph[0]
            #                            ] - positions_ang[self.neighbour_graph[1]]
            self.model_distances = torch.pow(torch.sum(differences**2, 1), 0.5)
            self.mean_graph_distance = torch.mean(self.model_distances)
        else:
            positions_ang = self.unmasked_positions.detach() * self.box_size * self.ang_pix
            self.radius_graph = radius_graph(
                positions_ang, r=1.5*self.mean_neighbour_distance, workers=8
            )
            differences = positions_ang[self.radius_graph[0]
                                        ] - positions_ang[self.radius_graph[1]]
            # differences = positions_ang[self.neighbour_graph[0]
            #                            ] - positions_ang[self.neighbour_graph[1]]
            self.model_distances = torch.pow(torch.sum(differences**2, 1), 0.5)
            self.mean_graph_distance = torch.mean(self.model_distances)

    def combine_graphs(self):
        positions_ang = self.model_positions.detach() * self.box_size * self.ang_pix
        self.radius_graph = graph_union(
            self.radius_graph, self.neighbour_graph)
        differences = positions_ang[self.radius_graph[0]
                                    ] - positions_ang[self.radius_graph[1]]
        self.model_distances = torch.pow(torch.sum(differences**2, 1), 0.5)
        self.mean_graph_distance = torch.mean(self.model_distances)

    def mask_model_positions(self):
        unmasked_indices = torch.round(
            (self.model_positions+0.5)*(self.box_size-1)).long().cpu()
        unmasked_positions = self.model_positions[
            self.mask[unmasked_indices[:, 0], unmasked_indices[:, 1], unmasked_indices[:, 2]] > 0.9]
        print('Number of flexible points:', unmasked_positions.shape[0])
        masked_positions = self.model_positions[
            self.mask[unmasked_indices[:, 0], unmasked_indices[:, 1], unmasked_indices[:, 2]] < 0.9]
        self.unmasked_positions = torch.nn.Parameter(
            unmasked_positions, requires_grad=True)
        self.masked_positions = torch.nn.Parameter(
            masked_positions, requires_grad=True)
        self.n_active_points = self.unmasked_positions.shape[0]
        self.active_indices = torch.where((self.mask[unmasked_indices[:, 0],
                                                     unmasked_indices[:, 1], unmasked_indices[:, 2]] > 0.9) == True)[0]

    @ property
    def physical_parameters(self) -> torch.nn.ParameterList:
        """Parameters which make up a coordinate model."""
        params = [
            self.model_positions,
            self.amp,
            self.ampvar,
            self.image_smoother.B,
            self.image_smoother.A
        ]
        return params

    @ property
    def baseline_parameters(self) -> torch.nn.ParameterList:
        """Parameters which make up a coordinate model."""
        params = [
            self.image_smoother.A,
            # self.amp
            # self.image_smoother.B
        ]
        return params

    @ property
    def network_parameters(self) -> List[torch.nn.Parameter]:
        """Parameters which are not physically meaninful in a coordinate model."""
        network_params = []
        for name, param in self.named_parameters():
            # if name not in ['image_smoother.A', 'image_smoother.B', 'amp', 'ampvar', 'model_positions']:
            network_params.append(param)

        return network_params

    def make_layers(self, n_neurons, n_layers):
        layers = []
        for j in range(n_layers):
            layers += [self.block(n_neurons, n_neurons)]
        return nn.Sequential(*layers)

    def generate_consensus_volume(self):
        scaling_fac = self.box_size/self.vol_box
        self.batch_size = 2
        p2v = PointsToVolumes(self.vol_box, self.n_classes,
                              self.grid_oversampling_factor)
        amplitudes = torch.stack(
            2 * [self.amp*torch.nn.functional.softmax(self.ampvar, dim=0)], dim=0
        )
        if self.mask is None:
            volume = p2v(positions=torch.stack(
                2*[self.model_positions], 0), amplitudes=amplitudes)
        else:
            stacked_positions = torch.stack(
                2*[torch.cat([self.masked_positions, self.unmasked_positions], 0)], 0)
            volume = p2v(positions=torch.stack(
                2*[self.model_positions], 0), amplitudes=amplitudes)

        volume = torch.fft.fftn(torch.fft.fftshift(
            volume, dim=[-1, -2, -3]), dim=[-3, -2, -1], norm='ortho')
        if self.box_size > 360:
            R, M = radial_index_mask3(
                self.grid_oversampling_factor*self.vol_box, scale=2)
        else:
            R, M = radial_index_mask3(
                self.grid_oversampling_factor*self.vol_box)
        R = torch.stack(self.image_smoother.n_classes * [R.to(self.device)], 0)

        F = torch.exp(-(scaling_fac/(self.image_smoother.B[:, None, None,
                                                           None])**2) * R**2)  # * (torch.nn.functional.softmax(self.image_smoother.A[
        FF = torch.real(torch.fft.fftn(torch.fft.fftshift(
            F, dim=[-3, -2, -1]), dim=[-3, -2, -1], norm='ortho'))*(1+self.image_smoother.A[:, None, None, None]**2)*scaling_fac / (self.image_smoother.B[:, None, None, None])

        bs = 2
        Filts = torch.stack(bs * [FF], 0)
        Filtim = torch.sum(Filts * volume, 1)
        Filtim = fourier_crop3(Filtim, self.grid_oversampling_factor)
        volume = torch.real(torch.fft.fftshift(
            torch.fft.ifftn(Filtim, dim=[-3, -2, -1], norm='ortho'), dim=[-1, -2, -3]))

        return volume

    def generate_volume(self, z, r, shift):
        # controls how points are rendered as volumes
        scaling_fac = self.box_size/self.vol_box
        p2v = PointsToVolumes(self.vol_box, self.n_classes,
                              self.grid_oversampling_factor)
        bs = z.shape[0]
        amplitudes = torch.stack(
            2 * [self.amp*torch.nn.functional.softmax(self.ampvar, dim=0)], dim=0
        )
        _, pos, _ = self.forward(z, r, shift, positions=self.model_positions)

        if self.mask is None:
            V = p2v(pos,
                    amplitudes).to(self.device)
        else:
            masked_pos = torch.stack(bs*[self.model_positions])
            masked_pos[:, self.active_indices] = pos[:, self.active_indices]
            V = p2v(masked_pos,
                    amplitudes).to(self.device)
        V = torch.fft.fftn(torch.fft.fftshift(
            V, dim=[-3, -2, -1]), dim=[-3, -2, -1], norm='ortho')
        if self.box_size > 360:
            R, M = radial_index_mask3(
                self.grid_oversampling_factor*self.vol_box, scale=2)
        else:
            R, M = radial_index_mask3(
                self.grid_oversampling_factor*self.vol_box)
        R = torch.stack(self.image_smoother.n_classes * [R.to(self.device)], 0)
        # F = torch.exp(-(1/(self.image_smoother.B[:, None, None,
        #                                          None])**2) * R**2) * (self.image_smoother.A[
        #                                              0, None,
        #                                              None,
        #                                              None]**2)
        # F = torch.exp(-(1/(self.image_smoother.B[:, None, None,
        #                                          None])**2) * R**2) * (torch.nn.functional.softmax(self.image_smoother.A[
        #                                              :, None,
        #                                              None,
        #                                              None], 0)**2)# /self.image_smoother.B[:, None, None, None])
        F = torch.exp(-(scaling_fac/(self.image_smoother.B[:, None, None,
                                                           None])**2) * R**2)  # * (torch.nn.functional.softmax(self.image_smoother.A[
        #:, None,
        # None,
        # None]))
        # F = torch.exp(-(1/(self.image_smoother.B[:, None, None,
        #                                          None])**2) * R**2) * (self.image_smoother.A[
        #                                              :, None,
        #                                              None,
        #                                              None]**2)
        FF = torch.real(torch.fft.fftn(torch.fft.fftshift(
            F, dim=[-3, -2, -1]), dim=[-3, -2, -1], norm='ortho'))*(1+self.image_smoother.A[:, None, None, None]**2)*scaling_fac/(self.image_smoother.B[:, None, None, None])
        bs = V.shape[0]
        Filts = torch.stack(bs * [FF], 0)
        Filtim = torch.sum(Filts * V, 1)
        Filtim = fourier_crop3(Filtim, self.grid_oversampling_factor)
        V = torch.real(torch.fft.fftshift(torch.fft.ifftn(
            Filtim, dim=[-3, -2, -1], norm='ortho'), dim=[-1, -2, -3]))
        return V


class InverseDisplacementDecoder(torch.nn.Module):
    def __init__(self, device, latent_dim, n_points, n_layers, n_neurons, block,
                 pos_enc_dim,
                 box_size, mask=None):
        super(InverseDisplacementDecoder, self).__init__()
        self.acth = nn.ReLU()
        self.device = device
        self.latent_dim = latent_dim
        self.n_points = n_points
        self.act = torch.nn.ELU()
        self.box_size = box_size
        self.res_block = LinearBlock
        self.deform1 = self.make_layers(pos_enc_dim, latent_dim, n_neurons,
                                        n_layers)
        self.lin1b = nn.Linear(3, 3, bias=False)
        self.lin1a = nn.Linear(n_neurons, 3, bias=False)
        if pos_enc_dim == 0:
            self.lin0 = nn.Linear(3 + latent_dim, n_neurons, bias=False)
        else:
            self.lin0 = nn.Linear(3 * pos_enc_dim * 2 + latent_dim, n_neurons,
                                  bias=False)
        self.pos_enc_dim = pos_enc_dim
        self.mask = mask
        self.lin1b.weight.data.fill_(0.0)

    def make_layers(self, pos_enc_dim, latent_dim, n_neurons, n_layers):
        layers = []
        for j in range(n_layers):
            layers += [self.res_block(n_neurons, n_neurons)]

        return nn.Sequential(*layers)

    def forward(self, z, pos):
        self.batch_size = z.shape[0]

        if self.mask == None:
            posn = pos
            if self.pos_enc_dim == 0:
                enc_pos = posn
            else:
                enc_pos = positional_encoding(posn, self.pos_enc_dim,
                                              self.box_size)
            if enc_pos.dtype != posn.dtype:
                enc_pos = enc_pos.to(posn.dtype)
            #print('converting encoding')
            conf_feat = torch.stack(posn.shape[1] * [z],
                                    0).squeeze().movedim(0, 1)

            res = self.lin0(torch.cat([enc_pos, conf_feat], 2))
            res = self.deform1(res)
            res = self.act(self.lin1a(res))
            res = self.lin1b(res)
            c_pos = posn + res

        return c_pos


def align_halfs(decoder_half1, decoder_half2, nr_gaussians=None, n_epochs=100):
    if nr_gaussians is None:
        nr_gaussians = decoder_half1.model_positions.shape[0]//70
        print(nr_gaussians)

    V1_full = decoder_half1.generate_consensus_volume()
    V1_full = V1_full[0].detach()
    with mrcfile.new('/cephfs/schwab/high_half1_pre.mrc', overwrite=True) as mrc:
        mrc.set_data((V1_full / torch.mean(V1_full)
                      ).float().detach().cpu().numpy())
    s = V1_full.shape[-1]
    F_V1 = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(
        V1_full, dim=[-1, -2, -3]), dim=[-1, -2, -3], norm='ortho'), dim=[-1, -2, -3])
    V2_full = decoder_half2.generate_consensus_volume()
    V2_full = V2_full[0].detach()
    with mrcfile.new('/cephfs/schwab/high_half2_pre.mrc', overwrite=True) as mrc:
        mrc.set_data((V2_full / torch.mean(V2_full)
                      ).float().detach().cpu().numpy())
    F_V2 = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(
        V2_full, dim=[-1, -2, -3]), dim=[-1, -2, -3], norm='ortho'), dim=[-1, -2, -3])
    F_V1 = torch.fft.fftshift(F_V1[(s//2-s//8):(s//2+s//8), (s//2-s//8):(
        s//2+s//8), (s//2-s//8):(s//2+s//8)], dim=[-1, -2, -3])
    F_V2 = torch.fft.fftshift(F_V2[(s//2-s//8):(s//2+s//8), (s//2-s//8):(
        s//2+s//8), (s//2-s//8):(s//2+s//8)], dim=[-1, -2, -3])
    V1_full = torch.real(torch.fft.fftshift(torch.fft.fftn(
        F_V1, dim=[-1, -2, -3], norm='ortho'), dim=[-1, -2, -3]))
    V2_full = torch.real(torch.fft.fftshift(torch.fft.fftn(
        F_V2, dim=[-1, -2, -3], norm='ortho'), dim=[-1, -2, -3]))
    box_size = V1_full.shape[-1]
    ang_pix = decoder_half1.ang_pix*4
    device = decoder_half1.device
    n_latent_dims = decoder_half1.latent_dim
    initial_threshold_h1 = compute_threshold(V1_full, percentage=99)
    initial_points_h1 = initialize_points_from_volume(
        V1_full.movedim(0, 2).movedim(0, 1).float().cpu(),
        threshold=initial_threshold_h1,
        n_points=nr_gaussians)
    initial_threshold_h2 = compute_threshold(V2_full, percentage=99)
    initial_points_h2 = initialize_points_from_volume(
        V2_full.movedim(0, 2).movedim(0, 1).float().cpu(),
        threshold=initial_threshold_h2,
        n_points=nr_gaussians)
    low_model_half1 = DisplacementDecoder(
        box_size=box_size, ang_pix=ang_pix, device=device, n_latent_dims=n_latent_dims, n_points=nr_gaussians, n_classes=2, n_layers=4, n_neurons_per_layer=32, block=LinearBlock, pos_enc_dim=4, model_positions=initial_points_h1).to(device)
    low_model_half1.initialize_physical_parameters(
        reference_volume=V1_full.float(), n_epochs=400)
    low_model_half2 = DisplacementDecoder(
        box_size=box_size, ang_pix=ang_pix, device=device, n_latent_dims=n_latent_dims, n_points=nr_gaussians, n_classes=2, n_layers=4, n_neurons_per_layer=32, block=LinearBlock, pos_enc_dim=4, model_positions=initial_points_h2).to(device)
    low_model_half2.initialize_physical_parameters(
        reference_volume=V2_full.float(), n_epochs=400)

    V2_low = low_model_half2.generate_consensus_volume()
    V1_low = low_model_half1.generate_consensus_volume()
    half = torch.randint(0, 2, (1,))
    if half == 0:
        low_half1_params = add_weight_decay_to_named_parameters(
            low_model_half1, weight_decay=0.)
        optimizer = torch.optim.Adam(low_half1_params, lr=1e-3)
    elif half == 1:
        low_half2_params = add_weight_decay_to_named_parameters(
            low_model_half2, weight_decay=0.)
        optimizer = torch.optim.Adam(low_half2_params, lr=1e-3)
    z = torch.zeros(2, n_latent_dims).to(device)
    r = torch.zeros(2, 3).to(device)
    shift = torch.zeros(2, 2).to(device)
    for i in range(n_epochs):
        optimizer.zero_grad()
        if half == 0:
            V1_low_now = low_model_half1.generate_volume(z, r, shift)
            loss = torch.mean((V1_low_now[0]-V2_low[0].detach())**2)
        elif half == 1:
            V2_low_now = low_model_half2.generate_volume(z, r, shift)
            loss = torch.mean((V2_low_now[0]-V1_low[0].detach())**2)
        loss.backward()
        optimizer.step()
    if half == 0:
        with mrcfile.new('/cephfs/schwab/low_half1.mrc', overwrite=True) as mrc:
            mrc.set_data((V1_low_now[0] / torch.mean(V1_low_now[0])
                          ).float().detach().cpu().numpy())
        with mrcfile.new('/cephfs/schwab/low_half2.mrc', overwrite=True) as mrc:
            mrc.set_data((V2_low[0] / torch.mean(V2_low[0])
                          ).float().detach().cpu().numpy())
    elif half == 1:
        with mrcfile.new('/cephfs/schwab/low_half2.mrc', overwrite=True) as mrc:
            mrc.set_data((V2_low_now[0] / torch.mean(V2_low_now[0])
                          ).float().detach().cpu().numpy())
        with mrcfile.new('/cephfs/schwab/low_half1.mrc', overwrite=True) as mrc:
            mrc.set_data((V1_low[0] / torch.mean(V1_low[0])
                          ).float().detach().cpu().numpy())
    # update parameters of half 1

    _, updated_positions, _ = low_model_half1.forward(
        z, r, shift, decoder_half1.model_positions)
    with torch.no_grad():
        if half == 0:
            fsc, res = FSC(V1_low_now[0].float(), V2_low[0].float())
            decoder_half1.model_positions.data = updated_positions[0]

        elif half == 1:
            fsc, res = FSC(V2_low_now[0].float(), V1_low[0].float())
            decoder_half2.model_positions.data = updated_positions[0]

    V1_full = decoder_half1.generate_consensus_volume()
    V1_full = V1_full[0].detach()
    with mrcfile.new('/cephfs/schwab/high_half1_post.mrc', overwrite=True) as mrc:
        mrc.set_data((V1_full / torch.mean(V1_full)
                      ).float().detach().cpu().numpy())
    V2_full = decoder_half2.generate_consensus_volume()
    V2_full = V2_full[0].detach()
    with mrcfile.new('/cephfs/schwab/high_half2_post.mrc', overwrite=True) as mrc:
        mrc.set_data((V2_full / torch.mean(V2_full)
                      ).float().detach().cpu().numpy())

    return fsc
