from typing import Tuple, Optional, List

import torch
import torch.nn
import torch.nn.functional as F
from torch import nn as nn
from tqdm import tqdm

from dynamight.models.blocks import LinearBlock
from dynamight.models.utils import positional_encoding
from dynamight.utils.utils_new import PointProjector, PointsToImages, \
    FourierImageSmoother, PointsToVolumes, \
    maskpoints, fourier_shift_2d, radial_index_mask3, radial_index_mask, my_knn_graph, my_radius_graph


# decoder turns set(s) of points into 2D image(s)
# we have two types of decoder
# - consensus decoder
#     - optimises the positions of the input gaussians
#     - optimises the poses of the particles
#
# - displacement decoder
#    - optimises the positions of the input gaussians
#    - optimises the poses of the particles
#    - optimises the displacements of the gaussians

# both decoders can optionally be masked

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

        # parameter initialisation
        self.output[-1].weight.data.fill_(0.0)
        self.model_positions = torch.nn.Parameter(
            model_positions, requires_grad=True)
        self.amp = torch.nn.Parameter(
            30 * torch.ones(n_classes, n_points), requires_grad=True
        )
        self.ampvar = torch.nn.Parameter(
            0.5 * torch.randn(n_classes, n_points), requires_grad=True
        )
        # graphs and distances
        self.neighbour_graph = []
        self.radius_graph = []
        self.model_distances = []
        self.mean_neighbour_distance = []
        self.loss_mode = []

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

    # def _update_positions_masked(self, z, positions):
    #     # todo: schwab will maybe look at this
    #     # Alister: please make sure signature matches unmasked implementation
    #     posi = positions.expand(self.z_batch_size, positions.shape[0], 3)
    #     for i in range(len(self.mask)):
    #         consn, inds = maskpoints(positions, ampvar, self.mask[i],
    #                                  self.box_size)
    #         cons_pos = positional_encoding(consn, self.pos_enc_dim,
    #                                        self.box_size)
    #
    #         posi_n = torch.stack(self.z_batch_size * [cons_pos], 0)
    #
    #         conf_feat = torch.stack(consn.shape[0] * [z[i]],
    #                                 0).squeeze().movedim(0, 1)
    #         res = self.input(torch.cat([posi_n, conf_feat], 2))
    #         res = self.layers(res)
    #         res = self.activation(self.lin1a(res))
    #         res = self.lin1b(res)
    #         resn = torch.zeros_like(posi)
    #         resn[:, inds, :] = res
    #         poss, posinds = maskpoints(posi + resn, ampvar, self.mask[i],
    #                                    self.box_size)
    #         # resn = res
    #         posi[posinds] = poss
    #     pos = posi

    def forward(
        self,
        z: torch.Tensor,  # (b, d_latent_dims)
        orientation: torch.Tensor,  # (b, 3) euler angles
        shift: torch.Tensor,  # (b, 2)
        positions: Optional[torch.Tensor] = None,  # (n_points, 3)
    ):
        """Decode latent variable into coordinate model and make a projection image."""
        if positions is None:
            positions = self.model_positions
            amp = self.amp
            ampvar = self.ampvar
        else:
            amp = torch.tensor([1.0])
            ampvar = torch.tensor([1.0])

        self.batch_size = z.shape[0]

        if self.mask is None:
            updated_positions, displacements = self._update_positions_unmasked(
                z, positions
            )
        else:  # TODO: schwab, currently not working
            # updated_positions, displacements = self._update_positions_masked(
            #     z, positions
            # )
            pass

        # turn points into images
        projected_positions = self.projector(updated_positions, orientation)
        weighted_amplitudes = torch.stack(
            self.batch_size * [amp * F.softmax(ampvar, dim=0)], dim=0
        )  # (b, n_points, n_gaussian_widths)
        weighted_amplitudes.to(self.device)
        projection_images = self.p2i(projected_positions, weighted_amplitudes)
        projection_images = self.image_smoother(projection_images)
        projection_images = fourier_shift_2d(
            projection_images.squeeze(), shift[:, 0], shift[:, 1]
        )
        return projection_images, updated_positions, displacements

    def initialize_physical_parameters(
        self,
        reference_volume: torch.Tensor,
        lr: float = 0.001,
        n_epochs: int = 300,
    ):
        optimizer = torch.optim.Adam(self.physical_parameters, lr=lr)
        print(
            'Initializing gaussian positions from reference deformable_backprojection')
        for i in tqdm(range(n_epochs)):
            optimizer.zero_grad()
            V = self.generate_consensus_volume()
            loss = torch.nn.functional.mse_loss(
                V[0].float(), reference_volume.to(V.device))
            loss.backward()
            optimizer.step()
        print('Final error:', loss.item())

    def compute_neighbour_graph(self):
        positions_ang = self.model_positions * self.box_size * self.ang_pix
        knn = my_knn_graph(positions_ang, 2, workers=8)
        differences = positions_ang[knn[0]] - positions_ang[knn[1]]
        neighbour_distances = torch.linalg.norm(differences, dim=1)
        self.neighbour_graph = knn
        self.mean_neighbour_distance = torch.mean(neighbour_distances)

    def compute_radius_graph(self):
        positions_ang = self.model_positions * self.box_size * self.ang_pix
        self.radius_graph = my_radius_graph(
            positions_ang, r=1.5*self.mean_neighbour_distance, workers=8
        )
        differences = positions_ang[self.radius_graph[0]] - positions_ang[self.radius_graph[1]]
        self.model_distances = torch.linalg.norm(differences)

    @property
    def physical_parameters(self) -> torch.nn.ParameterList:
        """Parameters which make up a coordinate model."""
        params = [
            self.model_positions,
            self.amp,
            self.ampvar,
            self.image_smoother.A,
            self.image_smoother.B,
        ]
        return params

    @property
    def network_parameters(self) -> List[torch.nn.Parameter]:
        """Parameters which are not physically meaninful in a coordinate model."""
        network_params = [
            param for param in self.parameters()
            if param not in self.physical_parameters
        ]
        return network_params

    def make_layers(self, n_neurons, n_layers):
        layers = []
        for j in range(n_layers):
            layers += [self.block(n_neurons, n_neurons)]
        return nn.Sequential(*layers)

    def generate_consensus_volume(self):
        if self.box_size > 360:
            vol_box = self.box_size // 2
        else:
            vol_box = self.box_size
        self.batch_size = 2
        p2v = PointsToVolumes(vol_box, self.n_classes)
        amplitudes = torch.stack(
            2 * [torch.nn.functional.softmax(self.ampvar, dim=0)], dim=0
        ) * self.amp.to(self.device)
        volume = p2v(positions=torch.stack(
            2*[self.model_positions], 0), amplitudes=amplitudes)
        volume = torch.fft.fftn(volume, dim=[-3, -2, -1])
        R, M = radial_index_mask3(self.vol_box)
        R = torch.stack(self.image_smoother.n_classes * [R.to(self.device)], 0)
        FF = torch.exp(-self.image_smoother.B[:, None, None,
                                              None] ** 2 * R) * self.image_smoother.A[:, None, None,
                                                                                      None] ** 2
        bs = 2
        Filts = torch.stack(bs * [FF], 0)
        Filts = torch.fft.ifftshift(Filts, dim=[-3, -2, -1])
        volume = torch.real(
            torch.fft.ifftn(torch.sum(Filts * volume, 1), dim=[-3, -2, -1]))
        return volume

    def generate_volume(self, z, r, cons, amp, ampvar, shift):
        # controls how points are rendered as volumes
        if self.box_size > 360:
            vol_box = self.box_size // 2
        else:
            vol_box = self.box_size
        p2v = PointsToVolumes(vol_box, self.n_classes)
        bs = z[0].shape[0]
        _, pos, _ = self.forward(z, r, cons, amp, ampvar, shift)
        V = p2v(pos,
                torch.stack(
                    bs * [torch.nn.functional.softmax(ampvar, dim=0)],
                    0) * torch.clip(
                    amp, min=1).to(self.device))
        V = torch.fft.fftn(V, dim=[-3, -2, -1])
        R, M = radial_index_mask3(self.vol_box)
        R = torch.stack(self.image_smoother.n_classes * [R.to(self.device)], 0)
        FF = torch.exp(-self.image_smoother.B[:, None, None,
                                              None] ** 2 * R) * self.image_smoother.A[
            :, None,
            None,
            None] ** 2
        bs = V.shape[0]
        Filts = torch.stack(bs * [FF], 0)
        Filts = torch.fft.ifftshift(Filts, dim=[-3, -2, -1])
        V = torch.real(
            torch.fft.ifftn(torch.sum(Filts * V, 1), dim=[-3, -2, -1]))
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
        self.batch_size = z[0].shape[0]

        if self.mask == None:
            posn = pos
            if self.pos_enc_dim == 0:
                enc_pos = posn
            else:
                enc_pos = positional_encoding(posn, self.pos_enc_dim,
                                              self.box_size)

            conf_feat = torch.stack(posn.shape[1] * [z[0]],
                                    0).squeeze().movedim(0, 1)

            res = self.lin0(torch.cat([enc_pos, conf_feat], 2))
            res = self.deform1(res)
            res = self.act(self.lin1a(res))
            res = self.lin1b(res)
            c_pos = posn + res

        return c_pos
