from typing import Union

import torch

from .constants import RegularizationMode
from ..utils.utils_new import scatter


class GeometricLoss:
    def __init__(
        self,
        mode: RegularizationMode,
        neighbour_loss_weight: float = 0.01,
        repulsion_weight: float = 0.01,
        outlier_weight: float = 1,
        deformation_regularity_weight: float = 0.1,
    ):
        self.mode = mode
        self.neighbour_loss_weight = neighbour_loss_weight
        self.repulsion_weight = repulsion_weight
        self.outlier_weight = outlier_weight
        self.deformation_regularity_weight = deformation_regularity_weight
        print('mode is:', self.mode)

    def __call__(
        self,
        deformed_positions: torch.Tensor,
        mean_neighbour_distance: float,
        mean_graph_distance: float,
        consensus_pairwise_distances: torch.Tensor,
        knn: torch.Tensor,
        radius_graph: torch.Tensor,
        box_size: int,
        ang_pix: float,
        n_active_points: int,
    ) -> float:

        positions_angstroms = deformed_positions * box_size * ang_pix
        deformation_regularity_loss = self.calculate_deformation_regularity_loss(
            positions=positions_angstroms,
            radius_graph=radius_graph,
            consensus_pairwise_distances=consensus_pairwise_distances,
            n_active_points=n_active_points
        )
        loss = self.deformation_regularity_weight * deformation_regularity_loss
        if self.mode != RegularizationMode.MODEL:
            neighbour_loss = self.calculate_neighbour_loss(
                positions=positions_angstroms,
                radius_graph=radius_graph,
                mean_neighbour_distance=mean_neighbour_distance,
            )
            repulsion_loss = self.calculate_repulsion_loss(
                positions=positions_angstroms,
                radius_graph=radius_graph,
                mean_neighbour_distance=mean_graph_distance,
            )
            outlier_loss = self.calculate_outlier_loss(
                positions=positions_angstroms,
                knn=knn,
                mean_neighbour_distance=mean_neighbour_distance,
            )

            loss += (
                self.neighbour_loss_weight * neighbour_loss
                + self.repulsion_weight * repulsion_loss
                + self.outlier_weight * outlier_loss
            )
            # a = torch.randint(0, 1000, (2, 2))
            # if a[0, 0] > 500:
            #     print('deformation_loss:', self.deformation_regularity_weight *
            #           deformation_regularity_loss)
            #     print('neighbour_loss:', self.neighbour_loss_weight*neighbour_loss)
            #     print('repulsion_loss:', self.repulsion_weight*repulsion_loss)
            #     print('outlier_loss:', self.outlier_weight*outlier_loss)

        return loss

    def calculate_neighbour_loss(
        self,
        positions: torch.Tensor,
        radius_graph: torch.Tensor,
        mean_neighbour_distance: float
    ):
        """Loss term enforcing optimal local clustering of points.

        This adds a quadratic penalty on having a number of neighbours less than 1 or
        greater than 3 for each point.

        Parameters
        ----------
        positions: torch.Tensor
            in angstroms
        radius_graph: torch.Tensor
            (2, n_edges) set of indices into positions

        """
        differences = positions[:, radius_graph[0]] - \
            positions[:, radius_graph[1]]
        distances = torch.pow(torch.sum(differences ** 2, dim=-1) + 1e-7, 0.5)
        distance_activation = _distance_activation(
            distances, mean_neighbour_distance)
        n_neighbours = scatter(distance_activation, radius_graph[0])
        neighbour_activation = _neighbour_activation(
            n_neighbours, minimum=1, maximum=3)
        return torch.mean(neighbour_activation)

    def calculate_repulsion_loss(
        self,
        positions: torch.Tensor,
        radius_graph: torch.Tensor,
        mean_neighbour_distance: float,
    ):
        """

        Parameters
        ----------
        positions: torch.Tensor
            in angstroms
        radius_graph: torch.Tensor
            (2, n_edges) set of indices into positions
        mean_neighbour_distance: float
            mean of distance to neighbours
        """
        differences = positions[:, radius_graph[0]] - \
            positions[:, radius_graph[1]]
        distances = torch.pow(torch.sum(differences ** 2, dim=-1), 0.5)

        # set cutoff distance for repulsion penalty
        if mean_neighbour_distance < 0.5:
            cutoff_distance = 0.5
        else:
            cutoff_distance = mean_neighbour_distance

        # add quadratic penalty for distance being greater than cutoff
        x1 = torch.clamp(distances, max=cutoff_distance)
        # print(torch.min(x1))
        x1 = (x1 - cutoff_distance)**2
        return torch.mean(x1)

    def calculate_outlier_loss(
        self,
        positions: torch.Tensor,
        knn: torch.Tensor,
        mean_neighbour_distance: float,
    ):
        differences = positions[:, knn[0]] - positions[:, knn[1]]
        distances = torch.pow(torch.sum(differences ** 2, dim=-1) + 1e-7, 0.5)
        cutoff = 1.5 * mean_neighbour_distance

        # penalise points further away than cutoff
        distances = torch.clamp(distances, min=cutoff)
        return torch.mean(distances - cutoff)

    def calculate_deformation_regularity_loss(
        self,
        positions: torch.Tensor,
        radius_graph: torch.Tensor,
        consensus_pairwise_distances: torch.Tensor,
        n_active_points: int
    ):
        """Average difference in pairwise distance between consensus and updated model."""
        differences = positions[:, radius_graph[0]] - \
            positions[:, radius_graph[1]]
        distances = torch.pow(torch.sum(differences ** 2, dim=-1), 0.5)
        difference_in_pairwise_distances = (
            distances - consensus_pairwise_distances) ** 2

        return torch.mean(difference_in_pairwise_distances)


def _distance_activation(pairwise_distances, mean_neighbour_distance):
    """A continuous assignment of 'neighbour-like-ness' to each point."""
    # todo: @schwab - make sure this calculation behaves as expected, maybe document it
    cutoff_distance = mean_neighbour_distance
    x2 = torch.clamp(pairwise_distances, min=cutoff_distance,
                     max=1.5 * cutoff_distance)
    x2 = (1 - (4 / mean_neighbour_distance ** 2) * (
        x2 - mean_neighbour_distance) ** 2) ** 2
    return x2


def _neighbour_activation(neighbours_per_point, minimum: float = 1, maximum: float = 3):
    """Quadratic penalisation on number of neighbours outside range."""
    x1 = torch.clamp(neighbours_per_point, max=minimum)
    x2 = torch.clamp(neighbours_per_point, min=maximum)
    x1 = (x1 - minimum) ** 2
    x2 = (x2 - maximum) ** 2
    return x1 + x2
