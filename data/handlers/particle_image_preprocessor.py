#!/usr/bin/env python3

"""
Module for particle image preprocessing
"""

from typing import Tuple, TypeVar, Dict, Any, Union

import numpy as np

import torch

from voxelium.base.grid import dht, smooth_circular_mask, smooth_square_mask, \
    get_spectral_indices, get_spectral_avg, spectrum_to_grid

Tensor = TypeVar('torch.tensor')


def get_spectral_stats(img_stack: Union[Tensor, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(img_stack.shape) != 3:
        raise RuntimeError(f"Input is not a stack")
    if len(img_stack) < 2:
        raise RuntimeError(f"Image stack too small")

    spectral_indices = get_spectral_indices(img_stack.shape[1:])

    if torch.is_tensor(img_stack):
        stack_mean = torch.mean(img_stack, 0).cpu().numpy()
        stack_square_mean = torch.mean(torch.square(img_stack), 0).cpu().numpy()
        stack_std = torch.std(img_stack, 0).cpu().numpy()
    else:
        stack_mean = np.mean(img_stack, axis=0)
        stack_square = np.mean(np.sqaure(img_stack), axis=0)
        stack_std = np.std(img_stack, axis=0)

    mean_spectrum = get_spectral_avg(stack_mean, spectral_indices=spectral_indices)
    square_mean_spectrum = get_spectral_avg(stack_square_mean, spectral_indices=spectral_indices)
    std_spectrum = get_spectral_avg(stack_std, spectral_indices=spectral_indices)

    cutoff_idx = img_stack.shape[-1] // 2 + 1

    # No standardization beyond Nyqvist
    mean_spectrum[cutoff_idx:] = 0
    std_spectrum[cutoff_idx:] = 1

    return mean_spectrum, square_mean_spectrum, std_spectrum


class ParticleImagePreprocessor:
    def __init__(self) -> None:
        self.image_size = None
        self.make_stats = None
        self.spectral_mean = None
        self.spectral_std = None
        self.circular_mask_radius = None
        self.circular_mask_thickness = None

        self.spectral_square_mean = None
        self.spectral_sigma2 = None

        self.precompute_spectral_mean = None
        self.precompute_spectral_std = None
        self.precompute_spectral_sigma2 = None
        self.precompute_spectral_mask = None
        self.precompute_circular_mask = None
        self.precompute_square_mask = None

    def initialize(
            self,
            image_size: int,
            circular_mask_radius: float,
            circular_mask_thickness: float,
            spectral_mean=None,
            spectral_std=None,
            spectral_square_mean=None,
    ) -> None:
        self.image_size = image_size
        self.circular_mask_radius = circular_mask_radius
        self.circular_mask_thickness = circular_mask_thickness
        self.spectral_mean = spectral_mean
        self.spectral_std = spectral_std
        self.spectral_square_mean = spectral_square_mean

        self._precompute()

    def initialize_from_stack(
            self,
            stack: np.ndarray,
            circular_mask_radius: float,
            circular_mask_thickness: float,
    ) -> None:
        self.image_size = stack.shape[-1]
        self.circular_mask_radius = circular_mask_radius
        self.circular_mask_thickness = circular_mask_thickness

        stack_ht = dht(stack, dim=2)
        self.spectral_mean, self.spectral_square_mean, self.spectral_std = get_spectral_stats(stack_ht)

        self.spectral_sigma2 = self.spectral_square_mean - np.square(self.spectral_mean)

        # import matplotlib.pylab as plt
        # x = np.arange(len(self.spectral_sigma2))
        # plt.plot(x, self.spectral_sigma2, "r")
        # plt.plot(x, self.spectral_square_mean, "b")
        # plt.plot(x, self.spectral_sigma2, "k")
        # plt.show()

        self._precompute()

    def _precompute(self) -> None:
        # Square mask
        self.precompute_square_mask = torch.Tensor(
            smooth_square_mask(
                image_size=self.image_size,
                square_side=self.image_size - self.circular_mask_thickness * 2,
                thickness=self.circular_mask_thickness
            )
        )

        # Circular mask
        self.precompute_circular_mask = torch.Tensor(
            smooth_circular_mask(
                image_size=self.image_size,
                radius=self.circular_mask_radius,
                thickness=self.circular_mask_thickness
            )
        )

        # Calculate standardization coefficients
        cutoff_idx = self.image_size // 2 + 1
        spectral_indices = get_spectral_indices([self.image_size] * 2)

        if self.spectral_mean is not None and \
                self.spectral_std is not None and \
                self.spectral_square_mean is not None:
            self.spectral_sigma2 = self.spectral_square_mean - np.square(self.spectral_mean)
            # If we have statistics
            self.precompute_spectral_mean = spectrum_to_grid(self.spectral_mean, spectral_indices)
            self.precompute_spectral_std = spectrum_to_grid(self.spectral_std, spectral_indices)
            self.precompute_spectral_sigma2 = spectrum_to_grid(self.spectral_sigma2, spectral_indices)
            self.precompute_spectral_mask = \
                ((spectral_indices < cutoff_idx) & (self.precompute_spectral_std > 1e-5))

            self.precompute_spectral_mean = torch.Tensor(self.precompute_spectral_mean)
            self.precompute_spectral_std = torch.Tensor(self.precompute_spectral_std)
            self.precompute_spectral_sigma2 = torch.Tensor(self.precompute_spectral_sigma2)
            self.precompute_spectral_mask = torch.Tensor(self.precompute_spectral_mask)
        else:
            # If we don't have statistics, make grids of ones rather than raising an exception
            self.precompute_spectral_mean = torch.ones_like(self.precompute_square_mask)
            self.precompute_spectral_std = torch.ones_like(self.precompute_square_mask)
            self.precompute_spectral_sigma2 = torch.ones_like(self.precompute_spectral_sigma2)
            self.precompute_spectral_mask = torch.ones_like(self.precompute_square_mask)

    def set_device(self, device: Any) -> None:
        self.precompute_square_mask = self.precompute_square_mask.to(device)
        self.precompute_circular_mask = self.precompute_circular_mask.to(device)
        self.precompute_spectral_mean = self.precompute_spectral_mean.to(device)
        self.precompute_spectral_std = self.precompute_spectral_std.to(device)
        self.precompute_spectral_sigma2 = self.precompute_spectral_sigma2.to(device)
        self.precompute_spectral_mask = self.precompute_spectral_mask.to(device)

    def apply_square_mask(self, img_stack: Tensor) -> Tensor:
        return img_stack * self.precompute_square_mask[None, ...]

    def apply_circular_mask(self, img_stack: Tensor) -> Tensor:
        return img_stack * self.precompute_circular_mask[None, ...]

    def apply_translation(
            self, grids: Tensor,
            shift: Union[np.ndarray, Tensor],
            shift_y: Union[np.ndarray, Tensor] = None
    ) -> Tensor:
        # Generate the 3x4 matrix for affine grid RR = (E|S) = (eye|shift)
        if torch.is_tensor(shift):
            if shift_y is None:
                S = shift
            else:
                S = torch.stack([shift, shift_y], 1)
        else:
            if shift_y is None:
                S = torch.tensor(shift)
            else:
                S = torch.tensor(np.stack([shift, shift_y], 1))

        S = -S.unsqueeze(2) * 2 / self.image_size
        B = S.shape[0]
        I = torch.eye(2).unsqueeze(0).to(S.device)

        RR = torch.cat([torch.tile(I, (B, 1, 1)), S], 2)

        # generate affine Grid
        grid = torch.nn.functional.affine_grid(RR, (B, 1, self.image_size, self.image_size), align_corners=False)
        grid = grid.to(grids.device)

        # apply shift
        img_stack_out = torch.nn.functional.grid_sample(
            input=grids.reshape(B, 1, self.image_size, self.image_size).float(),
            grid=grid.float(),
            mode='bilinear',
            align_corners=False
        )
        img_stack_out = torch.squeeze(img_stack_out, 1)
        return img_stack_out

    def get_state_dict(self) -> Dict:
        return {
            "type": "ParticleImagePreprocessor",
            "version": "0.0.1",
            "image_size": self.image_size,
            "spectral_mean": self.spectral_mean,
            "spectral_std": self.spectral_std,
            "spectral_square_mean": self.spectral_square_mean,
            "circular_mask_radius": self.circular_mask_radius,
            "circular_mask_thickness": self.circular_mask_thickness
        }

    def set_state_dict(self, state_dict) -> None:
        if "type" not in state_dict or state_dict["type"] != "ParticleImagePreprocessor":
            raise TypeError("Input is not an 'ParticleImagePreprocessor' instance.")

        if "version" not in state_dict:
            raise RuntimeError("ParticleImagePreprocessor instance lacks version information.")

        if state_dict["version"] == "0.0.1":
            self.initialize(
                image_size=state_dict["image_size"],
                spectral_mean=state_dict["spectral_mean"],
                spectral_std=state_dict["spectral_std"],
                spectral_square_mean=state_dict["spectral_square_mean"],
                circular_mask_radius=state_dict["circular_mask_radius"],
                circular_mask_thickness=state_dict["circular_mask_thickness"],
            )
        else:
            raise RuntimeError(f"Version '{state_dict['version']}' not supported.")
