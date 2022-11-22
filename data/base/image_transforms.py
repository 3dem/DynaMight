#!/usr/bin/env python3

"""
Module for pytorch transformations of particle images
"""

from typing import Tuple, TypeVar, Dict, Any, Union

import numpy as np

import torch

from voxelium.base.grid import dft, idft, dht, idht, spectrum_to_grid, get_spectral_indices

Tensor = TypeVar('torch.tensor')


class SquareMask(object):
    """Applies a square mask to the image
    """

    def __init__(self, image_size, square_side, thickness):
        square_side_2 = square_side / 2.
        y, x = np.meshgrid(
            np.linspace(-image_size // 2, image_size // 2 - 1, image_size),
            np.linspace(-image_size // 2, image_size // 2 - 1, image_size)
        )
        p = np.max([np.abs(x), np.abs(y)], axis=0)
        band_mask = (square_side_2 <= p) & (p <= square_side_2 + thickness)
        p_band_mask = p[band_mask]
        self.mask = np.zeros((image_size, image_size))
        self.mask[p < square_side_2] = 1
        self.mask[band_mask] = np.cos(np.pi * (p_band_mask - square_side_2) / thickness) / 2 + .5
        self.mask[square_side_2 + thickness < p] = 0
        self.mask = torch.Tensor(self.mask)

    def __call__(self, image):
        self.mask = self.mask.to(image.device)
        return image * self.mask

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CircularMask(object):
    """Applies a circular smooth mask to the image
    """

    def __init__(self, image_size, radius, thickness):
        y, x = np.meshgrid(
            np.linspace(-image_size // 2, image_size // 2 - 1, image_size),
            np.linspace(-image_size // 2, image_size // 2 - 1, image_size)
        )
        r = np.sqrt(x ** 2 + y ** 2)
        band_mask = (radius <= r) & (r <= radius + thickness)
        r_band_mask = r[band_mask]
        self.mask = np.zeros((image_size, image_size))
        self.mask[r < radius] = 1
        self.mask[band_mask] = np.cos(np.pi * (r_band_mask - radius) / thickness) / 2 + .5
        self.mask[radius + thickness < r] = 0
        self.mask = torch.Tensor(self.mask)

    def __call__(self, image):
        self.mask = self.mask.to(image.device)
        return image * self.mask

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CenterDFT(object):
    """Applies a centered DFT to the image
    """

    def __call__(self, image):
        return dft(image)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CenterIDFT(object):
    """Applies a centered inverse DFT to the image
    """

    def __call__(self, image):
        return idft(image)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CenterDHT(object):
    """Applies a centered DHT to the image
    """

    def __call__(self, image):
        return dht(image)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CenterIDHT(object):
    """Applies a centered inverse DHT to the image
    """

    def __call__(self, image):
        return idht(image)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def _spectral_standardization_stats(image_size, mean, std):
    device = mean.device
    cutoff_idx = image_size // 2 + 1
    spectral_indices = get_spectral_indices([image_size] * 2)

    mean = spectrum_to_grid(mean.cpu().numpy(), spectral_indices)
    std = spectrum_to_grid(std.cpu().numpy(), spectral_indices)
    mask = (spectral_indices < cutoff_idx) & (std > 1e-5)

    mean = torch.Tensor(mean).to(device)
    std = torch.Tensor(std).to(device)
    mask = torch.Tensor(mask).to(device)

    return mean, std, mask


class SpectralStandardize(object):
    """Applies a spectral standardization
    """

    def __init__(self, image_size, mean, std):
        self.mean, self.std, self.mask = _spectral_standardization_stats(image_size, mean, std)

    def __call__(self, image):
        return ((image - self.mean[None, ...]) / self.std[None, ...]) * self.mask[None, ...]

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SpectralDestandardize(object):
    """Inverses an applied spectral standardization
    """

    def __init__(self, image_size, mean, std):
        self.mean, self.std, self.mask = _spectral_standardization_stats(image_size, mean, std)

    def __call__(self, image):
        return (image * self.std[None, ...] + self.mean[None, ...]) * self.mask[None, ...]

    def __repr__(self):
        return self.__class__.__name__ + '()'
