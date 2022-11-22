#!/usr/bin/env python

"""
Module for calculations related to the contrast transfer function (CTF)
in cryo-EM single particle analysis.
"""

from typing import Tuple, Union, TypeVar, Dict

import numpy as np
import torch

Tensor = TypeVar('torch.tensor')


class ContrastTransferFunction:
    def __init__(
            self,
            voltage: float,
            spherical_aberration: float = 0.,
            amplitude_contrast: float = 0.,
            phase_shift: float = 0.,
            b_factor: float = 0.,
    ) -> None:
        """
        Initialization of the CTF parameter for an optics group.
        :param voltage: Voltage
        :param spherical_aberration: Spherical aberration
        :param amplitude_contrast: Amplitude contrast
        :param phase_shift: Phase shift
        :param b_factor: B-factor
        """

        if voltage <= 0:
            raise RuntimeError(
                f"Invalid value ({voltage}) for voltage of optics group {id}."
            )

        self.voltage = voltage
        self.spherical_aberration = spherical_aberration
        self.amplitude_contrast = amplitude_contrast
        self.phase_shift = phase_shift
        self.b_factor = b_factor

        # Adjust units
        spherical_aberration = spherical_aberration * 1e7
        voltage = voltage * 1e3

        # Relativistic wave length
        # See http://en.wikipedia.org/wiki/Electron_diffraction
        # lambda = h/sqrt(2*m*e) * 1/sqrt(V*(1+V*e/(2*m*c^2)))
        # h/sqrt(2*m*e) = 12.2642598 * 10^-10 meters -> 12.2642598 Angstrom
        # e/(2*m*c^2)   = 9.78475598 * 10^-7 coulombs/joules
        lam = 12.2642598 / np.sqrt(voltage * (1. + voltage * 9.78475598e-7))

        # Some constants
        self.c1 = -np.pi * lam
        self.c2 = np.pi / 2. * spherical_aberration * lam ** 3
        self.c3 = phase_shift * np.pi / 180.
        self.c4 = -b_factor/4.
        self.c5 = \
            np.arctan(
                amplitude_contrast / np.sqrt(1-amplitude_contrast**2)
            )

        self.xx = {}
        self.yy = {}
        self.xy = {}
        self.n4 = {}

        self.device = torch.device("cpu")

    def __call__(
            self,
            grid_size: int,
            pixel_size: float,
            u: Tensor,
            v: Tensor,
            angle: Tensor,
            h_sym: bool = False,
            antialiasing: int = 0
    ) -> Tensor:
        """
        Get the CTF in an numpy array, the size of freq_x or freq_y.
        Generates a Numpy array or a Torch tensor depending on the object type
        on freq_x and freq_y passed to the constructor.
        :param u: the U defocus
        :param v: the V defocus
        :param angle: the azimuthal angle defocus (degrees)
        :param antialiasing: Antialiasing oversampling factor (0 = no antialiasing)
        :param grid_size: the side of the box
        :param pixel_size: pixel size
        :param h_sym: Only consider the hermitian half
        :return: Numpy array or Torch tensor containing the CTF
        """

        # Use cache
        tag = f"{grid_size}_{round(pixel_size, 3)}_{h_sym}_{antialiasing}"
        if tag not in self.xx:
            freq_x, freq_y = self._get_freq(grid_size, pixel_size, h_sym, antialiasing)
            xx = freq_x**2
            yy = freq_y**2
            xy = freq_x * freq_y
            n4 = (xx + yy)**2  # Norms squared^2
            self.xx[tag] = xx.to(self.device)
            self.yy[tag] = yy.to(self.device)
            self.xy[tag] = xy.to(self.device)
            self.n4[tag] = n4.to(self.device)

        xx = self.xx[tag]
        yy = self.yy[tag]
        xy = self.xy[tag]
        n4 = self.n4[tag]

        angle = angle * np.pi / 180
        acos = torch.cos(angle)
        asin = torch.sin(angle)
        acos2 = torch.square(acos)
        asin2 = torch.square(asin)

        """
        Out line of math for following three lines of code
        Q = [[sin cos] [-sin cos]] sin/cos of the angle
        D = [[u 0] [0 v]]
        A = Q^T.D.Q = [[Axx Axy] [Ayx Ayy]]
        Axx = cos^2 * u + sin^2 * v
        Ayy = sin^2 * u + cos^2 * v
        Axy = Ayx = cos * sin * (u - v)
        defocus = A.k.k^2 = Axx*x^2 + 2*Axy*x*y + Ayy*y^2
        """

        xx_ = (acos2 * u + asin2 * v)[:, None, None] * xx[None, :, :]
        yy_ = (asin2 * u + acos2 * v)[:, None, None] * yy[None, :, :]
        xy_ = (acos * asin * (u - v))[:, None, None] * xy[None, :, :]

        gamma = self.c1 * (xx_ + 2. * xy_ + yy_) + self.c2 * n4[None, :, :] - self.c3 - self.c5
        ctf = -torch.sin(gamma)
        if self.c4 > 0:
            ctf *= torch.exp(self.c4 * n4)

        if antialiasing > 0:
            o = 2**antialiasing
            ctf = ctf.unsqueeze(1)  # Add singleton channel
            ctf = torch.nn.functional.avg_pool2d(ctf, kernel_size=o+o//2, stride=o)
            ctf = ctf.squeeze(1)  # Remove singleton channel

        return ctf

    def to(self, device):
        if self.device == device:
            return
        self.device = device
        for tag in self.xx:
            self.xx[tag] = self.xx[tag].to(device)
            self.yy[tag] = self.yy[tag].to(device)
            self.xy[tag] = self.xy[tag].to(device)
            self.n4[tag] = self.n4[tag].to(device)

    @staticmethod
    def _get_freq(
            grid_size: int,
            pixel_size: float,
            h_sym: bool = False,
            antialiasing: int = 0
    ) -> Union[
            Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]],
            Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]
         ]:
        """
        Get the inverted frequencies of the Fourier transform of a square or cuboid grid.
        Can generate both Torch tensors and Numpy arrays.
        TODO Add 3D
        :param antialiasing: Antialiasing oversampling factor (0 = no antialiasing)
        :param grid_size: the side of the box
        :param pixel_size: pixel size
        :param h_sym: Only consider the hermitian half
        :return: two or three numpy arrays or tensors,
                 containing frequencies along the different axes
        """
        if antialiasing > 0:
            o = 2**antialiasing
            grid_size *= o
            y_ls = np.linspace(
                -(grid_size + o) // 2,
                (grid_size - o) // 2,
                grid_size + o//2
            )
            x_ls = y_ls if not h_sym else torch.linspace(0, grid_size // 2, grid_size // 2 + o + 1)
        else:
            y_ls = np.linspace(-grid_size // 2, grid_size // 2 - 1, grid_size)
            x_ls = y_ls if not h_sym else torch.linspace(0, grid_size // 2, grid_size // 2 + 1)

        y, x = torch.meshgrid(torch.Tensor(y_ls), torch.Tensor(x_ls))
        freq_x = x / (grid_size * pixel_size)
        freq_y = y / (grid_size * pixel_size)

        return freq_x, freq_y

    def get_state_dict(self) -> Dict:
        return {
            "type": "ContrastTransferFunction",
            "version": "0.0.1",
            "voltage": self.voltage,
            "spherical_aberration": self.spherical_aberration,
            "amplitude_contrast": self.amplitude_contrast,
            "phase_shift": self.phase_shift,
            "b_factor": self.b_factor
        }

    @staticmethod
    def load_from_state_dict(state_dict):
        if "type" not in state_dict or state_dict["type"] != "ContrastTransferFunction":
            raise TypeError("Input is not an 'ContrastTransferFunction' instance.")

        if "version" not in state_dict:
            raise RuntimeError("ContrastTransferFunction instance lacks version information.")

        if state_dict["version"] == "0.0.1":
            return ContrastTransferFunction(
                voltage=state_dict['voltage'],
                spherical_aberration=state_dict['spherical_aberration'],
                amplitude_contrast=state_dict['amplitude_contrast'],
                phase_shift=state_dict['phase_shift'],
                b_factor=state_dict['b_factor'],
            )
        else:
            raise RuntimeError(f"Version '{state_dict['version']}' not supported.")


if __name__ == "__main__":
    os1 = 0
    os2 = 2
    box = 200
    df = torch.Tensor([[20000]])
    pixA = 1.

    freq1_x, freq1_y = ContrastTransferFunction.get_freq(box, pixA, antialiasing=os1)
    ctf1 = ContrastTransferFunction(freq1_x, freq1_y, 300, 2.7, 0.1, 0, 0, os1)

    freq2_x, freq2_y = ContrastTransferFunction.get_freq(box, pixA, antialiasing=os2)
    ctf2 = ContrastTransferFunction(freq2_x, freq2_y, 300, 2.7, 0.1, 0, 0, os2)

    test1 = ctf1(df, df, torch.zeros([1, 1])).cpu().numpy()
    test2 = ctf2(df, df, torch.zeros([1, 1])).cpu().numpy()
    diff = test1 - test2
    print("%.10f" % np.mean(np.abs(diff)))

    import matplotlib.pylab as plt
    _, [ax1, ax2, ax3] = plt.subplots(1, 3)
    ax1.imshow(test1)
    ax2.imshow(test2)
    ax3.imshow(diff)
    plt.show()
