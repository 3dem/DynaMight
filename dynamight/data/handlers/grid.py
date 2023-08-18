#!/usr/bin/env python

"""
Module for calculations related to grid manipulations.
This is temporary, functions should be organized in separate files.
"""

import numpy as np
import mrcfile as mrc
import torch
from typing import Tuple, Union, TypeVar, List
from scipy import interpolate

import matplotlib.pylab as plt

Tensor = TypeVar('torch.tensor')


def grid_iterator(
        s0: Union[int, np.ndarray],
        s1: Union[int, np.ndarray],
        s2: Union[int, np.ndarray] = None
):
    if isinstance(s0, int):
        if s2 is None:  # 2D grid
            for i in range(s0):
                for j in range(s1):
                    yield i, j
        else:  # 3D grid
            for i in range(s0):
                for j in range(s1):
                    for k in range(s2):
                        yield i, j, k
    else:
        if s2 is None:  # 2D grid
            for i in range(len(s0)):
                for j in range(len(s1)):
                    yield s0[i], s1[j]
        else:  # 3D grid
            for i in range(len(s0)):
                for j in range(len(s1)):
                    for k in range(len(s2)):
                        yield s0[i], s1[j], s2[k]


def save_mrc(grid, voxel_size, origin, filename):
    (z, y, x) = grid.shape
    o = mrc.new(filename, overwrite=True)
    o.header['cella'].x = x * voxel_size
    o.header['cella'].y = y * voxel_size
    o.header['cella'].z = z * voxel_size
    o.header['origin'].x = origin[0]
    o.header['origin'].y = origin[1]
    o.header['origin'].z = origin[2]
    out_box = np.reshape(grid, (z, y, x))
    o.set_data(out_box.astype(np.float32))
    o.update_header_stats()
    o.flush()
    o.close()


def load_mrc(mrc_fn):
    mrc_file = mrc.open(mrc_fn, 'r')
    c = mrc_file.header['mapc']
    r = mrc_file.header['mapr']
    s = mrc_file.header['maps']

    global_origin = mrc_file.header['origin']
    global_origin = np.array([global_origin.x, global_origin.y, global_origin.z])
    global_origin[0] += mrc_file.header['nxstart']
    global_origin[1] += mrc_file.header['nystart']
    global_origin[2] += mrc_file.header['nzstart']

    global_origin *= mrc_file.voxel_size.x

    if c == 1 and r == 2 and s == 3:
        grid = mrc_file.data
    elif c == 3 and r == 2 and s == 1:
        grid = np.moveaxis(mrc_file.data, [0, 1, 2], [2, 1, 0])
    elif c == 2 and r == 1 and s == 3:
        grid = np.moveaxis(mrc_file.data, [1, 2, 0], [2, 1, 0])
    else:
        raise RuntimeError("MRC file axis arrangement not supported!")

    return grid, float(mrc_file.voxel_size.x), global_origin


def get_spectral_indices(shape: Union[Tuple[int, int], Tuple[int, int, int]], centered=True):
    h_sym = shape[0] == shape[-1]  # Hermitian symmetric half included
    dim_2 = len(shape) == 2

    if shape[0] % 2 == 0:
        ls = np.linspace(-(shape[0] // 2), shape[0] // 2 - 1, shape[0])
    else:
        ls = np.linspace(-(shape[0] // 2), shape[0] // 2, shape[0])
    x_ls = ls if h_sym else np.linspace(0, shape[1] - 1, shape[1])

    if dim_2:
        assert shape[1] == shape[0] or \
               (shape[1] - 1) * 2 == shape[0]
        y, x = np.meshgrid(ls, x_ls, indexing="ij")
        indices = np.round(np.sqrt(x ** 2 + y ** 2)).astype(int)
    else:
        assert shape[2] == shape[1] == shape[0] or \
               (shape[2] - 1) * 2 == shape[1] == shape[0]
        z, y, x = np.meshgrid(ls, ls, x_ls, indexing="ij")
        indices = np.round(np.sqrt(x ** 2 + y ** 2 + z ** 2)).astype(int)

    if not centered:
        if h_sym:
            indices = np.fft.ifftshift(indices)
        else:
            if dim_2:
                indices = np.fft.ifftshift(indices, axes=0)
            else:
                indices = np.fft.ifftshift(indices, axes=(0, 1))
    return indices


def get_spectral_sum(grid, max_index=None, spectral_indices=None):
    if spectral_indices is None:
        spectral_indices = get_spectral_indices(grid.shape)

    sz = np.max(spectral_indices) if max_index is None else max_index

    sums = np.zeros(sz)
    counts = np.zeros(sz)

    for r in range(sz):
        i = r == spectral_indices
        sums[r] = np.sum(grid[i])
        counts[r] = np.sum(i)

    return sums, counts


def get_spectral_avg(grid, max_index=None, spectral_indices=None):
    sums, counts = get_spectral_sum(grid, max_index, spectral_indices)
    return sums / counts


def spectrum_to_grid(spectrum, spectral_indices):
    return np.interp(spectral_indices, np.arange(len(spectrum)), spectrum)


def get_fourier_shell_resolution(ft_size, voxel_size):
    res = np.zeros(ft_size)
    res[1:] = np.arange(1, ft_size) / (2 * voxel_size * ft_size)
    return res


def get_spectral_index_from_resolution(resolution: float, image_size: int, pixel_size: float):
    return round(image_size * pixel_size / resolution)


def get_resolution_from_spectral_index(index: int, image_size: int, pixel_size: float):
    return pixel_size * image_size / float(index)


def make_power_spectra(ft_size, voxel_size, bfac):
    r = get_fourier_shell_resolution(ft_size, voxel_size)
    return np.exp(-bfac * r * r), r


def make_cubic(box):
    bz = np.array(box.shape)
    s = np.max(box.shape)
    s += s % 2
    if np.all(box.shape == s):
        return box, np.zeros(3, dtype=int), bz
    nbox = np.zeros((s, s, s))
    c = np.array(nbox.shape) // 2 - bz // 2
    nbox[c[0]:c[0] + bz[0], c[1]:c[1] + bz[1], c[2]:c[2] + bz[2]] = box
    return nbox, c, c + bz


def resize_grid(
        grid: np.ndarray,
        shape: Union[List, Tuple],
        pad_value: float = 0
) -> np.ndarray:
    """
    Resize grid to shape, by cropping larger and padding smaller dimensions.
    :param grid: The gird to resize
    :param shape: New shape
    :param pad_value: Value to pad smaller dimensions (default=0)
    :return:
    """
    shape = np.array(shape)
    new_shape = np.max([shape, np.array(grid.shape)], axis=0)
    b = np.ones(new_shape) * pad_value
    c = np.array(b.shape) // 2 - np.array(grid.shape) // 2
    assert np.sum(c < 0) == 0
    s = grid.shape
    b[c[0]:c[0] + s[0], c[1]:c[1] + s[1], c[2]:c[2] + s[2]] = grid
    if not np.all(np.equal(b.shape, shape)):
        c_ = np.array(b.shape) // 2 - np.array(shape) // 2
        assert np.sum(c < 0) == 0
        s = shape
        b = b[c_[0]:c_[0] + s[0], c_[1]:c_[1] + s[1], c_[2]:c_[2] + s[2]]
        c -= c_

    return b, c


def rescale_fourier(box, out_sz):
    if out_sz % 2 != 0:
        raise Exception("Bad output size")
    if box.shape[0] != box.shape[1] or \
            box.shape[1] != (box.shape[2] - 1) * 2:
        raise Exception("Input must be cubic")

    ibox = np.fft.ifftshift(box, axes=(0, 1))
    obox = np.zeros((out_sz, out_sz, out_sz // 2 + 1), dtype=box.dtype)

    si = np.array(ibox.shape) // 2
    so = np.array(obox.shape) // 2

    if so[0] < si[0]:
        obox = ibox[
               si[0] - so[0]: si[0] + so[0],
               si[1] - so[1]: si[1] + so[1],
               :obox.shape[2]
               ]
    elif so[0] > si[0]:
        obox[
        so[0] - si[0]: so[0] + si[0],
        so[1] - si[1]: so[1] + si[1],
        :ibox.shape[2]
        ] = ibox
    else:
        obox = ibox

    obox = np.fft.ifftshift(obox, axes=(0, 1))

    return obox


def rescale_real(box, out_sz):
    if out_sz != box.shape[0]:
        f = np.fft.rfftn(box)
        f = rescale_fourier(f, out_sz)
        box = np.fft.irfftn(f)

    return box


def rescale_voxel_size(density: np.ndarray, voxel_size: float, new_voxel_size: float) -> np.ndarray:
    (iz, iy, ix) = np.shape(density)

    assert iz % 2 == 0 and iy % 2 == 0 and ix % 2 == 0
    assert ix == iy == iz

    in_sz = ix
    out_sz = int(round(in_sz * voxel_size / new_voxel_size))
    if out_sz % 2 != 0:
        vs1 = voxel_size * in_sz / (out_sz + 1)
        vs2 = voxel_size * in_sz / (out_sz - 1)
        if np.abs(vs1 - new_voxel_size) < np.abs(vs2 - new_voxel_size):
            out_sz += 1
        else:
            out_sz -= 1

    out_voxel_sz = voxel_size * in_sz / out_sz
    density = rescale_real(density, out_sz)

    return density, out_voxel_sz


def get_bounds_for_threshold(density, threshold=0.):
    """Finds the bounding box encapsulating volume segment above threshold"""
    xy = np.all(density < threshold, axis=0)
    c = [[], [], []]
    c[0] = ~np.all(xy, axis=0)
    c[1] = ~np.all(xy, axis=1)
    c[2] = ~np.all(np.all(density <= threshold, axis=2), axis=1)

    h = np.zeros(3)
    l = np.zeros(3)
    (h[2], h[1], h[0]) = np.shape(density)

    for i in range(3):
        for j in range(len(c[i])):
            if c[i][j]:
                l[i] = j
                break
        for j in reversed(range(len(c[0]))):
            if c[i][j]:
                h[i] = j
                break

    return l.astype(int), h.astype(int)


def get_fsc_ft(map1_ft, map2_ft, voxel_size=0):
    assert np.any(map1_ft.shape == map2_ft.shape)
    (fiz, fiy, fix) = map1_ft.shape

    R = get_spectral_indices(map1_ft)

    fsc = np.zeros(fix)
    res = np.zeros(fix)

    if voxel_size > 0:
        for i in np.arange(1, len(fsc)):
            j = i == R
            s1 = map1_ft[j]
            s2 = map2_ft[j]
            norm = np.sqrt(np.sum(np.square(np.abs(s1))) * np.sum(np.square(np.abs(s2))))
            fsc[i] = np.real(np.sum(s1 * np.conj(s2))) / (norm + 1e-12)
            res[i] = fix * 2 * voxel_size / i
    else:
        for i in np.arange(1, len(fsc)):
            j = i == R
            s1 = map1_ft[j]
            s2 = map2_ft[j]
            norm = np.sqrt(np.sum(np.square(np.abs(s1))) * np.sum(np.square(np.abs(s2))))
            fsc[i] = np.real(np.sum(s1 * np.conj(s2))) / (norm + 1e-12)

    res[0] = 1e9
    fsc[0] = 1

    if voxel_size > 0:
        return res, fsc
    else:
        return fsc


def get_fsc_torch(F1, F2, ang_pix=1, visualize=False):
    if F1.is_complex():
        pass
    else:
        F1 = torch.fft.fftn(F1, dim=[-3, -2, -1])
    if F2.is_complex():
        pass
    else:
        F2 = torch.fft.fftn(F2, dim=[-3, -2, -1])

    if F1.shape != F2.shape:
        print('The volumes have to be the same size')

    N = F1.shape[-1]
    ind = torch.linspace(-(N - 1) / 2, (N - 1) / 2 - 1, N)
    end_ind = torch.round(torch.tensor(N / 2)).long()
    X, Y, Z = torch.meshgrid(ind, ind, ind, indexing = 'ij')
    R = torch.fft.fftshift(torch.round(torch.pow(X ** 2 + Y ** 2 + Z ** 2, 0.5)).long())

    if len(F1.shape) == 3:
        num = torch.zeros(torch.max(R) + 1)
        den1 = torch.zeros(torch.max(R) + 1)
        den2 = torch.zeros(torch.max(R) + 1)
        num.scatter_add_(0, R.flatten(), torch.real(F1 * torch.conj(F2)).flatten())
        den = torch.pow(
            den1.scatter_add_(0, R.flatten(), torch.abs(F1.flatten(start_dim=-3)) ** 2) *
            den2.scatter_add_(0, R.flatten(), torch.abs(F2.flatten(start_dim=-3)) ** 2),
            0.5
        )
        FSC = num / den

    res = N * ang_pix / torch.arange(end_ind)

    if visualize:
        plt.figure(figsize=(10, 10))
        plt.rcParams['axes.xmargin'] = 0
        plt.plot(FSC[:end_ind], c='r')
        plt.plot(torch.ones(end_ind) * 0.5, c='black', linestyle='dashed')
        plt.plot(torch.ones(end_ind) * 0.143, c='slategrey', linestyle='dotted')
        plt.xticks(torch.arange(start=0, end=end_ind, step=4),
                   labels=np.round(res[torch.arange(start=0, end=end_ind, step=4)].numpy(), 1))

    return FSC[0:end_ind], res


def get_fsc(map1, map2, voxel_size=0):
    assert np.any(map1.shape == map2.shape)

    map1_ft = np.fft.rfftn(map1)
    map2_ft = np.fft.rfftn(map2)

    return get_fsc_ft(map1_ft, map2_ft, voxel_size)


def res_from_fsc(fsc, res, threshold=0.5):
    """
    Calculates the resolution (res) at the FSC (fsc) threshold.
    """
    assert len(fsc) == len(res)
    i = np.argmax(fsc < 0.5)
    if i > 0:
        return res[i - 1]
    else:
        res[0]


def _dt_set_axes(shape, dim):
    if dim is None:
        return tuple((np.arange(len(shape)).astype(int)))

    if len(shape) > dim:
        return tuple((np.arange(dim).astype(int)) + 1)
    else:
        return tuple((np.arange(dim).astype(int)))


def dft(
        grid: Union[Tensor, np.ndarray],
        dim: int = None,
        center: bool = True,
        real_in: bool = False
) -> Union[Tensor, np.ndarray]:
    """
    Discreet Fourier transform
    :param grid: Numpy array or Pytorch tensor to be transformed, can be stack
    :param dim: If stacked grids, specify the dimensionality
    :param center: If the zeroth frequency should be centered
    :param real_in: Input is real. Only returns hermitian half
    :return: Transformed Numpy array or Pytorch tensor
    """
    use_torch = torch.is_tensor(grid)
    axes = _dt_set_axes(grid.shape, dim)

    if real_in:
        grid_ft = torch.fft.rfftn(torch.fft.fftshift(grid, dim=axes), dim=axes) if use_torch \
            else np.fft.rfftn(np.fft.fftshift(grid, axes=axes), axes=axes)
    else:
        grid_ft = torch.fft.fftn(torch.fft.fftshift(grid, dim=axes), dim=axes) if use_torch \
            else np.fft.fftn(np.fft.fftshift(grid, axes=axes), axes=axes)

    if center:
        grid_ft = torch.fft.fftshift(grid_ft, dim=axes[:-1] if real_in else axes) if use_torch \
            else np.fft.fftshift(grid_ft, axes=axes[:-1] if real_in else axes)

    return grid_ft


def idft(
        grid_ft: Union[Tensor, np.ndarray],
        dim: int = None,
        centered: bool = True,
        real_in: bool = False
) -> Union[Tensor, np.ndarray]:
    """
    Inverse Discreet Fourier transform
    :param grid_ft: Numpy array or Pytorch tensor to be transformed, can be stack
    :param dim: If stacked grids, specify the dimensionality
    :param centered: If the zeroth frequency should be centered
    :param real_in: Input is real. Only returns hermitian half
    :return: Inverse transformed Numpy array or Pytorch tensor
    """
    use_torch = torch.is_tensor(grid_ft)
    axes = _dt_set_axes(grid_ft.shape, dim)

    if centered:
        grid_ft = torch.fft.ifftshift(grid_ft, dim=axes[:-1] if real_in else axes) if use_torch \
            else np.fft.ifftshift(grid_ft, axes=axes[:-1] if real_in else axes)

    if real_in:
        grid = torch.fft.ifftshift(torch.fft.irfftn(grid_ft, dim=axes), dim=axes) if use_torch \
            else np.fft.ifftshift(np.fft.irfftn(grid_ft, axes=axes), axes=axes)
    else:
        grid = torch.fft.ifftshift(torch.fft.ifftn(grid_ft, dim=axes), dim=axes) if use_torch \
            else np.fft.ifftshift(np.fft.ifftn(grid_ft, axes=axes), axes=axes)

    return grid


def dht(
        grid: Union[Tensor, np.ndarray],
        dim: int = None,
        center: bool = True
) -> Union[Tensor, np.ndarray]:
    """
    Discreet Hartley transform
    :param grid: Numpy array or Pytorch tensor to be transformed, can be stack
    :param dim: If stacked grids, specify the dimensionality
    :param center: If the zeroth frequency should be centered
    :return: Transformed Numpy array or Pytorch tensor
    """
    use_torch = torch.is_tensor(grid)
    axes = None if dim is None else _dt_set_axes(grid.shape, dim)

    grid_ht = torch.fft.fftn(torch.fft.fftshift(grid, dim=axes), dim=axes) if use_torch \
        else np.fft.fftn(np.fft.fftshift(grid, axes=axes), axes=axes)

    if center:
        grid_ht = torch.fft.fftshift(grid_ht, dim=axes) if use_torch \
            else np.fft.fftshift(grid_ht, axes=axes)

    return grid_ht.real - grid_ht.imag


def idht(
        grid_ht: Union[Tensor, np.ndarray],
        dim: int = None,
        centered: bool = True
) -> Union[Tensor, np.ndarray]:
    """
    Inverse Discreet Hartley transform
    :param grid_ht: Numpy array or Pytorch tensor to be transformed, can be stack
    :param dim: If stacked grids, specify the dimensionality
    :param centered: If the zeroth frequency should be centered
    :return: Inverse transformed Numpy array or Pytorch tensor
    """
    use_torch = torch.is_tensor(grid_ht)
    axes = None if dim is None else _dt_set_axes(grid_ht.shape, dim)

    if centered:
        grid_ht = torch.fft.ifftshift(grid_ht, dim=axes) if use_torch \
            else np.fft.ifftshift(grid_ht, axes=axes)

    f = torch.fft.fftshift(torch.fft.fftn(grid_ht, dim=axes), dim=axes) if use_torch \
        else np.fft.fftshift(np.fft.fftn(grid_ht, axes=axes), axes=axes)

    # Adjust for FFT normalization
    if axes is None:
        f /= np.product(f.shape)
    else:
        f /= np.product(np.array(f.shape)[list(axes)])

    return f.real - f.imag


def htToFt(
        grid_ht: Union[Tensor, np.ndarray],
        dim: int = None
):
    """
    Converts a batch of Hartley transforms to Fourier transforms
    :param grid_ht: Batch of Hartley transforms
    :param dim: Data dimension
    :return: The batch of Fourier transforms
    """
    axes = tuple(np.arange(len(grid_ht.shape))) if dim is None else _dt_set_axes(grid_ht.shape, dim)
    dtype = get_complex_float_type(grid_ht.dtype)

    if torch.is_tensor(grid_ht):
        grid_ft = torch.empty(grid_ht.shape, dtype=dtype).to(grid_ht.device)
        grid_ht_ = torch.flip(grid_ht, axes)
        if grid_ht.shape[-1] % 2 == 0:
            grid_ht_ = torch.roll(grid_ht_, [1] * len(axes), axes)
        grid_ft.real = (grid_ht + grid_ht_) / 2
        grid_ft.imag = (grid_ht - grid_ht_) / 2
    else:
        grid_ft = np.empty(grid_ht.shape, dtype=dtype)
        grid_ht_ = np.flip(grid_ht, axes)
        if grid_ht.shape[-1] % 2 == 0:
            grid_ht_ = np.roll(grid_ht_, 1, axes)
        grid_ft.real = (grid_ht + grid_ht_) / 2
        grid_ft.imag = (grid_ht - grid_ht_) / 2

    return grid_ft


def dt_symmetrize(dt: Tensor, dim: int = None) -> Tensor:
    s = dt.shape
    if dim is None:
        dim = 3 if len(s) >= 3 else 2

    if s[-2] % 2 != 0:
        raise RuntimeError("Box size must be even.")

    if dim == 2:
        if s[-1] == s[-2]:
            if len(s) == 2:
                sym_ht = torch.empty((s[0] + 1, s[1] + 1), dtype=dt.dtype).to(dt.device)
            else:
                sym_ht = torch.empty((s[0], s[1] + 1, s[2] + 1), dtype=dt.dtype).to(dt.device)
            sym_ht[..., 0:-1, 0:-1] = dt
            sym_ht[..., -1, :-1] = dt[..., 0, :]
            sym_ht[..., :, -1] = sym_ht[..., :, 0]
        if s[-1] == s[-2] // 2 + 1:
            if len(s) == 2:
                sym_ht = torch.empty((s[0] + 1, s[1]), dtype=dt.dtype).to(dt.device)
            else:
                sym_ht = torch.empty((s[0], s[1] + 1, s[2]), dtype=dt.dtype).to(dt.device)
            sym_ht[..., 0:-1, :] = dt
            sym_ht[..., -1, :] = dt[..., 0, :]

    elif dim == 3:
        if len(s) == 3:
            sym_ht = torch.empty((s[0] + 1, s[1] + 1, s[2] + 1), dtype=dt.dtype).to(dt.device)
        else:
            sym_ht = torch.empty((s[0], s[1] + 1, s[2] + 1, s[3] + 1), dtype=dt.dtype).to(dt.device)
        sym_ht[..., 0:-1, 0:-1, 0:-1] = dt
        sym_ht[..., -1, :-1, :-1] = sym_ht[..., 0, :-1, :-1]
        sym_ht[..., :, :-1, -1] = sym_ht[..., :, :-1, 0]
        sym_ht[..., :, -1, :] = sym_ht[..., :, 0, :]
    else:
        raise RuntimeError("Dimensionality not supported")
    return sym_ht


def dt_desymmetrize(dt: Tensor, dim: int = None) -> Tensor:
    s = dt.shape
    if dim is None:
        dim = 3 if len(s) >= 3 else 2
    if dim == 2:
        if s[-2] == s[-1] * 2 - 1:
            out = dt[..., :-1, :]
            out[..., 0, :] = (dt[..., 0, :] + dt[..., -1, :]) / 2.
        else:
            out = dt[..., :-1, :-1]
            out[..., 0, :] = (dt[..., 0, :-1] + dt[..., -1, :-1]) / 2.
            out[..., :, 0] = (dt[..., :-1, 0] + dt[..., :-1, -1]) / 2.
    elif dim == 3:
        if s[-2] == s[-1] * 2 - 1:
            out = dt[..., :-1, :-1, :]
            out[..., 0, :, :] = (dt[..., 0, :-1, :] + dt[..., -1, :-1, :]) / 2.
            out[..., :, 0, :] = (dt[..., :-1, 0, :] + dt[..., :-1, -1, :]) / 2.
        else:
            out = dt[..., :-1, :-1, :-1]
            out[..., 0, :, :] = (dt[..., 0, :-1, :-1] + dt[..., -1, :-1, :-1]) / 2.
            out[..., :, 0, :] = (dt[..., :-1, 0, :-1] + dt[..., :-1, -1, :-1]) / 2.
            out[..., :, :, 0] = (dt[..., :-1, :-1, 0] + dt[..., :-1, :-1, -1]) / 2.
    else:
        raise RuntimeError("Dimensionality not supported")

    return out


def rdht(
        grid: Union[Tensor, np.ndarray],
        dim: int = None,
        center: bool = True
) -> Union[Tensor, np.ndarray]:
    """
    Discreet Hartley transform
    :param grid: Numpy array or Pytorch tensor to be transformed, can be stack
    :param dim: If stacked grids, specify the dimensionality
    :param center: If the zeroth frequency should be centered
    :return: Transformed Numpy array or Pytorch tensor
    """
    use_torch = torch.is_tensor(grid)
    axes = tuple(np.arange(len(grid.shape))) if dim is None else _dt_set_axes(grid.shape, dim)

    if use_torch:
        grid_ft = torch.fft.rfftn(torch.fft.fftshift(grid, dim=axes), dim=axes)

        grid_ht = torch.empty_like(grid)
        grid_ht[..., :grid_ft.shape[-1]] = grid_ft.real - grid_ft.imag
        hh = torch.flip(grid_ft.real[..., 1:-1] + grid_ft.imag[..., 1:-1], axes)

        hh = torch.roll(hh, [1] * (len(axes) - 1), axes[:-1])
        grid_ht[..., grid_ft.shape[-1]:] = hh
    else:
        grid_ft = np.fft.rfftn(np.fft.fftshift(grid, axes=axes), axes=axes)

        grid_ht = np.empty(grid.shape)
        grid_ht[..., :grid_ft.shape[-1]] = grid_ft.real - grid_ft.imag
        hh = np.flip(grid_ft.real[..., 1:-1] + grid_ft.imag[..., 1:-1], axes)

        hh = np.roll(hh, 1, axis=axes[:-1])
        grid_ht[..., grid_ft.shape[-1]:] = hh

    if center:
        grid_ht = torch.fft.fftshift(grid_ht, dim=axes) if use_torch \
            else np.fft.fftshift(grid_ht, axes=axes)

    return grid_ht


def ird3ht(
        grid_ht: Tensor
) -> Tensor:
    """
    Inverse Discreet Hartley transform, carried out by doing a conversion
    to a Fourier transform and using rfft. Assumes centered!!!
    :param grid_ht: Pytorch tensor with batch of 3D grids to be transformed
    :return: Inverse transformed Pytorch tensor
    """
    s = grid_ht.shape[-1]
    assert len(grid_ht.shape) == 4
    assert s == grid_ht.shape[-3] and s == grid_ht.shape[-2]
    assert s % 2 == 1

    axes = (1, 2, 3)
    grid_ft = htToFt(grid_ht, dim=3)
    grid_ft = grid_ft[..., s // 2:]
    grid_ft = torch.fft.ifftshift(grid_ft, dim=axes)
    grid = torch.fft.ifftshift(torch.fft.irfftn(grid_ft, dim=axes), dim=axes)

    return grid


def smooth_circular_mask(image_size, radius, thickness):
    y, x = np.meshgrid(
        np.linspace(-image_size // 2, image_size // 2 - 1, image_size),
        np.linspace(-image_size // 2, image_size // 2 - 1, image_size)
    )
    r = np.sqrt(x ** 2 + y ** 2)
    band_mask = (radius <= r) & (r <= radius + thickness)
    r_band_mask = r[band_mask]
    mask = np.zeros((image_size, image_size))
    mask[r < radius] = 1
    mask[band_mask] = np.cos(np.pi * (r_band_mask - radius) / thickness) / 2 + .5
    mask[radius + thickness < r] = 0
    return mask


def smooth_square_mask(image_size, square_side, thickness):
    square_side_2 = square_side / 2.
    y, x = np.meshgrid(
        np.linspace(-image_size // 2, image_size // 2 - 1, image_size),
        np.linspace(-image_size // 2, image_size // 2 - 1, image_size)
    )
    p = np.max([np.abs(x), np.abs(y)], axis=0)
    band_mask = (square_side_2 <= p) & (p <= square_side_2 + thickness)
    p_band_mask = p[band_mask]
    mask = np.zeros((image_size, image_size))
    mask[p < square_side_2] = 1
    mask[band_mask] = np.cos(np.pi * (p_band_mask - square_side_2) / thickness) / 2 + .5
    mask[square_side_2 + thickness < p] = 0
    return mask


def get_complex_float_type(type):
    if type == np.float16:
        return np.complex32
    elif type == np.float32:
        return np.complex64
    elif type == np.float64:
        return np.complex128
    elif type == torch.float16:
        return torch.complex32
    elif type == torch.float32:
        return torch.complex64
    elif type == torch.float64:
        return torch.complex128
    else:
        raise RuntimeError("Unknown float type")


def fourier_shift_2d(
        grid_ft: Union[Tensor, np.ndarray],
        shift: Union[Tensor, np.ndarray],
        y_shift: Union[Tensor, np.ndarray] = None
) -> Union[Tensor, np.ndarray]:
    """
    Shifts a batch of 2D Fourier transformed images
    :param grid_ft: Batch of Fourier transformed images [B, Y, X]
    :param shift: Either array of size [B, 2] (X, Y) or [B, 1] (X)
    :param y_shift: If 'None' assumes 'shift' contains both X and Y shifts
    :return: The shifted 2D Fourier transformed images
    """
    complex_channels = len(grid_ft.shape) == 4 and grid_ft.shape[-1] == 2
    assert len(grid_ft.shape) == 3 or complex_channels
    assert shift.shape[0] == grid_ft.shape[0]
    assert grid_ft.shape[1] == grid_ft.shape[2] * 2 - 1, "Only square images are supported"
    s = grid_ft.shape[1]
    symmetrized = s % 2 == 1
    if symmetrized:
        s -= 1

    if y_shift is None:
        assert len(shift.shape) == 2 and shift.shape[1] == 2
        x_shift = shift[..., 0]
        y_shift = shift[..., 1]
    else:
        assert len(shift.shape) == 1 and len(y_shift.shape) == 1 and \
               shift.shape[0] == y_shift.shape[0]
        x_shift = shift
        y_shift = y_shift

    x_shift = x_shift / float(s)
    y_shift = y_shift / float(s)

    if symmetrized:
        ls = torch.linspace(-s // 2, s // 2, s + 1)
    else:
        ls = torch.linspace(-s // 2, s // 2 - 1, s)
    lsx = torch.linspace(0, s // 2, s // 2 + 1)
    y, x = torch.meshgrid(ls, lsx, indexing='ij')
    x = x.to(grid_ft.device)
    y = y.to(grid_ft.device)
    dot_prod = 2 * np.pi * (x[None, :, :] * x_shift[:, None, None] + y[None, :, :] * y_shift[:, None, None])
    a = torch.cos(dot_prod)
    b = torch.sin(dot_prod)

    if complex_channels:
        ar = a * grid_ft[..., 0]
        bi = b * grid_ft[..., 1]
        ab_ri = (a + b) * (grid_ft[..., 0] + grid_ft[..., 1])
        r = ar - bi
        i = ab_ri - ar - bi
        return torch.cat([r.unsqueeze(-1), i.unsqueeze(-1)], -1)
    else:
        ar = a * grid_ft.real
        bi = b * grid_ft.imag
        ab_ri = (a + b) * (grid_ft.real + grid_ft.imag)

        return ar - bi + 1j * (ab_ri - ar - bi)


def bilinear_shift_2d(
        grid: Tensor,
        shift: Tensor,
        y_shift: Tensor = None
) -> Tensor:
    """
    Shifts a batch of 2D images
    :param grid: Batch of images [B, Y, X]
    :param shift: Either array of size [B, 2] (X, Y) or [B, 1] (X)
    :param y_shift: If 'None' assumes 'shift' contains both X and Y shifts
    :return: The shifted 2D images
    """
    if y_shift is not None:
        assert len(shift.shape) == 1 and len(y_shift.shape) == 1 and \
               shift.shape[0] == y_shift.shape[0]
        shift_ = torch.empty([shift.shape[0], 2])
        shift_[:, 0] = shift
        shift_[:, 1] = y_shift
        shift = shift_

    assert len(shift.shape) == 2 and shift.shape[1] == 2
    int_shift = torch.floor(shift).long()

    s0 = shift - int_shift
    s1 = 1 - s0

    int_shift = int_shift.detach().cpu().numpy()
    g00 = torch.empty_like(grid)
    for i in range(len(grid)):
        g00[i] = torch.roll(grid[i], tuple(int_shift[i]), (-1, -2))

    g01 = torch.roll(g00, (0, 1), (-1, -2))
    g10 = torch.roll(g00, (1, 0), (-1, -2))
    g11 = torch.roll(g00, (1, 1), (-1, -2))

    g = g00 * s1[:, 0, None, None] * s1[:, 1, None, None] + \
        g10 * s0[:, 0, None, None] * s1[:, 1, None, None] + \
        g01 * s1[:, 0, None, None] * s0[:, 1, None, None] + \
        g11 * s0[:, 0, None, None] * s0[:, 1, None, None]

    return g


def integer_shift_2d(
        grid: Tensor,
        shift: Tensor,
        y_shift: Tensor = None
) -> Tensor:
    """
    Shifts a batch of 2D images
    :param grid: Batch of images [B, Y, X]
    :param shift: Either array of size [B, 2] (X, Y) or [B, 1] (X)
    :param y_shift: If 'None' assumes 'shift' contains both X and Y shifts
    :return: The shifted 2D images
    """
    if y_shift is not None:
        assert len(shift.shape) == 1 and len(y_shift.shape) == 1 and \
               shift.shape[0] == y_shift.shape[0]
        shift_ = torch.empty([shift.shape[0], 2])
        shift_[:, 0] = shift
        shift_[:, 1] = y_shift
        shift = shift_
    assert len(shift.shape) == 2 and shift.shape[1] == 2

    shift = shift.long().detach().cpu().numpy()
    g = torch.empty_like(grid)
    for i in range(len(grid)):
        g[i] = torch.roll(grid[i], tuple(shift[i]), (-1, -2))

    return g


def grid_spectral_sum_torch(grid, indices):
    if len(grid.shape) == len(indices.shape) and np.all(grid.shape == indices.shape):  # Has no batch dimension
        spectrum = torch.zeros(int(torch.max(indices)) + 1).to(grid.device)
        spectrum.scatter_add_(0, indices.long().flatten(), grid.flatten())
    elif len(grid.shape) == len(indices.shape) + 1 and np.all(grid.shape[1:] == indices.shape):  # Has batch dimension
        spectrum = torch.zeros([grid.shape[0], int(torch.max(indices)) + 1]).to(grid.device)
        indices = indices.long().unsqueeze(0).expand([grid.shape[0]] + list(indices.shape))
        spectrum.scatter_add_(1, indices.flatten(1), grid.flatten(1))
    else:
        raise RuntimeError("Shape of grid must match spectral_indices, except along the batch dimension.")
    return spectrum


def grid_spectral_average_torch(grid, indices):
    indices = indices.long()
    spectrum = grid_spectral_sum_torch(grid, indices)
    norm = grid_spectral_sum_torch(torch.ones_like(indices).float(), indices)
    return spectrum / norm[None, :]


def spectra_to_grid_torch(spectra, indices):
    if len(spectra.shape) == 1:  # Has no batch dimension
        grid = torch.gather(spectra, 0, indices.flatten().long())
    elif len(spectra.shape) == 2:  # Has batch dimension
        indices = indices.unsqueeze(0).expand([spectra.shape[0]] + list(indices.shape))
        grid = torch.gather(spectra.flatten(1), 1, indices.flatten(1).long())
    else:
        raise RuntimeError("Spectra must be at most two-dimensional (one batch dimension).")
    return grid.view(indices.shape)


def spectral_correlation_torch(grid1, grid2, indices, normalize=False, norm_eps=1e-12):
    if grid1.shape != grid2.shape:
        print('The grids have to be the same shape')

    if not grid1.is_complex() or not grid2.is_complex():
        grid1 = torch.view_as_complex(grid1)
        grid2 = torch.view_as_complex(grid2)

    shape = grid1.shape
    device = grid1.device
    indices = indices.long()
    output_size = int(torch.max(indices)) + 1

    if len(shape) == len(indices.shape) and np.all(shape == indices.shape):  # Has no batch dimension
        numer = torch.zeros(output_size).to(device)
        numer.scatter_add_(0, indices.flatten(), torch.real(grid1 * torch.conj(grid2)).flatten())
        if normalize:
            norm1 = torch.zeros(output_size).to(device)
            norm2 = torch.zeros(output_size).to(device)
            norm = torch.sqrt(
                norm1.scatter_add_(0, indices.flatten(), torch.square(torch.abs(grid1)).flatten()) *
                norm2.scatter_add_(0, indices.flatten(), torch.square(torch.abs(grid2)).flatten())
            )
    elif len(shape) == len(indices.shape) + 1 and np.all(shape[1:] == indices.shape):  # Has batch dimension
        indices_ = indices.flatten().unsqueeze(0).expand([shape[0], indices.nelement()])
        numer = torch.zeros([shape[0], output_size]).to(device)
        numer.scatter_add_(1, indices_, torch.real(grid1 * torch.conj(grid2)).flatten(1))
        if normalize:
            norm1 = torch.zeros([shape[0], output_size]).to(device)
            norm2 = torch.zeros([shape[0], output_size]).to(device)
            norm = torch.sqrt(
                norm1.scatter_add_(1, indices_, torch.square(torch.abs(grid1)).flatten(1)) *
                norm2.scatter_add_(1, indices_, torch.square(torch.abs(grid2)).flatten(1))
            )
    else:
        raise RuntimeError("Shape of grids must match spectral_indices, except along the batch dimension.")

    if normalize:
        return numer / (norm + norm_eps)
    else:
        norm = torch.zeros(output_size).to(device)
        norm.scatter_add_(0, indices.flatten(), torch.ones_like(indices).float().flatten())
        return numer / (norm[None, :] + norm_eps)


if __name__ == "__main__":
    spectrum = torch.zeros([2, 9])
    spectrum[0, 0] = 1.
    spectrum[0, 1] = .9
    spectrum[0, 2] = .8
    spectrum[0, 3] = .6
    spectrum[0, 4] = .2
    spectrum[0, 5] = .8
    spectrum[0, 6] = .2
    spectrum[1, 6] = 1.
    spectrum[1, 2] = 3.

    import matplotlib.pylab as plt

    idx = torch.Tensor(get_spectral_indices((17, 17)))
    idx[idx > 8] = 8
    grid = spectra_to_grid_torch(spectrum, idx)
    spectrum_ = grid_spectral_average_torch(grid, idx)

    # _, [ax1, ax2] = plt.subplots(2, 1)
    # ax1.imshow(grid[0])
    # ax2.imshow(grid[1])
    # plt.show()

    grid2 = grid.clone()
    grid2[0, 8, :] = 2
    grid_ft = dft(grid, dim=2)
    grid2_ft = dft(grid2, dim=2)
    fsc = spectral_correlation_torch(grid_ft, grid2_ft, idx).numpy()
    print(fsc)
    # _, [ax1, ax2] = plt.subplots(2, 1)
    # ax1.plot(np.arange(len(fsc[0])), fsc[0])
    # ax2.plot(np.arange(len(fsc[1])), fsc[1])
    # plt.show()

    # idx = torch.Tensor(get_spectral_indices((17, 17, 17)))
    # idx[idx > 9] = 9
    # grid = spectra_to_grid_torch(spectrum, idx[..., 8:])
    #
    # _, [ax1, ax2] = plt.subplots(2, 1)
    # ax1.imshow(grid[0, 9])
    # ax2.imshow(grid[1, 9])
    # plt.show()
