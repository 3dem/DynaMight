#!/usr/bin/env python

"""
Module for calculations related to rotation matrices and euler angles as defined in RELION.
"""
import sys
import numpy as np
import torch

from typing import Tuple, Union, TypeVar

Tensor = TypeVar('torch.tensor')


def eulerToMatrix(
        angles: Union[Tensor, np.ndarray]
) -> Union[Tensor, np.ndarray]:
    """
    Takes a batch of the three Euler angles as defined in RELION and
    returns a batch of the corresponding rotation matrices

    Supports both numpy arrays and torch tensor input

    :param angles: an array (B, 3) of the Euler angels, alpha, beta and gamma (rot, tilt, psi)
    :return: a 3x3 rotation matrix
    """
    if torch.is_tensor(angles):
        R = torch.zeros(len(angles), 3, 3, dtype=angles.dtype).to(angles.device)
        ca = torch.cos(angles[:, 0])
        cb = torch.cos(angles[:, 1])
        cg = torch.cos(angles[:, 2])
        sa = torch.sin(angles[:, 0])
        sb = torch.sin(angles[:, 1])
        sg = torch.sin(angles[:, 2])
    else:
        R = np.zeros((len(angles), 3, 3), dtype=angles.dtype)
        ca = np.cos(angles[:, 0])
        cb = np.cos(angles[:, 1])
        cg = np.cos(angles[:, 2])
        sa = np.sin(angles[:, 0])
        sb = np.sin(angles[:, 1])
        sg = np.sin(angles[:, 2])

    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa

    R[:, 0, 0] = cg * cc - sg * sa
    R[:, 0, 1] = cg * cs + sg * ca
    R[:, 0, 2] = -cg * sb
    R[:, 1, 0] = -sg * cc - cg * sa
    R[:, 1, 1] = -sg * cs + cg * ca
    R[:, 1, 2] = sg * sb
    R[:, 2, 0] = sc
    R[:, 2, 1] = ss
    R[:, 2, 2] = cb

    return R


def matrixToEuler(R: np.ndarray):
    """
    Takes a rotation matrix and returns the
    three Euler angles as defined in RELION

    TODO: Add support for batches

    :param R: a 3x3 rotation matrix
    :return: the three Euler angles, alpha, beta and gamma (rot, tilt, psi)
    """
    abs_sb = np.sqrt(R[0, 2] * R[0, 2] + R[1, 2] * R[1, 2])
    if abs_sb > 16*sys.float_info.epsilon:
        gamma = np.atan2(R[1, 2], -R[0, 2])
        alpha = np.atan2(R[2, 1], R[2, 0])
        if np.abs(np.sin(gamma)) < sys.float_info.epsilon:
            sign_sb = np.sgn(-R(0, 2) / np.cos(gamma))
        else:
            sign_sb = np.sgn(R[1, 2]) if np.sin(gamma) > 0 else -np.sgn(R[1, 2])
        beta  = np.atan2(sign_sb * abs_sb, R[2, 2])

    else:
        if R[2, 2] > 0:
            alpha = 0
            beta  = 0
            gamma = np.atan2(-R[1, 0], R[0, 0])
        else:
            alpha = 0
            beta  = np.pi
            gamma = np.atan2(R[1, 0], -R[0, 0])

    return alpha, beta, gamma
