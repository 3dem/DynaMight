#!/usr/bin/env python

"""
Module for calculations related to the SO(3) group
"""
import sys
import numpy as np
import torch

from typing import Tuple, Union, TypeVar

Tensor = TypeVar('torch.tensor')


def taitbryan_to_matrix(
        angles: Union[Tensor, np.ndarray]
) -> Union[Tensor, np.ndarray]:
    """
    Takes a batch of the three Tait-Bryan angles (rotation axis: xyz)
    in radians and returns a batch of the corresponding rotation matrices

    Supports both numpy arrays and torch tensor input

    :param angles: an array (B, 3) of the angels in radians
    :return: a 3x3 rotation matrix
    """

    if torch.is_tensor(angles):
        R = torch.zeros(len(angles), 3, 3, dtype=angles.dtype).to(angles.device)
        c0 = torch.cos(angles[:, 0])
        s0 = torch.sin(angles[:, 0])
        c1 = torch.cos(angles[:, 1])
        s1 = torch.sin(angles[:, 1])
        c2 = torch.cos(angles[:, 2])
        s2 = torch.sin(angles[:, 2])
    else:
        R = np.zeros((len(angles), 3, 3), dtype=angles.dtype)
        c0 = np.cos(angles[:, 0])
        s0 = np.sin(angles[:, 0])
        c1 = np.cos(angles[:, 1])
        s1 = np.sin(angles[:, 1])
        c2 = np.cos(angles[:, 2])
        s2 = np.sin(angles[:, 2])

    """
    Matrix multiplication of Rz * Ry * Rx
    |c2, -s2, 0|   |c1,  0, s1|   |1, 0,    0|
    |s2,  c2, 0| * |0,   1, 0 | * |0, c0, -s0| 
    |0,   0,  1|   |-s1, 0, c1|   |0, s0,  c0|
    """

    R[:, 0, 0] = c1 * c2
    R[:, 0, 1] = s0 * s1 * c2 - c0 * s2
    R[:, 0, 2] = c0 * s1 * c2 + s0 * s2
    R[:, 1, 0] = c1 * s2
    R[:, 1, 1] = c0 * c2 + s0 * s1 * s2
    R[:, 1, 2] = c0 * s1 * s2 - s0 * c2
    R[:, 2, 0] = -s1
    R[:, 2, 1] = s0 * c1
    R[:, 2, 2] = c0 * c1

    return R


def euler_to_matrix(
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


def quaternion_to_matrix(Q: Tensor) -> Tensor:
    """
    Covert quaternions into 3D rotation matrices.
    :param Q: a Bx4 quaternion (Bx4)
    :return: rotation matrix (Bx3x3)
    """

    # Extract the values from Q
    q0 = Q[:, 0]
    q1 = Q[:, 1]
    q2 = Q[:, 2]
    q3 = Q[:, 3]

    r = torch.empty(Q.shape[0], 3, 3).to(Q.device)
    r[:, 0, 0] = 2 * (q0 * q0 + q1 * q1) - 1
    r[:, 0, 1] = 2 * (q1 * q2 - q0 * q3)
    r[:, 0, 2] = 2 * (q1 * q3 + q0 * q2)
    r[:, 1, 0] = 2 * (q1 * q2 + q0 * q3)
    r[:, 1, 1] = 2 * (q0 * q0 + q2 * q2) - 1
    r[:, 1, 2] = 2 * (q2 * q3 - q0 * q1)
    r[:, 2, 0] = 2 * (q1 * q3 - q0 * q2)
    r[:, 2, 1] = 2 * (q2 * q3 + q0 * q1)
    r[:, 2, 2] = 2 * (q0 * q0 + q3 * q3) - 1

    return r


def matrix_to_quaternion(R: Tensor) -> Tensor:
    """
    Covert 3D rotation matrices to quaternions.
    :param R: rotation matrices (Bx3x3)
    :return: quaternion (Bx4)
    """
    # From paper:
    # Sarabandi, Soheil, and Federico Thomas.
    # "Accurate computation of quaternions from rotation matrices."
    # International Symposium on Advances in Robot Kinematics. Springer, Cham, 2018.
    Q = torch.empty(R.shape[0], 4).type(R.dtype).to(R.device)

    # Q0
    t = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    m = t > 0
    Q[m, 0] = .5 * torch.sqrt(1 + t[m])
    m = ~m
    Q[m, 0] = .5 * torch.sqrt(
        (
            torch.square(R[m, 2, 1] - R[m, 1, 2]) +
            torch.square(R[m, 0, 2] - R[m, 2, 0]) +
            torch.square(R[m, 1, 0] - R[m, 0, 1])
        ) / (3 - t[m])
    )

    # Q1
    t = R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]
    m = t > 0
    Q[m, 1] = .5 * torch.sqrt(1 + t[m])
    m = ~m
    Q[m, 1] = .5 * torch.sqrt(
        (
            torch.square(R[m, 2, 1] - R[m, 1, 2]) +
            torch.square(R[m, 0, 1] + R[m, 1, 0]) +
            torch.square(R[m, 2, 0] + R[m, 0, 2])
        ) / (3 - t[m])
    )
    Q[:, 1] *= torch.sign(R[:, 2, 1] - R[:, 1, 2])

    # Q2
    t = -R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2]
    m = t > 0
    Q[m, 2] = .5 * torch.sqrt(1 + t[m])
    m = ~m
    Q[m, 2] = .5 * torch.sqrt(
        (
            torch.square(R[m, 0, 2] - R[m, 2, 0]) +
            torch.square(R[m, 0, 1] + R[m, 1, 0]) +
            torch.square(R[m, 1, 2] + R[m, 2, 1])
        ) / (3 - t[m])
    )
    Q[:, 2] *= torch.sign(R[:, 0, 2] - R[:, 2, 0])

    # Q3
    t = -R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2]
    m = t > 0
    Q[m, 3] = .5 * torch.sqrt(1 + t[m])
    m = ~m
    Q[m, 3] = .5 * torch.sqrt(
        (
            torch.square(R[m, 1, 0] - R[m, 0, 1]) +
            torch.square(R[m, 2, 0] + R[m, 0, 2]) +
            torch.square(R[m, 2, 1] + R[m, 1, 2])
        ) / (3 - t[m])
    )
    Q[:, 3] *= torch.sign(R[:, 1, 0] - R[:, 0, 1])

    return Q


def normalize_quaternions(Q: Tensor) -> Tensor:
    norm = torch.sqrt(torch.sum(torch.square(Q), dim=1))
    assert torch.all(norm > 0)
    return Q / norm[:, None]


def random_rotation_matrix(count):
    """
    Returns random rotation matrices.
    :param count: number of rotation matrices to return (B)
    :return: rotation matrices(Bx3x3)
    """
    a12 = 2 * np.pi * torch.rand(count, 2)
    a3 = torch.rand((count, 1)).mul(2).sub(1).acos()
    return euler_to_matrix(torch.cat([a12, a3], 1))


def is_rotation_matrix(R, eps: float = 1e-6):
    """
    Test if R is a rotation matrix.
    :param R: rotational matrix to test (Bx3x3)
    :param eps: numerical error margin (default: 1e-6)
    """
    eye = torch.eye(R.shape[-1]).type(R.dtype).to(R.device)
    RRt = torch.matmul(R, torch.transpose(R, 1, 2))
    eye_ae = torch.abs(RRt - eye[None, ...])
    det_ae = torch.abs(torch.linalg.det(R) - 1)
    return torch.all(torch.all((eye_ae < eps), dim=-1), dim=-1) & (det_ae < eps)


if __name__ == "__main__":
    angles = torch.Tensor([[0.1, 0.2, 0.1], [1.1, 0.2, 1.2]])

    R1 = euler_to_matrix(angles)
    Q1 = matrix_to_quaternion(R1)
    R2 = quaternion_to_matrix(Q1)
