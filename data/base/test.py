#!/usr/bin/env python3

"""
Test module for the base module
"""

import unittest

from voxelium.base import matrix_to_quaternion, quaternion_to_matrix, is_rotation_matrix, \
    euler_to_matrix, taitbryan_to_matrix
from voxelium.sparse_refine.svr_linear.sparse_linear import *

ERROR_EPS = 1e-5


class TestSparseLinear(unittest.TestCase):
    def test_euler_to_matrix(self):
        a = self._get_random_angles(1000).double()
        R = euler_to_matrix(a)
        self.assertTrue(torch.all(is_rotation_matrix(R)))

    def test_taitbyant_to_matrix(self):
        a = self._get_random_angles(1000).double()
        R = taitbryan_to_matrix(a)
        self.assertTrue(torch.all(is_rotation_matrix(R)))

    def test_quaternions(self):
        a = self._get_random_angles(1000).double()
        R1 = euler_to_matrix(a)
        self.assertTrue(torch.all(is_rotation_matrix(R1)))
        Q1 = matrix_to_quaternion(R1)
        R2 = quaternion_to_matrix(Q1)
        self.assertTrue(torch.all(is_rotation_matrix(R2)))
        D1 = torch.abs(R1 - R2)
        self.assertTrue(torch.all(D1 < ERROR_EPS))
        Q2 = matrix_to_quaternion(R2)
        D2 = torch.abs(Q1 - Q2)
        self.assertTrue(torch.all(D2 < ERROR_EPS))

    @staticmethod
    def _get_random_angles(count):
        a12 = 2 * np.pi * torch.rand(count, 2)
        a3 = torch.rand((count, 1)).mul(2).sub(1).acos()
        return torch.cat([a12, a3], 1)

if __name__ == "__main__":
    test = TestSparseLinear()
    test.test_quaternions()
    print("All good!")
