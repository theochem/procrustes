# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2022 The QC-Devs Community
#
# This file is part of Procrustes.
#
# Procrustes is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Procrustes is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
r"""Testings for PSDP (positive semi-definite Procrustes) module."""

import numpy as np
from numpy.testing import assert_almost_equal
from procrustes.psdp import psdp_woodgate


def test_psdp_identity(n=np.random.randint(50, 100)):
    r"""Test PSDP with identity matrix."""
    a = np.eye(n)
    b = np.eye(n)
    out = psdp_woodgate(a=a, b=b)
    res = out.__getattr__("s")
    assert_almost_equal(res, np.eye(n))


def test_psdp_diagonal():
    r"""Test PSDP with diagonal matrix."""
    a = np.diag([1, 2, 3, 4])
    b = np.eye(4)
    out = psdp_woodgate(a=a, b=b)
    res = out.__getattr__("s")
    actual_result = np.diag([0.99999, 0.5, 0.33333, 0.25])
    assert_almost_equal(res, actual_result, decimal=5)


def test_psdp_generic_square():
    r"""Test PSDP with 2 generic square matrices."""
    # The example given here is from the original paper.
    a = np.array([[1, 6, 0], [4, 3, 0], [0, 0, -0.5]])
    b = np.array([[1, 0, 0], [0, -2, 3], [0, 2, 4]])
    out = psdp_woodgate(a=a, b=b)
    res = out.__getattr__("s")
    actual_result = np.array(
        [
            [0.22351489, -0.11059539, 0.24342428],
            [-0.11059539, 0.05472271, -0.12044658],
            [0.24342428, -0.12044658, 0.26510708],
        ]
    )
    assert_almost_equal(res, actual_result)


def test_psdp_generic_non_square():
    r"""Test PSDP with 2 generic non-square matrices."""
    # The example given here is from the original paper.
    a = np.array([[5, 1, 6, -1], [3, 2, 0, 2], [2, 4, 3, -3]])
    b = np.array([[15, 1, 15 - 3, 2 + 5], [10, 5, 6, 3], [-3, 3, -3, -2 + 4]])
    out = psdp_woodgate(a=a, b=b)
    res = out.__getattr__("s")
    actual_result = np.array(
        [
            [2.57997197, 1.11007896, -1.08770156],
            [1.11007896, 1.68429863, 0.12829214],
            [-1.08770156, 0.12829214, 0.75328052],
        ]
    )
    assert_almost_equal(res, actual_result, decimal=5)
