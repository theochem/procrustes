# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2024 The QC-Devs Community
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
"""Testing for rotational Procrustes module."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.stats import special_ortho_group

from procrustes import rotational


@pytest.mark.parametrize("m, n", np.random.randint(500, 1000, (5, 2)))
def test_rotational_orthogonal_identical(m, n):
    r"""Test rotational Procrustes with identical matrices."""
    # define an arbitrary array
    array_a = np.random.uniform(-10.0, 10.0, (m, n))
    array_b = np.copy(array_a)
    # compute Procrustes transformation
    res = rotational(array_a, array_b, translate=False, scale=False)
    # check result is rotation matrix, and error is zero.
    assert_almost_equal(np.dot(res.t, res.t.T), np.eye(n), decimal=6)
    assert_almost_equal(np.abs(np.linalg.det(res.t)), 1.0, decimal=6)
    assert_almost_equal(res.error, 0, decimal=6)


@pytest.mark.parametrize("m, n", np.random.randint(50, 500, (5, 2)))
def test_rotational_orthogonal_identical_lapack_driver(m, n):
    r"""Test rotational Procrustes with identical matrices and different lapack driver."""
    # define an arbitrary array
    array_a = np.random.uniform(-10.0, 10.0, (m, n))
    array_b = np.copy(array_a)
    # compute Procrustes transformation
    res = rotational(array_a, array_b, translate=False, scale=False, lapack_driver="gesdd")
    # check result is rotation matrix, and error is zero.
    assert_almost_equal(np.dot(res.t, res.t.T), np.eye(n), decimal=6)
    assert_almost_equal(np.abs(np.linalg.det(res.t)), 1.0, decimal=6)
    assert_almost_equal(res.error, 0, decimal=6)


@pytest.mark.parametrize("m, n, col_npad, row_npad", np.random.randint(100, 500, (5, 4)))
def test_rotational_orthogonal_rotation_unpadding(m, n, col_npad, row_npad):
    r"""Test rotational Procrustes with arrays being unpadded."""
    # define an arbitrary array
    array_a = np.random.uniform(-10.0, 10.0, (m, n))
    # Generate random rotation array and define array_b by rotating array_a and pad with zeros
    rot_array = special_ortho_group.rvs(n)
    array_b = np.dot(array_a, rot_array)
    array_b = np.concatenate((array_b, np.zeros((m, col_npad))), axis=1)
    array_b = np.concatenate((array_b, np.zeros((row_npad, n + col_npad))), axis=0)
    # compute procrustes transformation
    res = rotational(array_a, array_b, unpad_col=True, unpad_row=True)
    # check transformation array and error
    assert_almost_equal(np.dot(res.t, res.t.T), np.eye(n), decimal=6)
    assert_almost_equal(np.abs(np.linalg.det(res.t)), 1.0, decimal=6)
    assert_almost_equal(res.error, 0, decimal=6)


@pytest.mark.parametrize("m, n", np.random.randint(500, 1000, (5, 2)))
def test_rotational_orthogonal_rotation_translate_scale(m, n):
    r"""Test rotational Procrustes with translated and scaled array."""
    # define an arbitrary array
    array_a = np.random.uniform(-10.0, 10.0, (m, n))
    # Translate the rows, generate random rotation array
    # define array_b by scale and translation of array_a followed by rotation
    shift = np.array([np.random.uniform(-10.0, 10.0, (n,))] * m)
    rot_array = special_ortho_group.rvs(n)
    array_b = np.dot(2.0 * array_a, rot_array) + shift
    # compute procrustes transformation
    res = rotational(array_a, array_b, translate=True, scale=True)
    # check transformation array and error
    assert_almost_equal(np.dot(res.t, res.t.T), np.eye(n), decimal=6)
    assert_almost_equal(np.abs(np.linalg.det(res.t)), 1.0, decimal=6)
    assert_almost_equal(res.error, 0, decimal=6)


@pytest.mark.parametrize("m, n", np.random.randint(500, 1000, (5, 2)))
def test_rotational_orthogonal_almost_zero_array(m, n):
    r"""Test rotational Procrustes with matrices with almost zero entries."""
    # define an arbitrary array
    array_a = np.random.uniform(0.0, 1e-6, (m, n))
    # define array_b by scale and translation of array_a and then rotation
    shift = np.array([np.random.uniform(0.0, 1e-5, (n,))] * m)
    rot_array = special_ortho_group.rvs(n)
    array_b = np.dot(4.12 * array_a + shift, rot_array)
    # compute procrustes transformation
    res = rotational(array_a, array_b, translate=True, scale=True)
    # check transformation array and error
    assert_almost_equal(np.dot(res.t, res.t.T), np.eye(n), decimal=6)
    assert_almost_equal(np.abs(np.linalg.det(res.t)), 1.0, decimal=6)
    assert_almost_equal(res.error, 0, decimal=6)


def test_rotational_raises_error_shape_mismatch():
    r"""Test rotation Procrustes with inputs are not correct."""
    array_a = np.random.uniform(-10.0, 10.0, (100, 100))
    array_b = array_a.copy()
    # Set couple of the columns of b and rows of b (at the ends of the matrix) to zero.
    array_b[:, -3:] = 0.0
    array_b[-4:, :] = 0.0
    with pytest.raises(ValueError):
        rotational(array_a, array_b, pad=False, unpad_col=True)
    with pytest.raises(ValueError):
        rotational(array_a, array_b, pad=False, unpad_row=True)
    with pytest.raises(ValueError):
        rotational(array_a, array_b, pad=False, unpad_row=True, unpad_col=True)
