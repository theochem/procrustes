# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2021 The QC-Devs Community
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
from numpy.testing import assert_almost_equal
from procrustes import rotational
from scipy.stats import special_ortho_group
import pytest


@pytest.mark.parametrize("m, n", np.random.randint(500, 1000, (5, 2)))
def test_rotational_orthogonal_identical(m, n):
    r"""Test rotational Procrustes with identical matrices."""
    # define an arbitrary array
    array_a = np.random.uniform(-10.0, 10.0, (m, n))
    array_b = np.copy(array_a)
    # compute Procrustes transformation
    res = rotational(array_a, array_b, translate=False, scale=False)
    # check result is rotation matrix, and error is zero.
    assert_almost_equal(np.dot(res["t"], res["t"].T), np.eye(n), decimal=6)
    assert_almost_equal(np.abs(np.linalg.det(res["t"])), 1.0, decimal=6)
    assert_almost_equal(res["error"], 0, decimal=6)


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
    assert_almost_equal(np.dot(res["t"], res["t"].T), np.eye(n), decimal=6)
    assert_almost_equal(np.abs(np.linalg.det(res["t"])), 1.0, decimal=6)
    assert_almost_equal(res["error"], 0, decimal=6)


@pytest.mark.parametrize("m, n", np.random.randint(500, 1000, (5, 2)))
def test_rotational_orthogonal_rotation_translate_scale(m, n):
    r"""Test rotational Procrustes with translated and scaled array."""
    # define an arbitrary array
    array_a = np.random.uniform(-10.0, 10.0, (m, n))
    # Translate the rows, generate random rotation array
    # define array_b by scale and translation of array_a followed by rotation
    shift = np.array([np.random.uniform(-10., 10., (n,))] * m)
    rot_array = special_ortho_group.rvs(n)
    array_b = np.dot(2. * array_a, rot_array) + shift
    # compute procrustes transformation
    res = rotational(array_a, array_b, translate=True, scale=True)
    # check transformation array and error
    assert_almost_equal(np.dot(res["t"], res["t"].T), np.eye(n), decimal=6)
    assert_almost_equal(np.abs(np.linalg.det(res["t"])), 1.0, decimal=6)
    assert_almost_equal(res["error"], 0, decimal=6)


def test_rotational_orthogonal_rotation_translate_scale_4by3():
    r"""Test rotational Procrustes with 4by3 translated and scaled array."""
    # define an arbitrary array
    array_a = np.array([[31.4, 17.5, 18.4], [34.5, 26.5, 28.6],
                        [17.6, 19.3, 34.6], [46.3, 38.5, 23.3]])
    # define array_b by scale and translation of array_a and then rotation
    shift = np.array([[13.3, 21.5, 21.8], [13.3, 21.5, 21.8],
                      [13.3, 21.5, 21.8], [13.3, 21.5, 21.8]])
    theta = 4.24 * np.pi / 1.23
    rot_array = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    array_b = np.dot(12.54 * array_a + shift, rot_array)
    # compute procrustes transformation
    res = rotational(array_a, array_b, translate=True, scale=True)
    # check transformation array and error
    assert_almost_equal(np.dot(res["t"], res["t"].T), np.eye(3), decimal=6)
    assert_almost_equal(np.abs(np.linalg.det(res["t"])), 1.0, decimal=6)
    assert_almost_equal(res["error"], 0, decimal=6)


def test_rotational_orthogonal_zero_array():
    r"""Test rotational Procrustes with zero array."""
    # define an arbitrary array
    array_a = np.array([[4.35e-5, 1.52e-5, 8.16e-5], [4.14e-6, 16.41e-5, 18.3e-6],
                        [17.53e-5, 29.53e-5, 34.56e-5], [26.53e-5, 38.63e-5, 23.36e-5]])
    # define array_b by scale and translation of array_a and then rotation
    shift = np.array([[3.25e-6, 21.52e-6, 21.12e-6], [3.25e-6, 21.52e-6, 21.12e-6],
                      [3.25e-6, 21.52e-6, 21.12e-6], [3.25e-6, 21.52e-6, 21.12e-6]])
    theta = 1.12525 * np.pi / 5.642
    rot_array = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    array_b = np.dot(4.12 * array_a + shift, rot_array)
    # compute procrustes transformation
    res = rotational(array_a, array_b, translate=True, scale=True)
    # check transformation array and error
    assert_almost_equal(np.dot(res["t"], res["t"].T), np.eye(3), decimal=6)
    assert_almost_equal(np.abs(np.linalg.det(res["t"])), 1.0, decimal=6)
    assert_almost_equal(res["error"], 0, decimal=6)
