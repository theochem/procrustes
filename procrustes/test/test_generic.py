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
"""Test procrustes.generic module."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from procrustes.generic import generic

np.random.seed(2020)


@pytest.mark.parametrize("m", np.random.randint(2, 100, 25))
def test_generic_square(m):
    r"""Test generic Procrustes with random square matrices."""
    # random input & transformation arrays (size=mxm)
    array_a = np.random.uniform(-2.0, 2.0, (m, m))
    array_x = np.random.uniform(-2.0, 2.0, (m, m))
    array_b = np.dot(array_a, array_x)
    # compute procrustes transformation
    res = generic(array_a, array_b, translate=False, scale=False)
    # check error & arrays
    assert_almost_equal(res.error, 0.0, decimal=6)
    assert_almost_equal(res.t, array_x, decimal=6)
    assert_almost_equal(res.new_a, array_a, decimal=6)
    assert_almost_equal(res.new_b, array_b, decimal=6)


@pytest.mark.parametrize("m", np.random.randint(2, 100, 25))
def test_generic_square_lapack_driver_and_assertion_error(m):
    r"""Test generic Procrustes with random and non-default lapack driver with assertion error."""
    # random input & transformation arrays (size=mxm)
    array_a = np.random.uniform(-2.0, 2.0, (m, m))
    array_x = np.random.uniform(-2.0, 2.0, (m, m))
    array_b = np.dot(array_a, array_x)
    # compute procrustes transformation
    res = generic(array_a, array_b, translate=False, scale=False)
    # check error & arrays
    assert_almost_equal(res.error, 0.0, decimal=6)
    assert_almost_equal(res.t, array_x, decimal=6)
    assert_almost_equal(res.new_a, array_a, decimal=6)
    assert_almost_equal(res.new_b, array_b, decimal=6)


@pytest.mark.parametrize("m, n", np.random.randint(2, 100, (25, 2)))
def test_generic_rectangular(m, n):
    r"""Test generic Procrustes with random rectangular matrices."""
    # random input & transformation arrays (size=mxn)
    array_a = np.random.uniform(-2.0, 2.0, (m, n))
    array_x = np.random.uniform(-3.0, 3.0, (n, n))
    array_b = np.dot(array_a, array_x)
    # compute procrustes transformation
    res = generic(array_a, array_b, translate=False, scale=False)
    # check error & arrays
    assert_almost_equal(res.error, 0.0, decimal=6)
    assert_almost_equal(res.new_a, array_a, decimal=6)
    assert_almost_equal(res.new_b, array_b, decimal=6)
    # check transformation array, only if it is unique
    if m >= n:
        assert_almost_equal(res.t, array_x, decimal=6)


@pytest.mark.parametrize("m, n", np.random.randint(100, 300, (5, 2)))
def test_generic_rectangular_translate(m, n):
    r"""Test generic Procrustes with random rectangular matrices & translation."""
    # random input, transformation arrays, & translation (size=mxn)
    array_a = np.random.uniform(-4.0, 4.0, (m, n))
    array_x = np.random.uniform(-2.0, 2.0, (n, n))
    array_t = np.repeat(np.random.uniform(-3.0, 3.0, (1, n)), m, axis=0)
    array_b = np.dot(array_a, array_x) + array_t
    # compute procrustes transformation
    res = generic(array_a, array_b, translate=True, scale=False)
    # check error & arrays
    assert_almost_equal(res.error, 0.0, decimal=6)
    assert_almost_equal(res.new_a, array_a - np.mean(array_a, axis=0), decimal=6)
    assert_almost_equal(res.new_b, array_b - np.mean(array_b, axis=0), decimal=6)


@pytest.mark.parametrize("m, n", np.random.randint(200, 400, (10, 2)))
def test_generic_rectangular_scale(m, n):
    r"""Test generic Procrustes with random square matrices & translation."""
    # random input, transformation arrays, & translation (size=mxn)
    array_a = np.random.uniform(-5.0, 5.0, (m, n))
    array_x = np.random.uniform(-3.0, 3.0, (n, n))
    array_b = np.dot(array_a, array_x)
    # compute procrustes transformation
    res = generic(array_a, array_b, translate=False, scale=True)
    # check error & arrays
    assert_almost_equal(res.error, 0.0, decimal=6)
    norm_a = np.linalg.norm(array_a)
    norm_b = np.linalg.norm(array_b)
    assert_almost_equal(res.new_a, array_a / norm_a, decimal=6)
    assert_almost_equal(res.new_b, array_b / norm_b, decimal=6)
    # check transformation array, only if it is unique
    if m >= n:
        assert_almost_equal(res.t, array_x * norm_a / norm_b, decimal=6)


@pytest.mark.parametrize("m, n", np.random.randint(500, 1000, (5, 2)))
def test_generic_rectangular_translate_scale(m, n):
    r"""Test generic Procrustes with random square matrices & translation."""
    # random input, transformation arrays, & translation (size=mxn)
    array_a = np.random.uniform(-10.0, 10.0, (m, n))
    array_x = np.random.uniform(-7.0, 7.0, (n, n))
    array_b = np.dot(array_a, array_x)
    # compute procrustes transformation
    res = generic(array_a, array_b, translate=True, scale=True)
    # check error & arrays
    assert_almost_equal(res.error, 0.0, decimal=6)
    centered_a = array_a - np.mean(array_a, axis=0)
    centered_b = array_b - np.mean(array_b, axis=0)
    assert_almost_equal(res.new_a, centered_a / np.linalg.norm(centered_a), decimal=6)
    assert_almost_equal(res.new_b, centered_b / np.linalg.norm(centered_b), decimal=6)
