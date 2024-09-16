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
"""Testings for generalized Procrustes module."""

import numpy as np
from numpy.testing import assert_almost_equal, assert_raises

from procrustes.generalized import generalized


def test_generalized_with_reference():
    r"""Test generalized Procrustes with reference."""
    arr_b = np.array([[5, 0], [8, 0], [5, 5]])
    arr_c = np.dot(arr_b, _rotation(30))
    arr_d = np.dot(arr_b, _rotation(45))
    arr_e = np.dot(arr_b, _rotation(90))
    # arr_list = [arr_a, arr_b, arr_c, arr_d]
    # arr_list = [arr_b, arr_c, arr_d, arr_e]
    arr_list = [arr_c, arr_d, arr_e]
    arr_aligned, error = generalized(arr_list, ref=arr_b, tol=1.0e-7, n_iter=200)
    # one right alignment
    aligned = [
        np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0]]),
        np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0]]),
        np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0]]),
    ]
    assert_almost_equal(arr_aligned[0], aligned[0], decimal=7)
    assert_almost_equal(arr_aligned[1], aligned[1], decimal=7)
    assert_almost_equal(arr_aligned[2], aligned[2], decimal=7)
    assert_almost_equal(error, 0.0)


def test_generalized_without_reference():
    r"""Test generalized Procrustes without reference."""
    arr_b = np.array([[5, 0], [8, 0], [5, 5]])
    arr_c = np.dot(arr_b, _rotation(30))
    arr_d = np.dot(arr_b, _rotation(45))
    arr_e = np.dot(arr_b, _rotation(90))
    arr_list = [arr_b, arr_c, arr_d, arr_e]
    # arr_list = [arr_c, arr_d, arr_e]
    arr_aligned, error = generalized(arr_list, ref=None, tol=1.0e-7, n_iter=200)
    # one right alignment
    aligned = [
        np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0]]),
        np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0]]),
        np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0]]),
        np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0]]),
    ]
    assert_almost_equal(arr_aligned[0], aligned[0], decimal=7)
    assert_almost_equal(arr_aligned[1], aligned[1], decimal=7)
    assert_almost_equal(arr_aligned[2], aligned[2], decimal=7)
    assert_almost_equal(arr_aligned[3], aligned[3], decimal=7)
    assert_almost_equal(error, 0.0)


def test_generalized_invalid():
    """Test invalid input of n_iter for generalized Procrustes analysis."""
    arr_b = np.array([[5, 0], [8, 0], [5, 5]])
    arr_c = np.dot(arr_b, _rotation(30))
    arr_d = np.dot(arr_b, _rotation(45))
    arr_e = np.dot(arr_b, _rotation(90))
    arr_list = [arr_b, arr_c, arr_d, arr_e]
    assert_raises(ValueError, generalized, arr_list, None, 1.0e-7, n_iter=-5)


def _rotation(degree):
    """Generate the rotation matrix."""
    theta = np.radians(degree)
    rot = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
    return rot
