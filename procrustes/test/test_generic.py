# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2020 The Procrustes Development Team
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
"""Testings for generic Procrustes module."""

import numpy as np
from numpy.testing import assert_almost_equal
from procrustes.generic import generic


def test_generic_transformed():
    r"""Test generic Procrustes with random matrices."""
    # define arbitrary array and random transformation
    array_a = np.array([[3, 3, 5, 9],
                        [2, 6, 9, 9],
                        [1, 3, 5, 2],
                        [0, 4, 2, 9],
                        [9, 6, 0, 3]])
    array_x = np.array([[0.3, 0., 0.3, 0.6],
                        [0.7, 0.3, 0., 0.8],
                        [0.4, 0., 0.4, 0.5],
                        [0.3, 0.7, 0.1, 0.7]])
    array_b = np.dot(array_a, array_x)
    # compute procrustes transformation
    res = generic(array_a, array_b, translate=False, scale=False)
    # check transformation is right & error is zero
    assert_almost_equal(res["array_u"], array_x, decimal=6)
    assert_almost_equal(res["e_opt"], 0.0, decimal=6)


def test_generic_transformed_negative():
    r"""Test generic Procrustes with negative matrices."""
    # define arbitrary array and random transformation
    np.random.seed(999)
    array_a = np.random.randint(low=-10, high=10, size=(5, 4))
    array_x = np.array([[0.3, 0., 0.3, 0.6],
                        [0.7, 0.3, 0., 0.8],
                        [0.4, 0., 0.4, 0.5],
                        [0.3, 0.7, 0.1, 0.7]])
    array_b = np.dot(array_a, array_x)
    # compute procrustes transformation
    res = generic(array_a, array_b, translate=False, scale=False)
    # check transformation is right & error is zero
    assert_almost_equal(res["array_u"], array_x, decimal=6)
    assert_almost_equal(res["e_opt"], 0.0, decimal=6)


def test_generic_transformed_translate():
    r"""Test generic Procrustes with translation."""
    # define arbitrary array and random transformation
    np.random.seed(999)
    array_a = np.random.randint(low=-10, high=10, size=(5, 4))
    array_x = np.array([[0.3, 0., 0.3, 0.6],
                        [0.7, 0.3, 0., 0.8],
                        [0.4, 0., 0.4, 0.5],
                        [0.3, 0.7, 0.1, 0.7]])
    array_b = np.dot(array_a, array_x) + 100
    # compute procrustes transformation
    res = generic(array_a, array_b, translate=True, scale=False)
    # check transformation is right & error is zero
    assert_almost_equal(res["array_u"], array_x, decimal=6)
    assert_almost_equal(res["e_opt"], 0.0, decimal=6)


def test_generic_transformed_scale():
    r"""Test generic Procrustes with scaling."""
    # define arbitrary array and transformation
    np.random.seed(999)
    array_a = np.random.randint(low=-10, high=10, size=(5, 4))
    array_x = np.array([[0.3, 0., 0.3, 0.6],
                        [0.7, 0.3, 0., 0.8],
                        [0.4, 0., 0.4, 0.5],
                        [0.3, 0.7, 0.1, 0.7]])
    array_b = np.dot(array_a, array_x)
    # compute procrustes transformation
    res = generic(array_a, array_b, translate=False, scale=True)
    # check transformation is right & error is zero
    assert_almost_equal(res["e_opt"], 0.0, decimal=6)


def test_generic_transformed_translate_scale():
    r"""Test generic Procrustes with translation and scaling."""
    # define arbitrary array and random transformation
    np.random.seed(999)
    array_a = np.random.randint(low=-10, high=10, size=(5, 4))
    array_x = np.array([[0.3, 0., 0.3, 0.6],
                        [0.7, 0.3, 0., 0.8],
                        [0.4, 0., 0.4, 0.5],
                        [0.3, 0.7, 0.1, 0.7]])
    array_b = np.dot(4.5 * (array_a + 9), array_x)
    # compute procrustes transformation
    res = generic(array_a, array_b, translate=True, scale=True)
    # check transformation is error is zero
    assert_almost_equal(res["e_opt"], 0.0, decimal=6)


def test_generic_random_transformation():
    r"""Test generic Procrustes with randomized transformation matrices."""
    # define arbitrary array and random transformation
    np.random.seed(999)
    array_a = np.random.randint(low=-10, high=10, size=(5, 4))
    array_x = np.random.randint(low=0, high=20, size=(4, 4)) / 10
    array_b = np.dot(array_a, array_x)
    # compute procrustes transformation
    res = generic(array_a, array_b, translate=False, scale=False)
    # check transformation is right & error is zero
    assert_almost_equal(res["array_u"], array_x, decimal=6)
    assert_almost_equal(res["e_opt"], 0.0, decimal=6)
