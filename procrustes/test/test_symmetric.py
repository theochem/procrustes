# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2018 The Procrustes Development Team
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


import numpy as np
from procrustes import symmetric
from numpy.testing import assert_almost_equal


def test_symmetric_transformed():
    # define arbitrary array and symmetric transformation
    array_a = np.array([[1, 2, 4, 5], [5, 7, 3, 3], [1, 5, 1, 9], [1, 5, 2, 7], [5, 7, 9, 0]])
    sym_array = np.dot(np.array([[1, 7, 4, 9]]).T, np.array([[1, 7, 4, 9]]))
    # define array_b by transforming array_a and padding with zero
    array_b = np.dot(array_a, sym_array)
    array_b = np.concatenate((array_b, np.zeros((5, 2))), axis=1)
    array_b = np.concatenate((array_b, np.zeros((8, 6))), axis=0)
    # compute procrustes transformation
    new_a, new_b, array_x, e_opt = symmetric(array_a, array_b, translate=False, scale=False)
    # check transformation is symmetric & error is zero
    assert_almost_equal(array_x, array_x.T, decimal=6)
    assert_almost_equal(array_x, sym_array, decimal=6)
    assert_almost_equal(e_opt, 0.0, decimal=6)


def test_symmetric_scaled_shifted_tranformed():
    # define an arbitrary array_a, translation matrix & symmetric matrix
    array_a = np.array([[5, 2, 8], [2, 2, 3], [1, 5, 6], [7, 3, 2]], dtype=float)
    shift = np.array([[9., 4., 3.], [9., 4., 3.], [9., 4., 3.], [9., 4., 3.]])
    sym_array = np.dot(np.array([[1, 4, 9]]).T, np.array([[1, 4, 9]]))
    # define array_b by scaling, translating and transforming array_a
    array_b = 614.5 * array_a + shift
    array_b = np.dot(array_b, sym_array)
    # compute procrustes transformation
    new_a, new_b, array_x, e_opt = symmetric(
        array_a, array_b, translate=True, scale=True)
    # check transformation is symmetric & error is zero
    assert_almost_equal(array_x, array_x.T, decimal=6)
    assert_almost_equal(e_opt, 0, decimal=6)


def test_symmetric_scaled_shifted_tranformed_4by3():
    # define an arbitrary array_a, translation matrix & symmetric matrix
    array_a = np.array([[245.0, 122.4, 538.5], [122.5, 252.2, 352.2],
                        [152.5, 515.2, 126.5], [357.5, 312.5, 225.5]])
    shift = np.array([[19.3, 14.2, 13.1], [19.3, 14.2, 13.1],
                      [19.3, 14.2, 13.1], [19.3, 14.2, 13.1]])
    sym_array = np.dot(np.array([[111.4, 144.9, 249.6]]).T, np.array([[111.4, 144.9, 249.6]]))
    # define array_b by scaling, translating and transforming array_a
    array_b = 312.5 * array_a + shift
    array_b = np.dot(array_b, sym_array)
    # compute procrustes transformation
    new_a, new_b, array_x, e_opt = symmetric(
        array_a, array_b, translate=True, scale=True)
    # check transformation is symmetric & error is zero
    assert_almost_equal(array_x, array_x.T, decimal=6)
    assert_almost_equal(e_opt, 0, decimal=6)


def test_symmetric():
    # define an arbitrary array_a, translation matrix & symmetric matrix
    array_a = np.array([[5.52e-5, 2.15e-5, 8.12e-5], [2.14e-5, 2.22e-5, 3.14e-5],
                        [1.11e-5, 5.94e-5, 6.58e-5], [7.15e-5, 3.62e-5, 2.24e-5]])
    shift = np.array([[9.42e-6, 4.32e-6, 3.22e-5], [9.42e-6, 4.32e-6, 3.22e-5],
                      [9.42e-6, 4.32e-6, 3.22e-5], [9.42e-6, 4.32e-6, 3.22e-5]])
    sym_array = np.dot(np.array([[5.2, 6.7, 3.5]]).T, np.array([[5.2, 6.7, 3.5]]))
    # define array_b by scaling, translating and transforming array_a
    array_b = 6.61e-4 * array_a + shift
    array_b = np.dot(array_b, sym_array)
    # compute procrustes transformation
    new_a, new_b, array_x, e_opt = symmetric(
        array_a, array_b, translate=True, scale=True)
    # check transformation is symmetric & error is zero
    assert_almost_equal(array_x, array_x.T, decimal=6)
    assert_almost_equal(e_opt, 0, decimal=6)


def test_not_full_rank_case():
    # Define a random matrix and symmetric matrix
    array_a = np.array([[10, 83], [52, 58], [58, 44]])
    sym_array = np.array([[0.38895636, 0.30523869], [0.30523869, 0.30856369]])
    array_b = np.dot(array_a, sym_array)
    # compute procrustes transformation
    new_a, new_b, array_x, e_opt = symmetric(array_a, array_b)
    # check transformation is symmetric & error is zero
    assert_almost_equal(array_x, array_x.T, decimal=6)
    assert_almost_equal(e_opt, 0, decimal=6)
