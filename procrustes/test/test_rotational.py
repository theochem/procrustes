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
from procrustes import rotational
from numpy.testing import assert_almost_equal


def test_rotational_orthogonal_identical():
    # define an arbitrary array
    array_a = np.array([[3, 6, 2, 1], [5, 6, 7, 6], [2, 1, 1, 1]])
    array_b = np.copy(array_a)
    # compute Procrustes transformation
    new_a, new_b, array_u, e_opt = rotational(
        array_a, array_b, translate=False, scale=False)
    # check transformation array and error
    assert_almost_equal(np.dot(array_u, array_u.T), np.eye(4), decimal=6)
    assert_almost_equal(np.linalg.det(array_u), 1.0, decimal=6)
    assert_almost_equal(e_opt, 0, decimal=6)


def test_rotational_orthogonal_rotation_pad():
    # define an arbitrary array
    array_a = np.array([[1, 7], [9, 4]])
    # define array_b by rotating array_a and pad with zeros
    theta = np.pi/4
    rot_array = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    array_b = np.dot(array_a, rot_array)
    array_b = np.concatenate((array_b, np.zeros((2, 10))), axis=1)
    array_b = np.concatenate((array_b, np.zeros((15, 12))), axis=0)
    # compute procrustes transformation
    new_a, new_b, array_u, e_opt = rotational(
        array_a, array_b, translate=False, scale=False)
    # check transformation array and error
    assert_almost_equal(np.dot(array_u, array_u.T), np.eye(2), decimal=6)
    assert_almost_equal(np.linalg.det(array_u), 1.0, decimal=6)
    assert_almost_equal(e_opt, 0, decimal=6)



def test_rotational_orthogonal_rotation_translate_scale():
    # define an arbitrary array
    array_a = np.array([[1., 7., 8.], [4., 6., 8.], [7., 9., 4.], [6., 8., 23.]])
    # define array_b by scale and translation of array_a and then rotation
    shift = np.array([[3., 21., 21.], [3., 21., 21.], [3., 21., 21.], [3., 21., 21.]])
    theta = 44.3 * np.pi / 5.7
    rot_array = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    array_b = np.dot(477.412 * array_a + shift, rot_array)
    # compute procrustes transformation
    new_a, new_b, array_u, e_opt = rotational(
        array_a, array_b, translate=True, scale=True)
    # check transformation array and error
    assert_almost_equal(np.dot(array_u, array_u.T), np.eye(3), decimal=6)
    assert_almost_equal(np.linalg.det(array_u), 1.0, decimal=6)
    assert_almost_equal(e_opt, 0, decimal=6)


def test_rotational_orthogonal_rotation_translate_scale_4by3():
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
    new_a, new_b, array_u, e_opt = rotational(
        array_a, array_b, translate=True, scale=True)
    # check transformation array and error
    assert_almost_equal(np.dot(array_u, array_u.T), np.eye(3), decimal=6)
    assert_almost_equal(np.linalg.det(array_u), 1.0, decimal=6)
    assert_almost_equal(e_opt, 0, decimal=6)


def test_rotational_orthogonal_zero_array():
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
    new_a, new_b, array_u, e_opt = rotational(
        array_a, array_b, translate=True, scale=True)
    # check transformation array and error
    assert_almost_equal(np.dot(array_u, array_u.T), np.eye(3), decimal=6)
    assert_almost_equal(np.linalg.det(array_u), 1.0, decimal=6)
    assert_almost_equal(e_opt, 0, decimal=6)
