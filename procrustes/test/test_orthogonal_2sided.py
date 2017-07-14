# -*- coding: utf-8 -*-
# Procrustes is a collection of interpretive chemical tools for
# analyzing outputs of the quantum chemistry calculations.
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
from procrustes import TwoSidedOrthogonalProcrustes
from numpy.testing import assert_almost_equal


def make_rotation_array(theta):
    arr = np.array([[np.cos(theta), -np.sin(theta), 0.],
                    [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    return arr


def test_two_sided_orthogonal_identical():
    # define an arbitrary array
    array_a = np.array([[2, 5, 4, 1], [5, 3, 1, 2], [8, 9, 1, 0], [1, 5, 6, 7]])
    # compute procrustes transformation
    proc = TwoSidedOrthogonalProcrustes(array_a, array_a)
    # check transformation array and error
    assert_almost_equal(np.dot(proc.array_u1, proc.array_u1.T), np.eye(4), decimal=8)
    assert_almost_equal(np.dot(proc.array_u2, proc.array_u2.T), np.eye(4), decimal=8)
    assert_almost_equal(np.linalg.det(proc.array_u1), 1.0, decimal=8)
    assert_almost_equal(np.linalg.det(proc.array_u2), 1.0, decimal=8)
    # transformation should return zero error
    assert_almost_equal(proc.error, 0, decimal=8)


def test_two_sided_orthogonal_rotate_reflect_pad():
    # define an arbitrary array
    array_a = np.array([[1., 4., 6.], [6., 1., 4.], [7., 8., 1.]])
    # define transformation arrays as a combination of rotation and reflection
    rot1 = make_rotation_array(np.pi / 5.)
    rot2 = make_rotation_array(12. * np.pi / 9.)
    ref1 = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    ref2 = 1./3. * np.array([[1., -2., -2.], [-2., 1., -2.], [-2., -2., 1.]])
    trans1 = np.dot(ref1, rot1)
    trans2 = np.dot(rot2, ref2)
    # define array_b by transforming array_a and padding with zero
    array_b = np.dot(np.dot(trans1, array_a), trans2)
    array_b = np.concatenate((array_b, np.zeros((3, 5))), axis=1)
    array_b = np.concatenate((array_b, np.zeros((3, 8))), axis=0)
    # compute procrustes transformation
    proc = TwoSidedOrthogonalProcrustes(array_a, array_b)
    # check transformation array and error
    assert_almost_equal(np.dot(proc.array_u1, proc.array_u1.T), np.eye(3), decimal=8)
    assert_almost_equal(np.dot(proc.array_u2, proc.array_u2.T), np.eye(3), decimal=8)
    assert_almost_equal(np.linalg.det(proc.array_u1), 1.0, decimal=8)
    assert_almost_equal(np.linalg.det(proc.array_u2), 1.0, decimal=8)
    # transformation should return zero error
    assert_almost_equal(proc.error, 0, decimal=8)


def test_two_sided_orthogonal_translate_scale_rotate_reflect():
    # define an arbitrary array
    array_a = np.array([[1, 3, 5], [3, 5, 7], [8, 11, 15]])
    # define rotation and reflection arrays
    rot = make_rotation_array(1.8 * np.pi / 34.)
    ref = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    # define array_b by transforming scaled-and-traslated array_a
    shift = np.array([[16., 41., 33.], [16., 41., 33.], [16., 41., 33.]])
    array_b = np.dot(np.dot(ref, 23.5 * array_a + shift), rot)
    # compute procrustes transformation
    proc = TwoSidedOrthogonalProcrustes(array_a, array_b, translate=True, scale=True)
    # check transformation array and error
    assert_almost_equal(np.dot(proc.array_u1, proc.array_u1.T), np.eye(3), decimal=8)
    assert_almost_equal(np.dot(proc.array_u2, proc.array_u2.T), np.eye(3), decimal=8)
    assert_almost_equal(np.linalg.det(proc.array_u1), 1.0, decimal=8)
    assert_almost_equal(np.linalg.det(proc.array_u2), 1.0, decimal=8)
    # transformation should return zero error
    assert_almost_equal(proc.error, 0, decimal=8)


def test_two_sided_orthogonal_translate_scale_rotate_reflect_3by3():
    # define an arbitrary array
    array_a = np.array([[141.58, 315.25, 524.14], [253.25, 255.52, 357.51], [358.2, 131.6, 135.59]])
    # define rotation and reflection arrays
    rot = make_rotation_array(17.54 * np.pi / 6.89)
    ref = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    # define array_b by transforming scaled-and-traslated array_a
    shift = np.array([[146.56, 441.67, 343.56], [146.56, 441.67, 343.56], [146.56, 441.67, 343.56]])
    array_b = np.dot(np.dot(ref, 79.89 * array_a + shift), rot)
    # compute procrustes transformation
    proc = TwoSidedOrthogonalProcrustes(array_a, array_b, translate=True, scale=True)
    # check transformation array and error
    assert_almost_equal(np.dot(proc.array_u1, proc.array_u1.T), np.eye(3), decimal=8)
    assert_almost_equal(np.dot(proc.array_u2, proc.array_u2.T), np.eye(3), decimal=8)
    assert_almost_equal(np.linalg.det(proc.array_u1), 1.0, decimal=8)
    assert_almost_equal(np.linalg.det(proc.array_u2), 1.0, decimal=8)
    # transformation should return zero error
    assert_almost_equal(proc.error, 0, decimal=8)


def test_two_sided_orthogonal_rotate_reflect():
    # define an arbitrary array
    array_a = np.array([[41.8, 15.5, 24.4], [53.5, 55.2, 57.1], [58.2, 31.6, 35.9]])
    # define rotation and reflection arrays
    rot = make_rotation_array(-np.pi / 6)
    ref = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    # define array_b by transforming array_a
    array_b = np.dot(np.dot(ref, array_a), rot)
    # compute procrustes transformation
    proc = TwoSidedOrthogonalProcrustes(array_a, array_b, translate=True, scale=True)
    # check transformation array and error
    assert_almost_equal(np.dot(proc.array_u1, proc.array_u1.T), np.eye(3), decimal=8)
    assert_almost_equal(np.dot(proc.array_u2, proc.array_u2.T), np.eye(3), decimal=8)
    assert_almost_equal(np.linalg.det(proc.array_u1), 1.0, decimal=8)
    assert_almost_equal(np.linalg.det(proc.array_u2), 1.0, decimal=8)
    # transformation should return zero error
    assert_almost_equal(proc.error, 0, decimal=8)
