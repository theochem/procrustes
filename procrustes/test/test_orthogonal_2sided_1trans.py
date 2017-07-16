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
from procrustes import TwoSidedOrthogonalSingleTransformationProcrustes
from numpy.testing import assert_almost_equal


def make_rotation_array(theta):
    arr = np.array([[np.cos(theta), -np.sin(theta), 0.],
                    [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    return arr


def test_two_sided_orthogonal_single_transformation_idential():
    # define an arbitrary symmetric array
    array_a = np.array([[2, 5, 4, 1], [5, 3, 1, 2], [8, 9, 1, 0], [1, 5, 6, 7]])
    array_a = np.dot(array_a, array_a.T)
    # compute exact procrustes transformation
    proc = TwoSidedOrthogonalSingleTransformationProcrustes(array_a, array_a, scheme='exact')
    # check transformation array and error
    assert_almost_equal(np.dot(proc.array_u, proc.array_u.T), np.eye(4), decimal=8)
    assert_almost_equal(abs(proc.array_u), np.eye(4), decimal=8)
    assert_almost_equal(abs(np.linalg.det(proc.array_u)), 1.0, decimal=8)
    assert_almost_equal(proc.error, 0, decimal=8)
    # # compute approximate procrustes transformation
    # proc = TwoSidedOrthogonalSingleTransformationProcrustes(array_a, array_a, scheme='approx')
    # # check transformation array and error
    # assert_almost_equal(np.dot(proc.array_u, proc.array_u.T), np.eye(4), decimal=8)
    # assert_almost_equal(abs(proc.array_u), np.eye(4), decimal=8)
    # assert_almost_equal(abs(np.linalg.det(proc.array_u)), 1.0, decimal=8)
    # assert_almost_equal(proc.error, 0, decimal=8)


def test_two_sided_orthogonal_single_transformation_padded():
    # define an arbitrary symmetric array
    array = np.array([[5, 2, 1], [4, 6, 1], [1, 6, 3]])
    array_a = np.dot(array, array.T)
    # define transformation arrays as a combination of rotation and reflection
    rot = make_rotation_array(16. * np.pi / 5.)
    ref = 1./3 * np.array([[1, -2, -2], [-2, 1, -2], [-2, -2, 1]])
    trans = np.dot(rot, ref)
    # define array_b by transforming array_a and padding with zero
    array_b = np.dot(np.dot(trans.T, array_a), trans)
    array_b = np.concatenate((array_b, np.zeros((3, 5))), axis=1)
    array_b = np.concatenate((array_b, np.zeros((5, 8))), axis=0)
    # compute approximate procrustes transformation
    proc = TwoSidedOrthogonalSingleTransformationProcrustes(array_a, array_b, scheme='approx')
    # check transformation array and error
    assert_almost_equal(np.dot(proc.array_u, proc.array_u.T), np.eye(3), decimal=8)
    assert_almost_equal(abs(np.linalg.det(proc.array_u)), 1.0, decimal=8)
    # assert_almost_equal(proc.error, 0, decimal=8)
    # compute exact procrustes transformation
    proc = TwoSidedOrthogonalSingleTransformationProcrustes(array_a, array_b, scheme='exact')
    # check transformation array and error
    assert_almost_equal(np.dot(proc.array_u, proc.array_u.T), np.eye(3), decimal=8)
    assert_almost_equal(abs(np.linalg.det(proc.array_u)), 1.0, decimal=8)
    assert_almost_equal(proc.error, 0, decimal=8)


def test_two_sided_orthogonal_single_transformation_scale_translate():
    # define an arbitrary symmetric array
    array = np.array([[12.43, 16.15, 17.61], [11.4, 21.5, 16.7], [16.4, 19.4, 14.9]])
    array_a = np.dot(array, array.T)
    # define transformation composed of rotation and reflection
    rot = make_rotation_array(12. * np.pi / 6.3)
    ref = 1./3 * np.array([[1, -2, -2], [-2, 1, -2], [-2, -2, 1]])
    trans = np.dot(ref, rot)
    # define array_b by transforming scaled-and-traslated array_a
    array_b = 6.9 * array_a + np.array([[6.7, 6.7, 6.7], [6.7, 6.7, 6.7], [6.7, 6.7, 6.7]])
    array_b = np.dot(np.dot(trans.T, array_b), trans)
    # compute exact procrustes transformation
    proc = TwoSidedOrthogonalSingleTransformationProcrustes(array_a, array_b, True, True, 'exact')
    # check transformation array and error
    assert_almost_equal(np.dot(proc.array_u, proc.array_u.T), np.eye(3), decimal=8)
    assert_almost_equal(abs(np.linalg.det(proc.array_u)), 1.0, decimal=8)
    # assert_almost_equal(proc.error, 0, decimal=8)
    # # compute approximate procrustes transformation
    # proc = TwoSidedOrthogonalSingleTransformationProcrustes(array_a, array_b, True, True, 'approx')
    # # check transformation array and error
    # assert_almost_equal(np.dot(proc.array_u, proc.array_u.T), np.eye(3), decimal=8)
    # assert_almost_equal(abs(np.linalg.det(proc.array_u)), 1.0, decimal=8)
    # assert_almost_equal(proc.error, 0, decimal=8)


def test_two_sided_orthogonal_single_transformation_shift_scale_rot_ref_2by2():
    # define an arbitrary symmetric array
    array = np.array([[124.72, 147.93], [120.5, 59.41]])
    array_a = np.dot(array, array.T)
    # define transformation composed of rotation and reflection
    rot = make_rotation_array(43.89 * np.pi / 12.43)[:2, :2]
    ref = np.array([[1., 0.], [0., -1.]])
    trans = np.dot(ref, rot)
    # define array_b by transforming scaled-and-traslated array_a (symmetric shift)
    shift = np.array([[45.91, 45.91], [45.91, 45.91]])
    array_b = 88.89 * array_a + shift
    array_b = np.dot(np.dot(trans.T, array_b), trans)
    # compute exact procrustes transformation
    proc = TwoSidedOrthogonalSingleTransformationProcrustes(array_a, array_b, True, True, 'exact')
    assert(abs(np.dot(proc.array_u, proc.array_u.T) - np.eye(2)) < 1.e-8).all()
    assert_almost_equal(abs(np.linalg.det(proc.array_u)), 1.0, decimal=8)
    # assert_almost_equal(proc.error, 0, decimal=8)
    # # compute approximate procrustes transformation
    # proc = TwoSidedOrthogonalSingleTransformationProcrustes(array_a, array_b, True, True, 'approx')
    # assert(abs(np.dot(proc.array_u, proc.array_u.T) - np.eye(2)) < 1.e-8).all()
    # assert_almost_equal(abs(np.linalg.det(proc.array_u)), 1.0, decimal=8)
    # # assert_almost_equal(proc.error, 0, decimal=8)


def test_two_sided_orthogonal_single_transformation_shift_scale_rot_ref():
    # define an arbitrary symmetric array
    array = np.array([[67.93, 147.93, 32.78], [21.59, 59.41, 79.90], [58.4, 49.4, 85.9]])
    array_a = np.dot(array, array.T)
    # define transformation composed of rotation and reflection
    rot = make_rotation_array(68.54 * np.pi / 23.41)
    ref = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
    trans = np.dot(ref, rot)
    # define array_b by transforming scaled-and-traslated array_a (symmetric shift)
    shift = np.array([[26.98, 26.98, 26.98], [26.98, 26.98, 26.98], [26.98, 26.98, 26.98]])
    array_b = 6.9 * array_a + shift
    array_b = np.dot(np.dot(trans.T, array_b), trans)
    # compute exact procrustes transformation
    proc = TwoSidedOrthogonalSingleTransformationProcrustes(array_a, array_b, True, True, 'exact')
    assert(abs(np.dot(proc.array_u, proc.array_u.T) - np.eye(3)) < 1.e-8).all()
    assert_almost_equal(abs(np.linalg.det(proc.array_u)), 1.0, decimal=8)
    # assert_almost_equal(proc.error, 0, decimal=8)
