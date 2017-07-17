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
from procrustes import TwoSidedPermutationSingleTransformationProcrustes
from numpy.testing import assert_almost_equal


def make_rotation_array(theta):
    arr = np.array([[np.cos(theta), -np.sin(theta), 0.],
                    [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    return arr


def test_two_sided_permutation_single_transformation_identical():
    # define an arbitrary symmetric array
    array_a = np.array([[12, 22, 63, 45, 7], [23, 54, 66, 63, 21]])
    array_a = np.dot(array_a, array_a.T)
    # compute procrustes transformation with umeyama guess
    # proc = TwoSidedPermutationSingleTransformationProcrustes(array_a, array_a, guess='umeyama')
    # assert_almost_equal(proc.error, 0, decimal=8)
    # assert_almost_equal(abs(np.linalg.det(proc.array_p)), decimal=6)
    # assert_almost_equal(abs(proc.array_p), np.eye(2), decimal=6)


def test_two_sided_permutation_single_transformation_perm_pad():
    # define an arbitrary symmetric array
    array_a = np.array([[5., 2., 1.], [4., 6., 1.], [1., 6., 3.]])
    array_a = np.dot(array_a, array_a.T)
    # define array_b by permuting array_a and padding with zero
    perm = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
    array_b = np.dot(np.dot(perm.T, array_a), perm)
    array_b = np.concatenate((array_b, np.zeros((3, 2))), axis=1)
    array_b = np.concatenate((array_b, np.zeros((1, 5))), axis=0)
    # compute procrustes transformation
    # proc = TwoSidedPermutationSingleTransformationProcrustes(array_a, array_b, scheme='umeyama')
    # check transformation array and error
    # assert_almost_equal(np.dot(proc.array_p, proc.array_p.T), np.eye(3), decimal=8)
    # assert_almost_equal(abs(np.linalg.det(proc.array_p)), 1.0, decimal=8)
    # assert_almost_equal(proc.error, 0, decimal=8)


def test_two_sided_permutation_single_transformation_scale_translate_perm_3by3():
    # define an arbitrary symmetric array
    array_a = np.array([[5., 2., 1.], [4., 6., 1.], [1., 6., 3.]])
    array_a = np.dot(array_a, array_a.T)
    # define array_b by scale-translate array_a and permuting
    shift = np.array([[3.14, 3.14, 3.14], [3.14, 3.14, 3.14], [3.14, 3.14, 3.14]])
    perm = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
    array_b = np.dot(np.dot(perm.T, 14.7 * array_a + shift), perm)
    # compute procrustes transformation
    # proc = TwoSidedPermutationSingleTransformationProcrustes(array_a, array_b,
    #                                                          translate=True, scale=True)
    # check transformation array and error
    # assert_almost_equal(np.dot(proc.array_p, proc.array_p.T), np.eye(3), decimal=8)
    # assert_almost_equal(abs(np.linalg.det(proc.array_p)), 1.0, decimal=8)
    # assert_almost_equal(proc.error, 0, decimal=8)


def test_two_sided_permutation_single_transformation_scale_translate_perm_4by4():
    # define an arbitrary symmetric array
    array_a = np.array([[14.4, 16.2, 36.5, 53.1], [42.4, 43.1, 25.3, 53.1],
                        [11.3, 26.5, 37.2, 21.1], [35.2, 62.1, 12.12, 21.3]])
    array_a = np.dot(array_a, array_a.T)
    # define array_b by scale-translate array_a and permuting
    shift = np.array([[2.7818, 2.7818, 2.7818, 2.7818], [2.7818, 2.7818, 2.7818, 2.7818],
                      [2.7818, 2.7818, 2.7818, 2.7818], [2.7818, 2.7818, 2.7818, 2.7818]])
    perm = np.array([[0., 0., 1., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.], [0., 1., 0., 0.]])
    array_b = np.dot(np.dot(perm.T, 22.4 * array_a + shift), perm)
    # compute procrustes transformation
    # proc = TwoSidedPermutationSingleTransformationProcrustes(array_a, array_b,
    #                                                         translate=True, scale=True)
    # check transformation array and error
    # assert_almost_equal(np.dot(proc.array_p, proc.array_p.T), np.eye(3), decimal=8)
    # assert_almost_equal(abs(np.linalg.det(proc.array_p)), 1.0, decimal=8)
    # assert_almost_equal(proc.error, 0, decimal=8)
