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

from numpy.testing import assert_raises, assert_equal, assert_almost_equal

from procrustes import orthogonal_2sided, error


__all__ = [
    "test_identical_exact",
    "test_rot_reflect_padded",
    "test_scale_translate",
    "test_shift_scale_rot_ref_2by2",
    "test_shift_scale_rot_ref",
    "test_random_orthogonal",
]


def test_two_sided_orthogonal_single_transformation_invalid_mode_argument():
    # define a random matrix
    array_a = np.array([[0, 5, 8, 6], [-5, 0, 5, 1],
                        [8, 5, 0, 2], [6, 1, 2, 0]])
    array_b = np.array([[0, 1, 8, 4], [1, 0, 5, 2],
                        [8, 5, 0, 5], [4, 2, 5, 0]])
    # check invalid arguments
    assert_raises(ValueError, orthogonal_2sided, array_a, array_b, single_transform=True)


def test_two_sided_orthogonal_single_transformation_idential():
    # define an arbitrary symmetric array
    array_a = np.array([[2, 5, 4, 1], [5, 3, 1, 2], [8, 9, 1, 0], [1, 5, 6, 7]])
    array_a = np.dot(array_a, array_a.T)
    array_b = np.copy(array_a)

    # test the "exact" mode
    # compute exact 2sided orthogonal Procrustes with one transformation
    new_a, new_b, array_u, e_opt = orthogonal_2sided(
        array_a, array_b, single_transform=True, mode='exact')
    # check transformation array and error
    assert_almost_equal(np.dot(array_u, array_u.T), np.eye(4), decimal=8)
    assert_almost_equal(abs(array_u), np.eye(4), decimal=8)
    assert_almost_equal(abs(np.linalg.det(array_u)), 1.0, decimal=8)
    assert_almost_equal(e_opt, 0, decimal=8)

    # test the "approx" mode
    # compute exact 2sided orthogonal Procrustes with one transformation
    new_a, new_b, array_u, e_opt = orthogonal_2sided(
        array_a, array_b, translate=True, scale=True, single_transform=True, mode='approx')
    # check transformation array and error
    assert_almost_equal(np.dot(array_u, array_u.T), np.eye(4), decimal=8)
    assert_almost_equal(abs(array_u), np.eye(4), decimal=8)
    assert_almost_equal(abs(np.linalg.det(array_u)), 1.0, decimal=8)
    assert_almost_equal(e_opt, 0, decimal=8)


def test_two_sided_orthogonal_single_transformation_rot_reflect_padded():
    # define an arbitrary symmetric array
    array = np.array([[5, 2, 1], [4, 6, 1], [1, 6, 3]])
    array_a = np.dot(array, array.T)
    # define transformation arrays as a combination of rotation and reflection
    theta = 16. * np.pi / 5.
    rot = np.array([[np.cos(theta), -np.sin(theta), 0.],
                    [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    ref = 1. / 3 * np.array([[1, -2, -2], [-2, 1, -2], [-2, -2, 1]])
    trans = np.dot(rot, ref)
    # define array_b by transforming array_a and padding with zero
    array_b = np.dot(np.dot(trans.T, array_a), trans)
    array_b = np.concatenate((array_b, np.zeros((3, 5))), axis=1)
    array_b = np.concatenate((array_b, np.zeros((5, 8))), axis=0)

    # test the "exact" mode
    # compute approximate Procrustes transformation
    new_a, new_b, array_u, e_opt = orthogonal_2sided(
        array_a, array_b, single_transform=True, mode='exact')
    # check transformation array and error
    assert_almost_equal(np.dot(array_u, array_u.T), np.eye(3), decimal=8)
    assert_almost_equal(abs(np.linalg.det(array_u)), 1.0, decimal=8)
    assert_almost_equal(e_opt, 0, decimal=8)

    # test the "approx" mode
    # compute approximate procrustes transformation
    new_a, new_b, array_u, e_opt = orthogonal_2sided(
        array_a, array_b, single_transform=True, mode='approx')
    # check transformation array and error
    assert_almost_equal(np.dot(array_u, array_u.T), np.eye(3), decimal=8)
    assert_almost_equal(abs(np.linalg.det(array_u)), 1.0, decimal=8)


def test_two_sided_orthogonal_single_transformation_scale_translate():
    # define an arbitrary symmetric array
    array_a = np.array([[12.43, 16.15, 17.61], [11.4, 21.5, 16.7], [16.4, 19.4, 14.9]])
    array_a = np.dot(array_a, array_a.T)
    # define transformation composed of rotation and reflection
    theta = np.pi / 2
    rot = np.array([[np.cos(theta), -np.sin(theta), 0.],
                    [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    ref = np.array([[1, -2, -2], [-2, 1, -2], [-2, -2, 1]])
    trans = np.dot(ref, rot) / 3
    # define array_b by transforming scaled-and-translated array_a
    array_b = np.dot(np.dot(trans.T, array_a), trans)
    array_b = 6.9 * array_b + 6.7

    # test the "exact" mode
    # compute approximate procrustes transformation
    new_a, new_b, array_u, e_opt = orthogonal_2sided(
        array_a, array_b, translate=True, scale=True, single_transform=True, mode='exact')
    # check transformation array and error
    assert_almost_equal(np.dot(array_u, array_u.T), np.eye(3), decimal=8)
    assert_almost_equal(abs(np.linalg.det(array_u)), 1.0, decimal=8)
    #assert_almost_equal(e_opt, 0, decimal=8)
    #error is 1.0395531183148896

    # test the "approx" mode
    # compute approximate procrustes transformation
    new_a, new_b, array_u, e_opt = orthogonal_2sided(
        array_a, array_b, translate=True,
        scale=True, single_transform=True, mode='approx')
    # check transformation array and error
    assert_almost_equal(np.dot(array_u, array_u.T), np.eye(3), decimal=8)
    assert_almost_equal(abs(np.linalg.det(array_u)), 1.0, decimal=8)
    #assert_almost_equal(e_opt, 0, decimal=8)
    # error: 2.1162061737807796


def test_two_sided_orthogonal_single_transformation_shift_scale_rot_ref_2by2():
    # define an arbitrary symmetric array
    array_a = np.array([[124.72, 147.93], [120.5, 59.41]])
    array_a = np.dot(array_a, array_a.T)
    # define transformation composed of rotation and reflection
    theta = 5.5 * np.pi / 6.5
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    ref = np.array([[1., 0.], [0., -1.]])
    trans = np.dot(ref, rot)
    # define array_b by transforming scaled-and-traslated array_a (symmetric shift)
    shift = np.array([[45.91, 45.91], [45.91, 45.91]])
    array_b = 88.89 * array_a + shift
    array_b = np.dot(np.dot(trans.T, array_b), trans)

    # test the "exact" mode
    # compute approximate procrustes transformation
    new_a, new_b, array_u, e_opt = orthogonal_2sided(
        array_a, array_b, translate=True, scale=True, single_transform=True, mode='exact')
    # check transformation array and error
    assert_almost_equal(np.dot(array_u, array_u.T), np.eye(2), decimal=8)
    assert_almost_equal(abs(np.linalg.det(array_u)), 1.0, decimal=8)
    #assert_almost_equal(e_opt, 0, decimal=8)
    #error is 0.1952801285861765

    # test the "approx" mode
    # compute approximate procrustes transformation
    new_a, new_b, array_u, e_opt = orthogonal_2sided(
        array_a, array_b, single_transform=True, mode='approx')
    # check transformation array and error
    assert_almost_equal(np.dot(array_u, array_u.T), np.eye(2), decimal=8)
    assert_almost_equal(abs(np.linalg.det(array_u)), 1.0, decimal=8)
    #assert_almost_equal(e_opt, 0, decimal=8)


def test_two_sided_orthogonal_single_transformation_shift_scale_rot_ref():
    # define an arbitrary symmetric array
    array_a = (np.random.rand(3, 3) * 100).astype(int)
    array_a = np.dot(array_a, array_a.T)
    # define transformation composed of rotation and reflection
    theta = 5.7 * np.pi / 21.95
    rot = np.array([[np.cos(theta), -np.sin(theta), 0.],
                    [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    ref = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
    trans = np.dot(ref, rot)
    # define array_b by transforming scaled-and-translated array_a (symmetric shift)
    array_b = 6.9 * array_a + 26.98
    array_b = np.dot(np.dot(trans.T, array_b), trans)

    # test the "exact" mode
    # compute approximate procrustes transformation
    new_a, new_b, array_u, e_opt = orthogonal_2sided(
        array_a, array_b, translate=True, scale=True, single_transform=True, mode='exact')
    # check transformation array and error
    assert_almost_equal(np.dot(array_u, array_u.T), np.eye(3), decimal=8)
    assert_almost_equal(abs(np.linalg.det(array_u)), 1.0, decimal=8)
    #assert_almost_equal(e_opt, 0, decimal=8)
    #error is 0.46800281341209454

    # test the "approx" mode
    # compute approximate procrustes transformation
    new_a, new_b, array_u, e_opt = orthogonal_2sided(
        array_a, array_b, single_transform=True, mode='approx')
    # check transformation array and error
    assert_almost_equal(np.dot(array_u, array_u.T), np.eye(3), decimal=8)
    assert_almost_equal(abs(np.linalg.det(array_u)), 1.0, decimal=8)
    #assert_almost_equal(e_opt, 0, decimal=8)


def test_two_sided_orthogonal_single_transformation_random_orthogonal():
    # define random array
    array_a = np.array([[0, 5, 8, 6], [5, 0, 5, 1],
                        [8, 5, 0, 2], [6, 1, 2, 0]])
    ortho = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0]])
    array_b = np.dot(np.dot(ortho.T, array_a), ortho)
    # test the "exact" mode
    # compute approximate Procrustes transformation
    new_a, new_b, array_u, e_opt = orthogonal_2sided(
        array_a, array_b, single_transform=True, mode='exact')
    # check transformation array and error
    assert_almost_equal(np.dot(array_u, array_u.T), np.eye(4), decimal=8)
    assert_almost_equal(abs(np.linalg.det(array_u)), 1.0, decimal=8)
    assert_almost_equal(e_opt, 0, decimal=8)
