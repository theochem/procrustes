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
r"""Testings for orthogonal Procrustes module."""

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_raises
from procrustes.orthogonal import orthogonal, orthogonal_2sided
from procrustes.utils import compute_error


def make_rotation_array(theta):
    r"""Build the rotation array."""
    arr = np.array([[np.cos(theta), -np.sin(theta), 0.],
                    [np.sin(theta), np.cos(theta), 0.],
                    [0., 0., 1.]])
    return arr


def test_procrustes_orthogonal_identical():
    r"""Test orthogonal Procrustes with identity matrix."""
    # case of identical square arrays
    array_a = np.arange(9).reshape(3, 3)
    array_b = np.copy(array_a)
    res = orthogonal(array_a, array_b)
    # check transformation array is identity
    assert_almost_equal(res["new_a"], array_a, decimal=6)
    assert_almost_equal(res["new_b"], array_b, decimal=6)
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)
    # case of identical rectangular arrays (2 by 4)
    array_a = np.array([[1, 5, 6, 7], [1, 2, 9, 4]])
    array_b = np.copy(array_a)
    res = orthogonal(array_a, array_b)
    assert_almost_equal(res["new_a"], array_a, decimal=6)
    assert_almost_equal(res["new_b"], array_b, decimal=6)
    assert_equal(res["array_u"].shape, (4, 4))
    # assert_almost_equal(array_u, np.eye(4), decimal=6)
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)
    # case of identical rectangular arrays (5 by 3)
    array_a = np.arange(15).reshape(5, 3)
    array_b = np.copy(array_a)
    res = orthogonal(array_a, array_b)
    assert_almost_equal(res["new_a"], array_a, decimal=6)
    assert_almost_equal(res["new_b"], array_b, decimal=6)
    assert_equal(res["array_u"].shape, (3, 3))
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)


def test_procrustes_rotation_square():
    r"""Test orthogonal Procrustes with squared array."""
    # square array
    array_a = np.arange(4).reshape(2, 2)
    # rotation by 90 degree
    array_b = np.array([[1, 0], [3, -2]])
    res = orthogonal(array_a, array_b)
    assert_almost_equal(res["array_u"], np.array([[0., -1.], [1., 0.]]), decimal=6)
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)
    # rotation by 180 degree
    array_b = -array_a
    res = orthogonal(array_a, array_b)
    assert_almost_equal(res["array_u"], np.array([[-1., 0.], [0., -1.]]), decimal=6)
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)
    # rotation by 270 degree
    array_b = np.array([[-1, 0], [-3, 2]])
    res = orthogonal(array_a, array_b)
    assert_almost_equal(res["array_u"], np.array([[0., 1.], [-1., 0.]]), decimal=6)
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)
    # rotation by 45 degree
    rotation = 0.5 * np.sqrt(2) * np.array([[1, -1], [1, 1]])
    array_b = np.dot(array_a, rotation)
    res = orthogonal(array_a, array_b)
    assert_almost_equal(res["array_u"], rotation, decimal=6)
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)
    # rotation by 30 degree
    theta = np.pi / 6
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    array_b = np.dot(array_a, rotation)
    res = orthogonal(array_a, array_b)
    assert_almost_equal(res["array_u"], rotation, decimal=6)
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)
    # rotation by 72 degree
    theta = 1.25664
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    array_b = np.dot(array_a, rotation)
    res = orthogonal(array_a, array_b)
    assert_almost_equal(res["array_u"], rotation, decimal=6)
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)


def test_procrustes_reflection_square():
    r"""Test orthogonal Procrustes with reflected squared array."""
    # square array
    array_a = np.array([[2.0, 0.1], [0.5, 3.0]])
    # reflection through origin
    array_b = -array_a
    res = orthogonal(array_a, array_b)
    assert_almost_equal(res["new_a"], array_a, decimal=6)
    assert_almost_equal(res["new_b"], array_b, decimal=6)
    assert_almost_equal(res["array_u"], np.array([[-1, 0], [0, -1]]), decimal=6)
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)
    # reflection in the x-axis
    array_b = np.array([[2.0, -0.1], [0.5, -3.0]])
    res = orthogonal(array_a, array_b)
    assert_almost_equal(res["array_u"], np.array([[1, 0], [0, -1]]), decimal=6)
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)
    # reflection in the y-axis
    array_b = np.array([[-2.0, 0.1], [-0.5, 3.0]])
    res = orthogonal(array_a, array_b)
    assert_almost_equal(res["array_u"], np.array([[-1, 0], [0, 1]]), decimal=6)
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)
    # reflection in the line y=x
    array_b = np.array([[0.1, 2.0], [3.0, 0.5]])
    res = orthogonal(array_a, array_b)
    assert_almost_equal(res["array_u"], np.array([[0, 1], [1, 0]]), decimal=6)
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)


def test_procrustes_shifted():
    r"""Test orthogonal Procrustes with shifted array."""
    # square array
    array_a = np.array([[3.5, 0.1, 7.0], [0.5, 2.0, 1.0], [8.1, 0.3, 0.7]])
    # expected_a = array_a - np.mean(array_a, axis=0)
    # constant shift
    array_b = array_a + 4.1
    res = orthogonal(array_a, array_b, translate=True)
    # assert_almost_equal(new_b, array_b, decimal=6)
    assert_almost_equal(res["array_u"], np.eye(3), decimal=6)
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)
    # different shift along each axis
    array_b = array_a + np.array([0, 3.2, 5.0])
    res = orthogonal(array_a, array_b, translate=True)
    # assert_almost_equal(new_b, array_b, decimal=6)
    assert_almost_equal(res["array_u"], np.eye(3), decimal=6)
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)
    # rectangular (2 by 3)
    array_a = np.array([[1, 2, 3], [7, 9, 5]])
    # expected_a = array_a - np.array([4., 5.5, 4.])
    # constant shift
    array_b = array_a + 0.71
    res = orthogonal(array_a, array_b, translate=True)
    # assert_almost_equal(new_b, array_b, decimal=6)
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)
    # different shift along each axis
    array_b = array_a + np.array([0.3, 7.1, 4.2])
    res = orthogonal(array_a, array_b, translate=True)
    # assert_almost_equal(new_b, array_b, decimal=6)
    assert_equal(res["array_u"].shape, (3, 3))
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)


def test_procrustes_rotation_translation():
    r"""Test orthogonal Procrustes with rotation and translation."""
    # initial arrays
    array_a = np.array([[-7.3, 2.8], [-7.1, -0.2], [4.0, 1.4], [1.3, 0]])
    # rotation by 20 degree & reflection in the x-axis
    theta = 0.34907
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    reflection = np.array([[1, 0], [0, -1]])
    array_b = np.dot(array_a, np.dot(rotation, reflection))
    # procrustes without translation and scaling
    res = orthogonal(array_a, array_b)
    assert_almost_equal(res["new_a"], array_a, decimal=6)
    assert_almost_equal(res["new_b"], array_b, decimal=6)
    assert_almost_equal(res["array_u"], np.dot(rotation, reflection), decimal=6)
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)
    # procrustes with translation
    res = orthogonal(array_a, array_b, translate=True)
    assert_almost_equal(res["new_a"], array_a - np.mean(array_a, axis=0), decimal=6)
    assert_almost_equal(res["new_b"], array_b - np.mean(array_b, axis=0), decimal=6)
    assert_almost_equal(res["array_u"], np.dot(rotation, reflection), decimal=6)
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)
    # procrustes with translation and scaling
    res = orthogonal(array_a, array_b, translate=True, scale=True)
    assert_almost_equal(res["array_u"], np.dot(rotation, reflection), decimal=6)
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)


def test_rotation_translate_scale():
    r"""Test orthogonal Procrustes with rotation, translation and scaling."""
    # initial arrays
    array_a = np.array([[5.1, 0], [-1.1, 4.8], [3.9, 7.3], [9.1, 6.3]])
    # rotation by 68 degree & reflection in the Y=X
    theta = 1.18682
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    reflection = np.array([[0, 1], [1, 0]])
    array_b = np.dot(4 * array_a + 3.0, np.dot(rotation, reflection))
    # procrustes with translation and scaling
    res = orthogonal(array_a, array_b, translate=True, scale=True)
    assert_almost_equal(res["array_u"], np.dot(rotation, reflection), decimal=6)
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)


def test_orthogonal_translate_scale2():
    r"""Test orthogonal Procrustes with rotation, translation and scaling with a different array."""
    # initial array
    array_a = np.array([[1, 4], [7, 9]])
    # define a transformation composed of rotation & reflection
    theta = np.pi / 2
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    ref = np.array([[1, 0], [0, -1]])
    trans = np.dot(rot, ref)
    # define array_b by transforming array_a and padding with zero
    array_b = np.dot(array_a, trans)
    array_b = np.concatenate((array_b, np.zeros((2, 5))), axis=1)
    array_b = np.concatenate((array_b, np.zeros((5, 7))), axis=0)
    # compute procrustes transformation
    res = orthogonal(array_a, array_b, translate=False, scale=False)
    assert_almost_equal(res["array_u"], np.dot(rot, ref), decimal=6)
    assert_almost_equal(compute_error(res["new_a"], res["new_b"], res["array_u"]), 0., decimal=6)


def test_two_sided_orthogonal_identical():
    r"""Test 2-sided orthogonal with identical matrix."""
    # case of identical square arrays
    array_a = np.arange(16).reshape(4, 4)
    array_b = np.copy(array_a)
    result = orthogonal_2sided(array_a, array_b, single_transform=False)
    # check transformation array is identity
    assert_almost_equal(np.linalg.det(result["array_p"]), 1.0, decimal=6)
    assert_almost_equal(np.linalg.det(result["array_q"]), 1.0, decimal=6)
    assert_almost_equal(result["array_p"], np.eye(4), decimal=6)
    assert_almost_equal(result["array_q"], np.eye(4), decimal=6)
    assert_almost_equal(result["error"], 0., decimal=6)


def test_two_sided_orthogonal_raises_error_non_symmetric_matrices():
    r"""Test that 2-sided orthogonal procrustes non-symmetric matrices return an error."""
    # Test simple example with one matrix that is not square
    array_a = np.array([[1., 2., 3.], [1., 2., 3.]])
    array_b = np.array([[1., 2.], [2., 1.]])
    assert_raises(ValueError, orthogonal_2sided, array_a, array_b, single_transform=True)
    assert_raises(ValueError, orthogonal_2sided, array_b, array_a, single_transform=True)

    # Test one which is square but not symmetric.
    array_a = np.array([[1., 1.], [2., 2.]])
    array_b = np.array([[1., 2.], [2., 1.]])
    assert_raises(ValueError, orthogonal_2sided, array_a, array_b, single_transform=True)
    assert_raises(ValueError, orthogonal_2sided, array_b, array_a, single_transform=True)

    # Test one that works but removal of rows with bad padding gives an error.
    array_a = np.array([[1., 0.], [0., 0.]])
    array_b = np.array([[1., 2.], [2., 1.]])
    assert_raises(ValueError, orthogonal_2sided, array_a, array_b, single_transform=True,
                  remove_zero_col=True, pad_mode="row")
    assert_raises(ValueError, orthogonal_2sided, array_b, array_a, single_transform=True,
                  remove_zero_col=True, pad_mode="col")


def test_two_sided_orthogonal_rotate_reflect():
    r"""Test 2sided orthogonal by 3by3 array with rotation and reflection."""
    # define an arbitrary array
    array_a = np.array([[41.8, 15.5, 24.4], [53.5, 55.2, 57.1], [58.2, 31.6, 35.9]])
    # define rotation and reflection arrays
    rot = make_rotation_array(-np.pi / 6)
    ref = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    # define array_b by transforming array_a
    array_b = np.dot(np.dot(ref, array_a), rot)
    # compute procrustes transformation
    result = orthogonal_2sided(array_a, array_b, translate=True, scale=True, single_transform=False)
    # check transformation array orthogonality
    assert_almost_equal(np.dot(result["array_p"], result["array_p"].T), np.eye(3), decimal=6)
    assert_almost_equal(np.dot(result["array_q"], result["array_q"].T), np.eye(3), decimal=6)
    assert_almost_equal(np.abs(np.linalg.det(result["array_p"])), 1.0, decimal=6)
    assert_almost_equal(np.abs(np.linalg.det(result["array_q"])), 1.0, decimal=6)
    # transformation should return zero error
    assert_almost_equal(result["error"], 0, decimal=6)


def test_two_sided_orthogonal_rotate_reflect_pad():
    r"""Test 2sided orthogonal by 3by3 array with rotation, reflection and zero padding."""
    # define an arbitrary array
    array_a = np.array([[1., 4.], [6., 7]])
    # rotation by -45 degrees
    theta = -np.pi / 4
    rot2 = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
    array_b = np.dot(array_a, rot2)
    array_b = np.concatenate((array_b, np.zeros((2, 4))), axis=1)
    array_b = np.concatenate((array_b, np.zeros((2, 6))), axis=0)

    # compute Procrustes transformation
    result = orthogonal_2sided(array_a, array_b, translate=True, scale=True, single_transform=False)
    # check transformation array and error
    assert_almost_equal(np.dot(result["array_p"], result["array_p"].T), np.eye(2), decimal=6)
    assert_almost_equal(np.dot(result["array_q"], result["array_q"].T), np.eye(2), decimal=6)
    assert_almost_equal(abs(np.linalg.det(result["array_p"])), 1.0, decimal=6)
    assert_almost_equal(abs(np.linalg.det(result["array_q"])), 1.0, decimal=6)
    assert_almost_equal(result["error"], 0, decimal=6)


def test_two_sided_orthogonal_translate_scale_rotate_reflect():
    r"""Test 2sided orthogonal by 3by3 array with translation, rotation and reflection."""
    # define an arbitrary array
    array_a = np.array([[1, 3, 5], [3, 5, 7], [8, 11, 15]])
    # define rotation and reflection arrays
    rot = make_rotation_array(1.8 * np.pi / 34.)
    ref = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    # define array_b by transforming scaled-and-traslated array_a
    shift = np.array([[16., 41., 33.], [16., 41., 33.], [16., 41., 33.]])
    array_b = np.dot(np.dot(ref, 23.5 * array_a + shift), rot)
    # compute procrustes transformation
    result = orthogonal_2sided(array_a, array_b, translate=True, scale=True, single_transform=False)
    # check transformation array and error
    assert_almost_equal(np.dot(result["array_p"], result["array_p"].T), np.eye(3), decimal=6)
    assert_almost_equal(np.dot(result["array_q"], result["array_q"].T), np.eye(3), decimal=6)
    assert_almost_equal(abs(np.linalg.det(result["array_p"])), 1.0, decimal=6)
    assert_almost_equal(abs(np.linalg.det(result["array_q"])), 1.0, decimal=6)
    # transformation should return zero error
    assert_almost_equal(result["error"], 0, decimal=6)


def test_two_sided_orthogonal_translate_scale_rotate_reflect_3by3():
    r"""Test 2sided orthogonal by a another 3by3 array with translation, rotation and reflection."""
    # define an arbitrary array
    array_a = np.array([[141.58, 315.25, 524.14], [253.25, 255.52, 357.51], [358.2, 131.6, 135.59]])
    # define rotation and reflection arrays
    rot = make_rotation_array(17.54 * np.pi / 6.89)
    ref = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    # define array_b by transforming scaled-and-translated array_a
    shift = np.array([[146.56, 441.67, 343.56], [146.56, 441.67, 343.56], [146.56, 441.67, 343.56]])
    array_b = np.dot(np.dot(ref, 79.89 * array_a + shift), rot)
    # compute procrustes transformation
    result = orthogonal_2sided(array_a, array_b, translate=True, scale=True, single_transform=False)
    # check transformation array and error
    assert_almost_equal(np.dot(result["array_p"], result["array_p"].T), np.eye(3), decimal=6)
    assert_almost_equal(np.dot(result["array_q"], result["array_q"].T), np.eye(3), decimal=6)
    assert_almost_equal(abs(np.linalg.det(result["array_p"])), 1.0, decimal=6)
    assert_almost_equal(abs(np.linalg.det(result["array_q"])), 1.0, decimal=6)
    assert_almost_equal(result["error"], 0, decimal=6)


def test_two_sided_orthogonal_single_transformation_identical():
    r"""Test 2sided orthogonal with identical arrays."""
    # define an arbitrary symmetric array
    array_a = np.array([[2, 5, 4, 1], [5, 3, 1, 2], [8, 9, 1, 0], [1, 5, 6, 7]])
    array_a = np.dot(array_a, array_a.T)
    array_b = np.copy(array_a)

    result = orthogonal_2sided(array_a, array_b, single_transform=True)
    # check transformation array and error
    assert_almost_equal(np.dot(result["array_u"], result["array_u"].T), np.eye(4), decimal=8)
    assert_almost_equal(abs(result["array_u"]), np.eye(4), decimal=8)
    assert_almost_equal(abs(np.linalg.det(result["array_u"])), 1.0, decimal=8)
    assert_almost_equal(result["error"], 0, decimal=8)


def test_two_sided_orthogonal_single_transformation_rot_reflect_padded():
    r"""Test 2sided orthogonal by array with translation, rotation, reflection and zero padding."""
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

    # check transformation array and error.
    result = orthogonal_2sided(array_a, array_b, single_transform=True)
    assert_almost_equal(np.dot(result["array_u"], result["array_u"].T), np.eye(3), decimal=8)
    assert_almost_equal(abs(np.linalg.det(result["array_u"])), 1.0, decimal=8)
    assert_almost_equal(result["error"], 0, decimal=8)


def test_two_sided_orthogonal_single_transformation_scale_3by3():
    r"""Test 2sided orthogonal by 3by3 array with translation and scaling."""
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
    array_b = np.dot(np.dot(trans.T, 6.9 * array_a), trans)

    # check transformation array and error
    result = orthogonal_2sided(array_a, array_b, translate=False, scale=True,
                               single_transform=True)
    assert_almost_equal(np.dot(result["array_u"], result["array_u"].T), np.eye(3), decimal=8)
    assert_almost_equal(abs(np.linalg.det(result["array_u"])), 1.0, decimal=8)
    assert_almost_equal(result["error"], 0, decimal=8)


def test_two_sided_orthogonal_single_transformation_scale_rot_ref_2by2():
    r"""Test 2sided orthogonal by 3by3 array with scaling, rotation and reflection."""
    # define an arbitrary symmetric array
    array_a = np.array([[124.72, 147.93], [120.5, 59.41]])
    array_a = np.dot(array_a, array_a.T)
    # define transformation composed of rotation and reflection
    theta = 5.5 * np.pi / 6.5
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    ref = np.array([[1., 0.], [0., -1.]])
    trans = np.dot(ref, rot)
    # define array_b by transforming scaled array_a
    array_b = np.dot(np.dot(trans.T, 88.89 * array_a), trans)

    # check transformation array and error
    result = orthogonal_2sided(array_a, array_b, translate=False, scale=True,
                               single_transform=True)
    assert_almost_equal(np.dot(result["array_u"], result["array_u"].T), np.eye(2), decimal=8)
    assert_almost_equal(abs(np.linalg.det(result["array_u"])), 1.0, decimal=8)
    assert_almost_equal(result["error"], 0, decimal=8)


def test_two_sided_orthogonal_single_transformation_scale_rot_ref_3by3():
    r"""Test 2sided orthogonal by 3by3 array with single translation, scling and rotation."""
    # define an arbitrary symmetric array
    array_a = (np.random.rand(3, 3) * 100).astype(int)
    array_a = np.dot(array_a, array_a.T)
    # define transformation composed of rotation and reflection
    theta = 5.7 * np.pi / 21.95
    rot = np.array([[np.cos(theta), -np.sin(theta), 0.],
                    [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    ref = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
    trans = np.dot(ref, rot)
    # define array_b by transforming scaled array_a
    array_b = np.dot(np.dot(trans.T, 6.9 * array_a), trans)

    # check transformation array and error
    result = orthogonal_2sided(array_a, array_b, translate=False, scale=True,
                               single_transform=True)
    assert_almost_equal(np.dot(result["array_u"], result["array_u"].T), np.eye(3), decimal=8)
    assert_almost_equal(abs(np.linalg.det(result["array_u"])), 1.0, decimal=8)
    assert_almost_equal(result["error"], 0, decimal=8)


def test_two_sided_orthogonal_single_transformation_random_orthogonal():
    r"""Test 2sided orthogonal by 3by3 array."""
    # define random array
    array_a = np.array([[0, 5, 8, 6], [5, 0, 5, 1],
                        [8, 5, 0, 2], [6, 1, 2, 0]])
    ortho = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0]])
    array_b = np.dot(np.dot(ortho.T, array_a), ortho)
    # check transformation array and error
    result = orthogonal_2sided(array_a, array_b, single_transform=True)
    assert_almost_equal(np.dot(result["array_u"], result["array_u"].T), np.eye(4), decimal=8)
    assert_almost_equal(abs(np.linalg.det(result["array_u"])), 1.0, decimal=8)
    assert_almost_equal(result["error"], 0, decimal=8)
