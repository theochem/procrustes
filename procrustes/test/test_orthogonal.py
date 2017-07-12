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
from procrustes import OrthogonalProcrustes
from numpy.testing import assert_almost_equal, assert_equal


def test_procrustes_orthogonal_identical():
    # case of identical square arrays
    array_a = np.arange(9).reshape(3, 3)
    proc = OrthogonalProcrustes(array_a, array_a)
    # check transformation array is identity
    assert_almost_equal(proc.array_a, array_a, decimal=6)
    assert_almost_equal(proc.array_b, array_a, decimal=6)
    assert_almost_equal(proc.array_u, np.eye(3), decimal=6)
    assert_almost_equal(proc.error, 0., decimal=6)
    # case of identical rectangular arrays (2 by 4)
    array_a = np.array([[1, 5, 6, 7], [1, 2, 9, 4]])
    proc = OrthogonalProcrustes(array_a, array_a)
    assert_almost_equal(proc.array_a, array_a, decimal=6)
    assert_almost_equal(proc.array_b, array_a, decimal=6)
    assert_equal(proc.array_u.shape, (4, 4))
    # assert_almost_equal(proc.array_u, np.eye(4), decimal=6)
    assert_almost_equal(proc.error, 0., decimal=6)
    # case of identical rectangular arrays (5 by 3)
    array_a = np.arange(15).reshape(5, 3)
    proc = OrthogonalProcrustes(array_a, array_a)
    assert_almost_equal(proc.array_a, array_a, decimal=6)
    assert_almost_equal(proc.array_b, array_a, decimal=6)
    assert_equal(proc.array_u.shape, (3, 3))
    assert_almost_equal(proc.array_u, np.eye(3), decimal=6)
    assert_almost_equal(proc.error, 0., decimal=6)


def test_procrustes_rotation_square():
    # square array
    array_a = np.arange(4).reshape(2, 2)
    # rotation by 90 degree
    array_b = np.array([[1, 0], [3, -2]])
    proc = OrthogonalProcrustes(array_a, array_b)
    assert_almost_equal(proc.array_a, array_a, decimal=6)
    assert_almost_equal(proc.array_b, array_b, decimal=6)
    assert_almost_equal(proc.array_u, np.array([[0., -1.], [1., 0.]]), decimal=6)
    assert_almost_equal(proc.error, 0., decimal=6)
    # rotation by 180 degree
    array_b = - array_a
    proc = OrthogonalProcrustes(array_a, array_b)
    assert_almost_equal(proc.array_a, array_a, decimal=6)
    assert_almost_equal(proc.array_b, array_b, decimal=6)
    assert_almost_equal(proc.array_u, np.array([[-1., 0.], [0., -1.]]), decimal=6)
    assert_almost_equal(proc.error, 0., decimal=6)
    # rotation by 270 degree
    array_b = np.array([[-1, 0], [-3, 2]])
    proc = OrthogonalProcrustes(array_a, array_b)
    assert_almost_equal(proc.array_a, array_a, decimal=6)
    assert_almost_equal(proc.array_b, array_b, decimal=6)
    assert_almost_equal(proc.array_u, np.array([[0., 1.], [-1., 0.]]), decimal=6)
    assert_almost_equal(proc.error, 0., decimal=6)
    # rotation by 45 degree
    rotation = 0.5 * np.sqrt(2) * np.array([[1, -1], [1, 1]])
    array_b = np.dot(array_a, rotation)
    proc = OrthogonalProcrustes(array_a, array_b)
    assert_almost_equal(proc.array_a, array_a, decimal=6)
    assert_almost_equal(proc.array_b, array_b, decimal=6)
    assert_almost_equal(proc.array_u, rotation, decimal=6)
    assert_almost_equal(proc.error, 0., decimal=6)
    # rotation by 30 degree
    theta = np.pi / 6
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    array_b = np.dot(array_a, rotation)
    proc = OrthogonalProcrustes(array_a, array_b)
    assert_almost_equal(proc.array_a, array_a, decimal=6)
    assert_almost_equal(proc.array_b, array_b, decimal=6)
    assert_almost_equal(proc.array_u, rotation, decimal=6)
    assert_almost_equal(proc.error, 0., decimal=6)
    # rotation by 72 degree
    theta = 1.25664
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    array_b = np.dot(array_a, rotation)
    proc = OrthogonalProcrustes(array_a, array_b)
    assert_almost_equal(proc.array_a, array_a, decimal=6)
    assert_almost_equal(proc.array_b, array_b, decimal=6)
    assert_almost_equal(proc.array_u, rotation, decimal=6)
    assert_almost_equal(proc.error, 0., decimal=6)


def test_procrustes_reflection_square():
    # square array
    array_a = np.array([[2.0, 0.1], [0.5, 3.0]])
    # reflection through origin
    array_b = - array_a
    proc = OrthogonalProcrustes(array_a, array_b)
    assert_almost_equal(proc.array_a, array_a, decimal=6)
    assert_almost_equal(proc.array_b, array_b, decimal=6)
    assert_almost_equal(proc.array_u, np.array([[-1, 0], [0, -1]]), decimal=6)
    assert_almost_equal(proc.error, 0., decimal=6)
    # reflection in the x-axis
    array_b = np.array([[2.0, -0.1], [0.5, -3.0]])
    proc = OrthogonalProcrustes(array_a, array_b)
    assert_almost_equal(proc.array_a, array_a, decimal=6)
    assert_almost_equal(proc.array_b, array_b, decimal=6)
    assert_almost_equal(proc.array_u, np.array([[1, 0], [0, -1]]), decimal=6)
    assert_almost_equal(proc.error, 0., decimal=6)
    # reflection in the y-axis
    array_b = np.array([[-2.0, 0.1], [-0.5, 3.0]])
    proc = OrthogonalProcrustes(array_a, array_b)
    assert_almost_equal(proc.array_a, array_a, decimal=6)
    assert_almost_equal(proc.array_b, array_b, decimal=6)
    assert_almost_equal(proc.array_u, np.array([[-1, 0], [0, 1]]), decimal=6)
    assert_almost_equal(proc.error, 0., decimal=6)
    # reflection in the line y=x
    array_b = np.array([[0.1, 2.0], [3.0, 0.5]])
    proc = OrthogonalProcrustes(array_a, array_b)
    assert_almost_equal(proc.array_a, array_a, decimal=6)
    assert_almost_equal(proc.array_b, array_b, decimal=6)
    assert_almost_equal(proc.array_u, np.array([[0, 1], [1, 0]]), decimal=6)
    assert_almost_equal(proc.error, 0., decimal=6)


def test_procrustes_shifted():
    # square array
    array_a = np.array([[3.5, 0.1, 7.0], [0.5, 2.0, 1.0], [8.1, 0.3, 0.7]])
    expected_a = array_a - np.mean(array_a, axis=0)
    # constant shift
    array_b = array_a + 4.1
    proc = OrthogonalProcrustes(array_a, array_b, translate=True)
    assert_almost_equal(proc.array_a, expected_a, decimal=6)
    assert_almost_equal(proc.array_b, expected_a, decimal=6)
    assert_almost_equal(proc.array_u, np.eye(3), decimal=6)
    assert_almost_equal(proc.error, 0., decimal=6)
    # different shift along each axis
    array_b = array_a + np.array([0, 3.2, 5.0])
    proc = OrthogonalProcrustes(array_a, array_b, translate=True)
    assert_almost_equal(proc.array_a, expected_a, decimal=6)
    assert_almost_equal(proc.array_b, expected_a, decimal=6)
    assert_almost_equal(proc.array_u, np.eye(3), decimal=6)
    assert_almost_equal(proc.error, 0., decimal=6)
    # rectangular (2 by 3)
    array_a = np.array([[1, 2, 3], [7, 9, 5]])
    expected_a = array_a - np.array([4., 5.5, 4.])
    # constant shift
    array_b = array_a + 0.71
    proc = OrthogonalProcrustes(array_a, array_b, translate=True)
    assert_almost_equal(proc.array_a, expected_a, decimal=6)
    assert_almost_equal(proc.array_b, expected_a, decimal=6)
    assert_almost_equal(proc.error, 0., decimal=6)
    # different shift along each axis
    array_b = array_a + np.array([0.3, 7.1, 4.2])
    proc = OrthogonalProcrustes(array_a, array_b, translate=True)
    assert_almost_equal(proc.array_a, expected_a, decimal=6)
    assert_almost_equal(proc.array_b, expected_a, decimal=6)
    assert_equal(proc.array_u.shape, (3, 3))
    assert_almost_equal(proc.error, 0., decimal=6)


def test_procrustes_rotation_translation():
    # initial arrays
    array_a = np.array([[-7.3, 2.8], [-7.1, -0.2], [4.0, 1.4], [1.3, 0]])
    # rotation by 20 degree & reflection in the x-axis
    theta = 0.34907
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    reflection = np.array([[1, 0], [0, -1]])
    array_b = np.dot(array_a, np.dot(rotation, reflection))
    # procrustes without translation and scaling
    proc = OrthogonalProcrustes(array_a, array_b)
    assert_almost_equal(proc.array_a, array_a, decimal=6)
    assert_almost_equal(proc.array_b, array_b, decimal=6)
    assert_almost_equal(proc.array_u, np.dot(rotation, reflection), decimal=6)
    assert_almost_equal(proc.error, 0., decimal=6)
    # procrustes with translation
    proc = OrthogonalProcrustes(array_a, array_b, translate=True)
    assert_almost_equal(proc.array_a, array_a - np.mean(array_a, axis=0), decimal=6)
    assert_almost_equal(proc.array_b, array_b - np.mean(array_b, axis=0), decimal=6)
    assert_almost_equal(proc.array_u, np.dot(rotation, reflection), decimal=6)
    assert_almost_equal(proc.error, 0., decimal=6)
    # procrustes with translation and scaling
    proc = OrthogonalProcrustes(array_a, array_b, translate=True, scale=True)
    assert_almost_equal(proc.array_u, np.dot(rotation, reflection), decimal=6)
    assert_almost_equal(proc.error, 0., decimal=6)


def test_procrustes_rotation_translate_scale():
    # initial arrays
    array_a = np.array([[5.1, 0], [-1.1, 4.8], [3.9, 7.3], [9.1, 6.3]])
    # rotation by 68 degree & reflection in the Y=X
    theta = 1.18682
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    reflection = np.array([[0, 1], [1, 0]])
    array_b = np.dot(4 * array_a + 3.0, np.dot(rotation, reflection))
    # procrustes with translation and scaling
    proc = OrthogonalProcrustes(array_a, array_b, translate=True, scale=True)
    assert_almost_equal(proc.array_u, np.dot(rotation, reflection), decimal=6)
    assert_almost_equal(proc.error, 0., decimal=6)
