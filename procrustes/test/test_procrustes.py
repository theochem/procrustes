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
"""
"""


import numpy as np
from procrustes import *



# def test_procrustes_orthogonal_translate_scale2():
#     array = np.array([[1, 4], [7, 9]])
#     # an arbitrary rotational transformation
#     theta = np.pi / 2
#     rot_trans = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

#     # Define an arbitrary reflection transformation
#     refl_trans = np.array([[1, 0], [0, -1]])

#     # Define the orthogonal transformation (composition of rotation and reflection transformations)
#     ortho_trans = np.dot(rot_trans, refl_trans)
#     assert(abs(np.dot(ortho_trans, ortho_trans.T) - np.eye(2)) < 1.e-8).all()

#     # Define the orthogonally transformed original array
#     array_ortho_transf = np.dot(array, ortho_trans)
#     # Create (arbitrary) pads to add onto the orthogonally transformed input array, array_ortho_transf
#     m, n = array_ortho_transf.shape
#     arb_pad_col = 3
#     arb_pad_row = 7
#     pad_vertical = np.zeros((m, arb_pad_col))
#     pad_horizontal = np.zeros((arb_pad_row, n+arb_pad_col))
#     array_ortho_transf = np.concatenate((array_ortho_transf, pad_vertical), axis=1)
#     array_ortho_transf = np.concatenate((array_ortho_transf, pad_horizontal), axis=0)

#     # Proceed with orthogonal procrustes analysis
#     ortho = OrthogonalProcrustes(array, array_ortho_transf)
#     # Assert that the analysis returns zero error and is correct
#     assert ortho.error < 1.e-10



def test_two_sided_orthogonal():
    # Define arbitrary array
    array = np.array([[1., 4., 6.], [6., 1., 4.], [7., 8., 1.]])
    # Define two arbitrary rotation transformations
    theta1 = np.pi / 5.
    rot_array1 = np.array([[np.cos(theta1), -np.sin(theta1), 0.], [np.sin(theta1), np.cos(theta1), 0.], [0., 0., 1.]])
    assert(abs(np.dot(rot_array1, rot_array1.T) - np.eye(3)) < 1.e-8).all()
    assert(abs(np.linalg.det(rot_array1) - 1.) < 1.e-10)
    theta2 = 12. * np.pi / 9.
    rot_array2 = np.array([[np.cos(theta2), -np.sin(theta2), 0.], [np.sin(theta2), np.cos(theta2), 0.], [0., 0., 1.]])
    assert(abs(np.dot(rot_array2, rot_array2.T) - np.eye(3)) < 1.e-8).all()
    assert(abs(np.linalg.det(rot_array2) - 1) < 1.e-10)
    # Define two arbitrary reflection transformations
    refl_array1 = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    assert(abs(np.linalg.det(refl_array1) + 1) < 1.e-8)
    assert(abs(np.dot(refl_array1, refl_array1.T) - np.eye(3)) < 1.e-8).all()
    refl_array2 = 1./3. * np.array([[1., -2., -2.], [-2., 1., -2.], [-2., -2., 1.]])
    assert(abs(np.linalg.det(refl_array2) + 1) < 1.e-8)
    assert(abs(np.dot(refl_array2, refl_array2.T) - np.eye(3)) < 1.e-8).all()
    # Define two orthogonal transformations
    ortho_trans1 = np.dot(refl_array1, rot_array1)
    assert(abs(np.dot(ortho_trans1, ortho_trans1.T) - np.eye(3)) < 1.e-10).all()
    ortho_trans2 = np.dot(rot_array2, refl_array2)
    assert(abs(np.dot(ortho_trans2, ortho_trans2.T) - np.eye(3)) < 1.e-10).all()
    # Define the two-sided orthogonally transformed original array
    array_ortho_transformed = np.dot(np.dot(ortho_trans1.T, array), ortho_trans2)
    # Create (arbitrary) pads to add onto the two-sided orthogonally transformed input array, array_ortho_transformed
    m, n = array_ortho_transformed.shape
    arb_pad_col = 8
    arb_pad_row = 4
    pad_vertical = np.zeros((m, arb_pad_col))
    pad_horizontal = np.zeros((arb_pad_row, n+arb_pad_col))
    array_ortho_transformed = np.concatenate((array_ortho_transformed, pad_vertical), axis=1)
    array_ortho_transformed = np.concatenate((array_ortho_transformed, pad_horizontal), axis=0)
    # Proceed with two-sided procrustes analysis
    twosided_ortho = TwoSidedOrthogonalProcrustes(array, array_ortho_transformed)
    u1, u2, array_transformed, error = twosided_ortho.calculate()
    assert((np.dot(u1, u1.T) - np.eye(3)) < 1.e-10).all()
    assert((np.dot(u2, u2.T) - np.eye(3)) < 1.e-10).all()
    # The transformation should return zero error
    assert error < 1.e-10
    """
    This test verifies that two-sided orthogonal procrustes analysis
    works when the input arrays are identical
    """
    # Define an arbitrary array
    array_a = np.array([[2, 5, 4, 1], [5, 3, 1, 2], [8, 9, 1, 0], [1, 5, 6, 7]])
    # Match arbitrary array (above) to itself
    array_b = np.array([[2, 5, 4, 1], [5, 3, 1, 2], [8, 9, 1, 0], [1, 5, 6, 7]])
    assert(abs(array_a - array_b) < 1.e-8).all()
    # Proceed with two-sided orthogonal procrustes analysis
    twosided_ortho = TwoSidedOrthogonalProcrustes(array_a, array_b)
    u1, u2, array_transformed, error = twosided_ortho.calculate()
    assert((np.dot(u1, u1.T) - np.eye(4)) < 1.e-10).all()
    assert((np.dot(u2, u2.T) - np.eye(4)) < 1.e-10).all()
    # The transformation should return zero error
    assert error < 1.e-8
    """
    This test verifies that the two-sided orthogonal procrustes analysis is capable of matching an input array
    to itself after it undergoes translation, scaling, and orthogonal transformations
    """
    # Define an arbitrary array.
    array_a = np.array([[1, 3, 5], [3, 5, 7], [8, 11, 15]])
    # Define an arbitrary translation
    shift = np.array([[16., 41., 33.], [16., 41., 33.], [16., 41., 33.]])
    # Translate and scale the initial array
    array_b = 23.5 * array_a + shift
    # Define an arbitrary rotation transformations
    theta = 1.8 * np.pi / 34.
    rot_array = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    assert(abs(np.dot(rot_array1, rot_array1.T) - np.eye(3)) < 1.e-8).all()
    assert(abs(np.linalg.det(rot_array1) - 1) < 1.e-10)
    # Define an arbitrary reflection transformations
    refl_array = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    assert(abs(np.linalg.det(refl_array1) + 1) < 1.e-8)
    assert(abs(np.dot(refl_array1, refl_array1.T) - np.eye(3)) < 1.e-8).all()
    # Compute the two-sided orthogonally transformed original array
    array_twosided_ortho_transf = np.dot(np.dot(refl_array, array_b), rot_array)
    # Proceed with two-sided orthogonal procrustes analysis
    twosided_ortho = TwoSidedOrthogonalProcrustes(array_a, array_twosided_ortho_transf, translate=True, scale=True)
    u1, u2, a_transformed, error = twosided_ortho.calculate()
    assert((np.dot(u1, u1.T) - np.eye(3)) < 1.e-10).all()
    assert((np.dot(u2, u2.T) - np.eye(3)) < 1.e-10).all()
    # The transformation should return zero error
    assert error < 1.e-8

    # Define an arbitrary array.
    array_a = np.array([[141.58, 315.25, 524.14], [253.25, 255.52, 357.51], [358.2, 131.6, 135.59]])
    # Define an arbitrary translation
    shift = np.array([[146.56, 441.67, 343.56], [146.56, 441.67, 343.56], [146.56, 441.67, 343.56]])
    # Translate and scale the initial array
    array_b = 79.89 * array_a + shift
    # Define an arbitrary rotation transformations
    theta = 17.54 * np.pi / 6.89
    rot_array = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    assert(abs(np.dot(rot_array1, rot_array1.T) - np.eye(3)) < 1.e-8).all()
    assert(abs(np.linalg.det(rot_array1) - 1) < 1.e-10)
    # Define an arbitrary reflection transformations
    refl_array = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    assert(abs(np.linalg.det(refl_array1) + 1) < 1.e-8)
    assert(abs(np.dot(refl_array1, refl_array1.T) - np.eye(3)) < 1.e-8).all()
    # Compute the two-sided orthogonally transformed original array
    array_twosided_ortho_transf = np.dot(np.dot(refl_array, array_b), rot_array)
    # Proceed with two-sided orthogonal procrustes analysis
    twosided_ortho = TwoSidedOrthogonalProcrustes(array_a, array_twosided_ortho_transf, translate=True, scale=True)
    u1, u2, a_transformed, error = twosided_ortho.calculate()
    assert((np.dot(u1, u1.T) - np.eye(3)) < 1.e-10).all()
    assert((np.dot(u2, u2.T) - np.eye(3)) < 1.e-10).all()
    # The transformation should return zero error
    assert error < 1.e-8

    # Define an arbitrary array.
    array_a = np.array([[41.8, 15.5, 24.4], [53.5, 55.2, 57.1], [58.2, 31.6, 35.9]])
    # Define an arbitrary translation
    shift = np.array([[46.6, 41.7, 43.6], [46.6, 41.7, 43.6], [46.6, 41.7, 43.6]])
    # Translate and scale the initial array
    array_b = 79.89 * array_a + shift
    # Define an arbitrary rotation transformations
    theta = 17.54 * np.pi / 6.89
    rot_array = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    assert(abs(np.dot(rot_array1, rot_array1.T) - np.eye(3)) < 1.e-8).all()
    assert(abs(np.linalg.det(rot_array1) - 1) < 1.e-10)
    # Define an arbitrary reflection transformations
    refl_array = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    assert(abs(np.linalg.det(refl_array1) + 1) < 1.e-8)
    assert(abs(np.dot(refl_array1, refl_array1.T) - np.eye(3)) < 1.e-8).all()
    # Compute the two-sided orthogonally transformed original array
    array_twosided_ortho_transf = np.dot(np.dot(refl_array, array_b), rot_array)
    # Proceed with two-sided orthogonal procrustes analysis
    twosided_ortho = TwoSidedOrthogonalProcrustes(array_a, array_twosided_ortho_transf, translate=True, scale=True)
    u1, u2, a_transformed, error = twosided_ortho.calculate()
    assert((np.dot(u1, u1.T) - np.eye(3)) < 1.e-10).all()
    assert((np.dot(u2, u2.T) - np.eye(3)) < 1.e-10).all()
    # The transformation should return zero error
    assert error < 1.e-8


# def test_two_sided_orthogonal_single_transformation():
#     # Define arbitrary symmetric array
#     array = np.array([[5, 2, 1], [4, 6, 1], [1, 6, 3]])
#     sym_array = np.dot(array, array.T)
#     # Define an arbitrary rotation transformation
#     theta = 16. * np.pi / 5.
#     rot_array = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
#     assert(abs(np.linalg.det(rot_array) - 1) < 1.e-8)
#     assert((np.dot(rot_array, rot_array.T) - np.eye(3)) < 1.e-8).all()
#     # Define an arbitrary reflection transformation
#     refl_array = 1./3 * np.array([[1, -2, -2], [-2, 1, -2], [-2, -2, 1]])
#     assert((np.dot(refl_array, refl_array.T) - np.eye(3)) < 1.e-8).all()
#     assert(abs(np.linalg.det(refl_array) + 1) < 1.e-8)
#     # Define the orthogonal transformation
#     orth_transf = np.dot(rot_array, refl_array)
#     assert(abs(np.dot(orth_transf, orth_transf.T) - np.eye(3)) < 1.e-8).all()
#     # Define the two-sided orthogonal transformation of the original array
#     array_singleortho_transf = np.dot(np.dot(orth_transf.T, sym_array), orth_transf)
#     assert(abs(array_singleortho_transf - array_singleortho_transf.T) < 1.e-10).all()
#     # Create (arbitrary) pads to add onto the two-sided single transformation
#     # orthogonally transformed input array, array_singleortho_transf
#     m, n = array_singleortho_transf.shape
#     arb_pad_col = 7
#     arb_pad_row = 4
#     pad_vertical = np.zeros((m, arb_pad_col))
#     pad_horizontal = np.zeros((arb_pad_row, n+arb_pad_col))
#     array_singleortho_transf = np.concatenate((array_singleortho_transf, pad_vertical), axis=1)
#     array_singleortho_transf = np.concatenate((array_singleortho_transf, pad_horizontal), axis=0)
#     # Proceed with two-sided single transformation procrustes analysis
#     twosided_single_ortho = TwoSidedOrthogonalSingleTransformationProcrustes(sym_array, array_singleortho_transf)
#     u_approx, u_exact, array_transformed_approx, array_transformed_exact, error_approx, error_best, \
#         = twosided_single_ortho.calculate(return_u_approx=True, return_u_best=True)
#     assert(abs(np.dot(u_approx, u_approx.T) - np.eye(3)) < 1.e-8).all()
#     assert(abs(np.dot(u_exact, u_exact.T) - np.eye(3)) < 1.e-8).all()
#     assert error_best < 1.e-10
#     """
#     This test verifies that two-sided single transformation orthogonal procrustes analysis
#     works when the input arrays are identical.
#     """
#     # Define an arbitrary symmetric array
#     sym_part = np.array([[2, 5, 4, 1], [5, 3, 1, 2], [8, 9, 1, 0], [1, 5, 6, 7]])
#     sym_array_a = np.dot(sym_array, sym_array.T)
#     # Match arbitrary array (above) to itself
#     sym_array_b = sym_array_a
#     assert(abs(sym_array_a - sym_array_b) < 1.e-8).all()
#     # Proceed with two-sided single transformation orthogonal procrustes analysis
#     twosided_single_ortho = TwoSidedOrthogonalSingleTransformationProcrustes(sym_array_a, sym_array_b)
#     u_approx, u_exact, array_transformed_approx, array_transformed_exact, error_approx, error_best,\
#         = twosided_single_ortho.calculate(return_u_approx=True, return_u_best=True)
#     assert(abs(np.dot(u_approx, u_approx.T) - np.eye(3)) < 1.e-8).all()
#     assert(abs(np.dot(u_exact, u_exact.T) - np.eye(3)) < 1.e-8).all()
#     assert error_best < 1.e-10
#     """
#     This test verifies that the two-sided single transformation orthogonal procrustes analysis is capable of
#     matching an input array to itself after it undergoes translation, scaling, and orthogonal transformations.
#     """

#     # Define an arbitrary symmetric array.
#     sym_part = np.array([[12.43, 16.15, 17.61], [11.4, 21.5, 16.7], [16.4, 19.4, 14.9]])
#     array_a = np.dot(sym_part, sym_part.T)
#     assert(abs(array_a - array_a.T) < 1.e-10).all()
#     # Define an arbitrary translation. The shift must preserve the symmetry of the array
#     # (i.e. the shift must too be symmetric)
#     sym_shift = np.array([[6.7, 6.7, 6.7], [6.7, 6.7, 6.7], [6.7, 6.7, 6.7]])
#     assert(abs(sym_shift - sym_shift.T) < 1.e-10).all()
#     # Translate and scale the initial array
#     array_b = 6.9 * array_a + sym_shift
#     # Define an arbitrary rotation transformations
#     theta = 12. * np.pi / 6.3
#     rot_array = np.array([[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
#     assert(abs(np.dot(rot_array, rot_array.T) - np.eye(3)) < 1.e-8).all()
#     assert(abs(np.linalg.det(rot_array) - 1.) < 1.e-10)
#     # Define an arbitrary reflection transformations
#     refl_array = 1./3 * np.array([[1, -2, -2], [-2, 1, -2], [-2, -2, 1]])
#     # refl_array = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
#     assert(abs(np.linalg.det(refl_array) + 1) < 1.e-8)
#     assert(abs(np.dot(refl_array, refl_array.T) - np.eye(3)) < 1.e-8).all()
#     # Define the single orthogonal transformation
#     # single_ortho_transf = np.dot(rot_array, refl_array)
#     single_ortho_transf = np.dot(refl_array, rot_array)
#     assert(abs(np.dot(single_ortho_transf, single_ortho_transf.T) - np.eye(3)) < 1.e-8).all()
#     # Define the sinlge othogonal transformation of the original array
#     array_singleortho_transf = np.dot(np.dot(single_ortho_transf.T, array_b), single_ortho_transf)
#     assert(abs(array_singleortho_transf - array_singleortho_transf.T) < 1.e-10).all()
#     # Proceed with two-sided single transformation orthogonal procrustes analysis
#     twosided_single_ortho = TwoSidedOrthogonalSingleTransformationProcrustes(array_a, array_singleortho_transf,
#                                                                              translate=True, scale=True)
#     u_approx, u_exact, array_transformed_approx, array_transformed_exact, error_approx, error_best, \
#          = twosided_single_ortho.calculate(return_u_approx=True, return_u_best=True)
#     assert(abs(np.dot(u_approx, u_approx.T) - np.eye(3)) < 1.e-8).all()
#     assert(abs(np.dot(u_exact, u_exact.T) - np.eye(3)) < 1.e-8).all()

#     # Define an arbitrary symmetric array.
#     sym_part = np.array([[67.93, 147.93, 32.78], [21.59, 59.41, 79.90], [58.4, 49.4, 85.9]])
#     array_a = np.dot(sym_part, sym_part.T)
#     assert(abs(array_a - array_a.T) < 1.e-10).all()
#     # Define an arbitrary translation. The shift must preserve the symmetry of the array
#     # (i.e. the shift must too be symmetric)
#     sym_shift = np.array([[26.98, 26.98, 26.98], [26.98, 26.98, 26.98], [26.98, 26.98, 26.98]])
#     assert(abs(sym_shift - sym_shift.T) < 1.e-10).all()
#     # Translate and scale the initial array
#     array_b = 6.9 * array_a + sym_shift
#     # Define an arbitrary rotation transformations
#     theta = 68.54 * np.pi / 23.41
#     rot_array = np.array([[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
#     assert(abs(np.dot(rot_array, rot_array.T) - np.eye(3)) < 1.e-8).all()
#     assert(abs(np.linalg.det(rot_array) - 1.) < 1.e-10)
#     # Define an arbitrary reflection transformations
#     # refl_array = 1./3 * np.array([[1, -2, -2], [-2, 1, -2], [-2, -2, 1]])
#     refl_array = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
#     assert(abs(np.linalg.det(refl_array) + 1) < 1.e-8)
#     assert(abs(np.dot(refl_array, refl_array.T) - np.eye(3)) < 1.e-8).all()
#     # Define the single orthogonal transformation
#     single_ortho_transf = np.dot(refl_array, rot_array)
#     assert(abs(np.dot(single_ortho_transf, single_ortho_transf.T) - np.eye(3)) < 1.e-8).all()
#     # Define the sinlge othogonal transformation of the original array
#     array_singleortho_transf = np.dot(np.dot(single_ortho_transf.T, array_b), single_ortho_transf)
#     assert(abs(array_singleortho_transf - array_singleortho_transf.T) < 1.e-10).all()
#     # Proceed with two-sided single transformation orthogonal procrustes analysis
#     twosided_single_ortho = TwoSidedOrthogonalSingleTransformationProcrustes(array_a, array_singleortho_transf,
#                                                                              translate=True, scale=True)
#     u_approx, u_exact, array_transformed_approx, array_transformed_exact, error_approx, error_best,\
#          = twosided_single_ortho.calculate(return_u_approx=True, return_u_best=True)
#     assert(abs(np.dot(u_approx, u_approx.T) - np.eye(3)) < 1.e-8).all()
#     assert(abs(np.dot(u_exact, u_exact.T) - np.eye(3)) < 1.e-8).all()

#     # Define an arbitrary symmetric array.
#     sym_part = np.array([[124.72, 147.93], [120.5, 59.41]])
#     array_a = np.dot(sym_part, sym_part.T)
#     assert(abs(array_a - array_a.T) < 1.e-10).all()
#     # Define an arbitrary translation. The shift must preserve the symmetry of the array
#     # (i.e. the shift must too be symmetric)
#     sym_shift = np.array([[45.91, 45.91], [45.91, 45.91]])
#     assert(abs(sym_shift - sym_shift.T) < 1.e-10).all()
#     # Translate and scale the initial array
#     array_b = 88.89 * array_a + sym_shift
#     # Define an arbitrary rotation transformations
#     theta = 43.89* np.pi / 12.43
#     rot_array = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
#     assert(abs(np.dot(rot_array, rot_array.T) - np.eye(2)) < 1.e-8).all()
#     assert(abs(np.linalg.det(rot_array) - 1.) < 1.e-10)
#     # Define an arbitrary reflection transformations
#     refl_array = np.array([[1., 0.], [0., -1.]])
#     assert(abs(np.linalg.det(refl_array) + 1) < 1.e-8)
#     assert(abs(np.dot(refl_array, refl_array.T) - np.eye(2)) < 1.e-8).all()
#     # Define the single orthogonal transformation
#     single_ortho_transf = np.dot(refl_array, rot_array)
#     assert(abs(np.dot(single_ortho_transf, single_ortho_transf.T) - np.eye(2)) < 1.e-8).all()
#     # Define the sinlge othogonal transformation of the original array
#     array_singleortho_transf = np.dot(np.dot(single_ortho_transf.T, array_b), single_ortho_transf)
#     assert(abs(array_singleortho_transf - array_singleortho_transf.T) < 1.e-10).all()
#     # Proceed with two-sided single transformation orthogonal procrustes analysis
#     twosided_single_ortho = TwoSidedOrthogonalSingleTransformationProcrustes(array_a, array_singleortho_transf,
#                                                                              translate=True, scale=True)
#     u_approx, u_exact, array_transformed_approx, array_transformed_exact, error_approx, error_best,\
#          = twosided_single_ortho.calculate(return_u_approx=True, return_u_best=True)
#     assert(abs(np.dot(u_approx, u_approx.T) - np.eye(2)) < 1.e-8).all()
#     assert(abs(np.dot(u_exact, u_exact.T) - np.eye(2)) < 1.e-8).all()


#     # Define an arbitrary symmetric array.
#     sym_part = np.array([[6948.184, 1481.51], [2592.51, 125.25]])
#     array_a = np.dot(sym_part, sym_part.T)
#     assert(abs(array_a - array_a.T) < 1.e-10).all()
#     # Define an arbitrary translation. The shift must preserve the symmetry of the array
#     # (i.e. the shift must too be symmetric)
#     sym_shift = np.array([[5892.5125, 5892.5125], [5892.5125, 5892.5125]])
#     assert(abs(sym_shift - sym_shift.T) < 1.e-10).all()
#     # Translate and scale the initial array
#     array_b = 88.89 * array_a + sym_shift
#     # Define an arbitrary rotation transformations
#     theta = 905.155 * np.pi / 51.65
#     rot_array = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
#     assert(abs(np.dot(rot_array, rot_array.T) - np.eye(2)) < 1.e-8).all()
#     assert(abs(np.linalg.det(rot_array) - 1.) < 1.e-10)
#     # Define an arbitrary reflection transformations
#     refl_array = np.array([[1., 0.], [0., -1.]])
#     assert(abs(np.linalg.det(refl_array) + 1) < 1.e-8)
#     assert(abs(np.dot(refl_array, refl_array.T) - np.eye(2)) < 1.e-8).all()
#     # Define the single orthogonal transformation
#     single_ortho_transf = np.dot(refl_array, rot_array)
#     assert(abs(np.dot(single_ortho_transf, single_ortho_transf.T) - np.eye(2)) < 1.e-8).all()
#     # Define the sinlge othogonal transformation of the original array
#     array_singleortho_transf = np.dot(np.dot(single_ortho_transf.T, array_b), single_ortho_transf)
#     assert(abs(array_singleortho_transf - array_singleortho_transf.T) < 1.e-10).all()
#     # Proceed with two-sided single transformation orthogonal procrustes analysis
#     twosided_single_ortho = TwoSidedOrthogonalSingleTransformationProcrustes(array_a, array_singleortho_transf,
#                                                                              translate=True, scale=True)
#     u_approx, u_exact, array_transformed_approx, array_transformed_exact, error_approx, error_best,\
#          = twosided_single_ortho.calculate(return_u_approx=True, return_u_best=True)
#     assert(abs(np.dot(u_approx, u_approx.T) - np.eye(2)) < 1.e-8).all()
#     assert(abs(np.dot(u_exact, u_exact.T) - np.eye(2)) < 1.e-8).all()


#     # Define an arbitrary symmetric array.
#     sym_part = np.array([[1.5925e-4, 5.7952e-4], [3.5862e-4, 8.4721e-4]])
#     array_a = np.dot(sym_part, sym_part.T)
#     assert(abs(array_a - array_a.T) < 1.e-10).all()
#     # Define an arbitrary translation. The shift must preserve the symmetry of the array
#     # (i.e. the shift must too be symmetric)
#     sym_shift = np.array([[1.5918571985e-5, 1.5918571985e-5], [1.5918571985e-5, 1.5918571985e-5]])
#     assert(abs(sym_shift - sym_shift.T) < 1.e-10).all()
#     # Translate and scale the initial array5.42
#     array_b = 4.524e-4 * array_a + sym_shift
#     # Define an arbitrary rotation transformations
#     theta = 1.49251351895159 * np.pi / 3.58401351558193
#     rot_array = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
#     assert(abs(np.dot(rot_array, rot_array.T) - np.eye(2)) < 1.e-8).all()
#     assert(abs(np.linalg.det(rot_array) - 1.) < 1.e-10)
#     # Define an arbitrary reflection transformations
#     refl_array = np.array([[1., 0.], [0., -1.]])
#     assert(abs(np.linalg.det(refl_array) + 1) < 1.e-8)
#     assert(abs(np.dot(refl_array, refl_array.T) - np.eye(2)) < 1.e-8).all()
#     # Define the single orthogonal transformation
#     single_ortho_transf = np.dot(refl_array, rot_array)
#     assert(abs(np.dot(single_ortho_transf, single_ortho_transf.T) - np.eye(2)) < 1.e-8).all()
#     # Define the sinlge othogonal transformation of the original array
#     array_singleortho_transf = np.dot(np.dot(single_ortho_transf.T, array_b), single_ortho_transf)
#     assert(abs(array_singleortho_transf - array_singleortho_transf.T) < 1.e-10).all()
#     # Proceed with two-sided single transformation orthogonal procrustes analysis
#     twosided_single_ortho = TwoSidedOrthogonalSingleTransformationProcrustes(array_a, array_singleortho_transf,
#                                                                              translate=True, scale=True)
#     u_approx, u_exact, array_transformed_approx, array_transformed_exact, error_approx, error_best, \
#          = twosided_single_ortho.calculate(return_u_approx=True, return_u_best=True)
#     assert(abs(np.dot(u_approx, u_approx.T) - np.eye(2)) < 1.e-8).all()
#     assert(abs(np.dot(u_exact, u_exact.T) - np.eye(2)) < 1.e-8).all()


# def test_two_sided_permutation_single_transformation():
#     # Define arbitrary array
#     sym_part = np.array([[5., 2., 1.], [4., 6., 1.], [1., 6., 3.]])
#     sym_array = np.dot(sym_part, sym_part.T)
#     # Define an arbitrary permutation transformation
#     perm_array = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
#     assert(abs(np.linalg.det(perm_array)) - 1. < 1.e-8)
#     assert(abs([x for x in perm_array.flatten().tolist() if x != 0] - np.ones(3)) < 1.e-8).all()
#     # Define the permuted original array
#     array_permuted = np.dot(np.dot(perm_array.T, sym_array), perm_array)
#     # Create (arbitrary) pads to add onto the permuted input array, array_permuted
#     m, n = array_permuted.shape
#     arb_pad_col = 4
#     arb_pad_row = 8
#     pad_vertical = np.zeros((m, arb_pad_col))
#     pad_horizontal = np.zeros((arb_pad_row, n+arb_pad_col))
#     array_permuted = np.concatenate((array_permuted, pad_vertical), axis=1)
#     array_permuted = np.concatenate((array_permuted, pad_horizontal), axis=0)
#     # Proceed with permutation procrustes analysis
#     twosided_single_perm = TwoSidedPermutationSingleTransformationProcrustes(sym_array, array_permuted)
#     least_error_perm, least_error_array_transformed, min_error =\
#         twosided_single_perm.calculate()
#     assert(abs(np.linalg.det(least_error_perm)) - 1. < 1.e-8)
#     assert(abs([x for x in least_error_perm.flatten().tolist() if x != 0] - np.ones(3)) < 1.e-8).all()
#     # Assert that the analysis returns zero error
#     assert min_error < 1.e-10
#     """
#     This test verifies that permutation procrustes analysis
#     works when the input arrays are identical
#     """
#     # If the input arrays are equivalent, the permutation transformation must be the identity
#     # Define an arbitrary symmetric array
#     sym_part = np.array([[12, 22, 63, 45, 7], [23, 54, 66, 63, 21]])
#     array_a = np.dot(sym_part, sym_part.T)
#     # Match arbitrary array (above) to itself
#     array_b = array_a
#     assert(abs(array_a - array_b) < 1.e-8).all()
#     # Proceed with two sided single-transformation procrustes analysis
#     twosided_single_perm = TwoSidedPermutationSingleTransformationProcrustes(array_a, array_b)
#     least_error_perm, least_error_array_transformed, min_error =\
#         twosided_single_perm.calculate()
#     # Perm_optimum must be a permutation array
#     assert(abs(np.linalg.det(least_error_perm)) - 1. < 1.e-8)
#     assert(abs([x for x in least_error_perm.flatten().tolist() if x != 0] - np.ones(2)) < 1.e-8).all()
#     # The expected permutation-transformation is the 4x4 identity array
#     expected = np.eye(2)
#     # The transformation should return zero error
#     assert min_error < 1.e-8
#     # The transformation must be the 4x4 identity
#     assert(abs(least_error_perm - expected) < 1.e-8).all()


#     """
#     This test verifies that permutation procrustes analysis is capable of matching an input array
#     to itself after it undergoes translation, scaling, and permutation transformations
#     """
#     # Define arbitrary array
#     sym_part = np.array([[5., 2., 1.], [4., 6., 1.], [1., 6., 3.]])
#     sym_array = np.dot(sym_part, sym_part.T)
#     # Define an arbitrary translation. The shift must preserve the symmetry of the array
#     #  (i.e. the shift must too be symmetric)
#     sym_shift = np.array([[3.14, 3.14, 3.14], [3.14, 3.14, 3.14], [3.14, 3.14, 3.14]])
#     assert(abs(sym_shift - sym_shift.T) < 1.e-10).all()
#     # Translate and scale the initial array
#     array_b = 14.7 * sym_array + sym_shift
#     # Define an arbitrary permutation transformation
#     perm_array = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
#     assert(abs(np.linalg.det(perm_array)) - 1. < 1.e-8)
#     assert(abs([x for x in perm_array.flatten().tolist() if x != 0] - np.ones(3)) < 1.e-8).all()
#     # Define the permuted original array
#     array_permuted = np.dot(np.dot(perm_array.T, array_b), perm_array)
#     # Proceed with permutation procrustes analysis
#     twosided_single_perm = TwoSidedPermutationSingleTransformationProcrustes(sym_array, array_permuted,
#                                                                              translate=True, scale=True)
#     least_error_perm, least_error_array_transformed, min_error =\
#         twosided_single_perm.calculate()
#     assert(abs(np.linalg.det(least_error_perm)) - 1. < 1.e-8)
#     assert(abs([x for x in least_error_perm.flatten().tolist() if x != 0] - np.ones(3)) < 1.e-8).all()
#     # Assert that the analysis returns zero error
#     assert min_error < 1.e-10

#     # Define arbitrary array
#     sym_part = np.array([[14.4, 16.2, 36.5, 53.1], [42.4, 43.1, 25.3, 53.1], [11.3, 26.5, 37.2, 21.1],
#                          [35.2, 62.1, 12.12, 21.3]])
#     sym_array = np.dot(sym_part, sym_part.T)
#     # Define an arbitrary translation. The shift must preserve the symmetry of the array
#     #  (i.e. the shift must too be symmetric)
#     sym_shift = np.array([[2.7818, 2.7818, 2.7818, 2.7818], [2.7818, 2.7818, 2.7818, 2.7818],
#                           [2.7818, 2.7818, 2.7818, 2.7818], [2.7818, 2.7818, 2.7818, 2.7818]])
#     assert(abs(sym_shift - sym_shift.T) < 1.e-10).all()
#     # Translate and scale the initial array
#     array_b = 22.4 * sym_array + sym_shift
#     # Define an arbitrary permutation transformation
#     perm_array = np.array([[0., 0., 1., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.], [0., 1., 0., 0.]])
#     assert(abs(np.linalg.det(perm_array)) - 1. < 1.e-8)
#     assert(abs([x for x in perm_array.flatten().tolist() if x != 0] - np.ones(4)) < 1.e-8).all()
#     # Define the permuted original array
#     array_permuted = np.dot(np.dot(perm_array.T, array_b), perm_array)
#     # Proceed with permutation procrustes analysis
#     twosided_single_perm = TwoSidedPermutationSingleTransformationProcrustes(sym_array, array_permuted,
#                                                                              translate=True, scale=True)
#     least_error_perm, least_error_array_transformed, min_error = \
#         twosided_single_perm.calculate()
#     assert(abs(np.linalg.det(least_error_perm)) - 1. < 1.e-8)
#     assert(abs([x for x in least_error_perm.flatten().tolist() if x != 0] - np.ones(4)) < 1.e-8).all()
#     # Assert that the analysis returns zero error
#     assert min_error < 1.e-10

#     # Define arbitrary array
#     sym_part = np.array([[24.4, 18.22, 16.5, 53.1], [12.4, 53.1, 64.3, 38.1], [31.3, 45.5, 67.2, 21.1],
#                          [56.2, 43.1, 25.12, 53.3]])
#     sym_array = np.dot(sym_part, sym_part.T)
#     # Define an arbitrary translation. The shift must preserve the symmetry of the array
#     #  (i.e. the shift must too be symmetric)
#     sym_shift = np.array([[28.36, 28.36, 28.36, 28.36], [28.36, 28.36, 28.36, 28.36],
#                           [28.36, 28.36, 28.36, 28.36], [28.36, 28.36, 28.36, 28.36]])

#     assert(abs(sym_shift - sym_shift.T) < 1.e-10).all()
#     # Translate and scale the initial array
#     array_b = 922.44 * sym_array + sym_shift
#     # Define an arbitrary permutation transformation
#     perm_array = np.array([[0., 0., 1., 0.], [0., 1., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]])
#     assert(abs(np.linalg.det(perm_array)) - 1. < 1.e-8)
#     assert(abs([x for x in perm_array.flatten().tolist() if x != 0] - np.ones(4)) < 1.e-8).all()
#     # Define the permuted original array
#     array_permuted = np.dot(np.dot(perm_array.T, array_b), perm_array)
#     # Proceed with permutation procrustes analysis
#     twosided_single_perm = TwoSidedPermutationSingleTransformationProcrustes(sym_array, array_permuted,
#                                                                              translate=True, scale=True)
#     least_error_perm, least_error_array_transformed, min_error = \
#         twosided_single_perm.calculate()
#     assert(abs(np.linalg.det(least_error_perm)) - 1. < 1.e-8)
#     assert(abs([x for x in least_error_perm.flatten().tolist() if x != 0] - np.ones(4)) < 1.e-8).all()
#     # Assert that the analysis returns zero error
#     assert min_error < 1.e-10

#     # Define arbitrary array
#     sym_part = np.array([[56.89, 49.22, 81.5, 76.1], [98.1, 64.3, 25.1, 64.75], [85.3, 90.5, 86.2, 55.1],
#                          [58.2, 63.1, 62.12, 53.3], [87.6, 56.9, 98.6, 69.69]])
#     sym_array = np.dot(sym_part, sym_part.T)
#     # Define an arbitrary translation. The shift must preserve the symmetry of the array
#     #  (i.e. the shift must too be symmetric)
#     sym_shift = np.array([[43.69, 43.69, 43.69, 43.69, 43.69], [43.69, 43.69, 43.69, 43.69, 43.69],
#                           [43.69, 43.69, 43.69, 43.69, 43.69], [43.69, 43.69, 43.69, 43.69, 43.69],
#                           [43.69, 43.69, 43.69, 43.69, 43.69]])

#     assert(abs(sym_shift - sym_shift.T) < 1.e-10).all()
#     # Translate and scale the initial array
#     array_b = 922.44 * sym_array + sym_shift
#     # Define an arbitrary permutation transformation
#     perm_array = np.array([[0., 0., 0., 0., 1.], [0., 1., 0., 0., 0.], [0., 0., 0., 1., 0.], [0., 0., 1., 0., 0.],
#                            [1., 0., 0., 0., 0.]])
#     assert(abs(np.linalg.det(perm_array)) - 1. < 1.e-8)
#     assert(abs([x for x in perm_array.flatten().tolist() if x != 0] - np.ones(5)) < 1.e-8).all()
#     # Define the permuted original array
#     array_permuted = np.dot(np.dot(perm_array.T, array_b), perm_array)
#     # Proceed with permutation procrustes analysis
#     twosided_single_perm = TwoSidedPermutationSingleTransformationProcrustes(sym_array, array_permuted,
#                                                                              translate=True, scale=True)
#     least_error_perm, least_error_array_transformed, min_error = \
#         twosided_single_perm.calculate()
#     assert(abs(np.linalg.det(least_error_perm)) - 1. < 1.e-8)
#     assert(abs([x for x in least_error_perm.flatten().tolist() if x != 0] - np.ones(5)) < 1.e-8).all()
#     # Assert that the analysis returns zero error
#     assert min_error < 1.e-10
