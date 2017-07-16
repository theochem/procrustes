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
