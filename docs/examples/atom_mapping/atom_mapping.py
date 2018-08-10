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


r"""Two sided permutation Procrustes example: molecular alignment alignment."""


from __future__ import absolute_import, division, print_function
import numpy as np
from procrustes import permutation_2sided, _utils


def mol_align(A, B):
    r"""Align two molecules using two sided permutation Procrustes with one
    transformation.
    """
    # Compute the permutation matrix
    _, _, U, e_opt = permutation_2sided(
        A, B, transform_mode='single_undirected',
        remove_zero_col=False, remove_zero_row=False)
    # Compute the transformed molecule A
    A = _utils._get_input_arrays(A)
    new_A = np.dot(U.T, np.dot(A, U))
    # B
    new_B = B

    return new_A, new_B, U, e_opt


if __name__ == "__main__":

    # Define molecule A
    # but‐1‐en‐3‐yne
    A = np.array([[6, 3, 0, 0],
                  [3, 6, 1, 0],
                  [0, 1, 6, 2],
                  [0, 0, 2, 6]])

    # Define molecule B
    # 3,3‐dimethylpent‐1‐en‐4‐yne
    B = np.array([[6, 3, 0, 0, 0, 0, 0],
                  [3, 6, 1, 0, 0, 0, 0],
                  [0, 1, 6, 1, 0, 1, 1],
                  [0, 0, 1, 6, 2, 0, 0],
                  [0, 0, 0, 2, 6, 0, 0],
                  [0, 0, 1, 0, 0, 6, 0],
                  [0, 0, 1, 0, 0, 0, 6]])
    # Compute the alignment
    new_A, new_B, U, e_opt = mol_align(A, B)


# the result new_A
# array([[6, 3, 0, 0, 0, 0, 0],
#        [3, 6, 0, 1, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0],
#        [0, 1, 0, 6, 2, 0, 0],
#        [0, 0, 0, 2, 6, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0]])

#########################################
# Rotate A by 180 degrees
# if __name__ == "__main__":
#
#     # Define molecule A
#     # but‐1‐en‐3‐yne
#     A = np.array([[6, 2, 0, 0],
#                   [2, 6, 1, 0],
#                   [0, 1, 6, 3],
#                   [0, 0, 3, 6]])
#
#     # Define molecule B
#     # 3,3‐dimethylpent‐1‐en‐4‐yne
#     B = np.array([[6, 3, 0, 0, 0, 0, 0],
#                   [3, 6, 1, 0, 0, 0, 0],
#                   [0, 1, 6, 1, 0, 1, 1],
#                   [0, 0, 1, 6, 2, 0, 0],
#                   [0, 0, 0, 2, 6, 0, 0],
#                   [0, 0, 1, 0, 0, 6, 0],
#                   [0, 0, 1, 0, 0, 0, 6]])
#     # Compute the alignment
#     new_A, new_B, U, e_opt = mol_align(A, B)

# the result new_A
# A, _ = _utils._get_input_arrays(
#     A, B, remove_zero_col=False,
#     remove_zero_row=False, translate=False,
#     scale=False, check_finite=False)
# np.dot(U.T, np.dot(A, U))
# array([[6, 3, 0, 0, 0, 0, 0],
#        [3, 6, 0, 1, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0],
#        [0, 1, 0, 6, 2, 0, 0],
#        [0, 0, 0, 2, 6, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0]])
