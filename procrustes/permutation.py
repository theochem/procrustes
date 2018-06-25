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
"""Permutation Procrustes Module."""


import numpy as np

from scipy.optimize import linear_sum_assignment

from procrustes.utils import _get_input_arrays, _check_rank, eigendecomposition, error


__all__ = [
    "permutation",
]


def permutation(A, B, remove_zero_col=True,
                remove_zero_row=True, translate=False, scale=False,
                check_finite=True):
    """
    Single sided permutation Procrustes.

    Parameters
    ----------
    A : ndarray
        The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
    B : ndarray
        The 2d-array :math:`\mathbf{B}_{m \times n}` representing the reference.
    remove_zero_col : bool, optional
        If True, the zero columns on the right side will be removed. Default= True.
    remove_zero_row : bool, optional
        If True, the zero rows on the top will be removed. Default= True.
    translate : bool, optional
        If True, both arrays are translated to be centered at origin. Default=False.
    scale : bool, optional
        If True, both arrays are column normalized to unity. Default=False.
    check_finite : bool, optional
        If true, convert the input to an array, checking for NaNs or Infs. Default=True.

    Returns
    -------
    A : ndarray
        The transformed ndarray A.
    B : ndarray
        The transformed ndarray B.
    U_opt : ndarray
        The optimum permutation transformation matrix.
    e_opt : float
        One-sided orthogonal Procrustes error.
    """
    # check inputs
    A, B = _get_input_arrays(A, B, remove_zero_col, remove_zero_row,
                             translate, scale, check_finite)
    # compute permutation Procrustes matrix
    P = np.dot(A.T, B)
    C = np.full(P.shape, np.max(P))
    C -= P
    U = np.zeros(P.shape)
    # set elements to 1 according to Hungarian algorithm (linear_sum_assignment)
    U[linear_sum_assignment(C)] = 1
    e_opt = error(A, B, U)
    return A, B, U, e_opt
