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
Orthogonal Procrustes Module.
"""


from procrustes.utils import singular_value_decomposition, _get_input_arrays, error
import numpy as np


def orthogonal(A, B, remove_zero_col=True, remove_zero_row=True,
               translate=False, scale=False, check_finite=True):
    r"""
    One-sided orthogonal Procrustes.

    The Procrustes analysis requires two 2d-arrays with the same number of rows, so the
    array with the smaller number of rows will automatically be padded with zero rows.

    Parameters
    ----------
    A : ndarray
        The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
    B : ndarray
        The 2d-array :math:`\mathbf{B}_{m \times n}` representing the reference array.
    remove_zero_col : bool, optional
        If True, the zero columns on the right side will be removed.
        Default= True.
    remove_zero_row : bool, optional
        If True, the zero rows on the top will be removed.
        Default= True.
    translate : bool, optional
        If True, both arrays are translated to be centered at origin.
        Default=False.
    scale : bool, optional
        If True, both arrays are column normalized to unity.
        Default=False.
    check_finite : bool, optional
        If true, convert the input to an array, checking for NaNs or Infs.
        Default=True.

    Given matrix :math:`\mathbf{A}_{m \times n}` and a reference :math:`\mathbf{B}_{m \times n}`,
    find the unitary/orthogonal transformation matrix :math:`\mathbf{U}_{n \times n}` that makes
    :math:`\mathbf{A}_{m \times n}` as close as possible to :math:`\mathbf{B}_{m \times n}`. I.e.,

    .. math::
       \underbrace{\min}_{\left\{\mathbf{U} | \mathbf{U}^{-1} = {\mathbf{U}}^\dagger \right\}}
                          \|\mathbf{A}\mathbf{U} - \mathbf{B}\|_{F}^2
       &= \underbrace{\text{min}}_{\left\{\mathbf{U} | \mathbf{U}^{-1} = {\mathbf{U}}^\dagger
                                   \right\}}
          \text{Tr}\left[\left(\mathbf{A}\mathbf{U} - \mathbf{B} \right)^\dagger
                         \left(\mathbf{A}\mathbf{U} - \mathbf{B} \right)\right] \\
       &= \underbrace{\text{max}}_{\left\{\mathbf{U} | \mathbf{U}^{-1} = {\mathbf{U}}^\dagger
                                   \right\}}
          \text{Tr}\left[\mathbf{U}^\dagger {\mathbf{A}}^\dagger \mathbf{B} \right]

    The solution is obtained by taking the singular value decomposition (SVD) of the product of the
    matrices,

    .. math::
       \mathbf{A}^\dagger \mathbf{B} &= \tilde{\mathbf{U}} \tilde{\mathbf{\Sigma}}
                                          \tilde{\mathbf{V}}^{\dagger} \\
       \mathbf{U}_{\text{optimum}} &= \tilde{\mathbf{U}} \tilde{\mathbf{V}}^{\dagger}

    The singular values are always listed in decreasing order, with the smallest singular
    value in the bottom-right-hand corner of :math:`\tilde{\mathbf{\Sigma}}`.
    """
    # check inputs
    A, B = _get_input_arrays(A, B, remove_zero_col, remove_zero_row, translate, scale, check_finite)

    # calculate SVD of A.T * B
    U, _, VT = singular_value_decomposition(np.dot(A.T, B))
    # compute optimum orthogonal transformation
    U_opt = np.dot(U, VT)
    # compute the error
    e_opt = error(A, B, U_opt)
    return A, B, U_opt, e_opt
