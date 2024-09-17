# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2022 The QC-Devs Community
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
"""Generic Procrustes Module."""

from typing import Optional

import numpy as np
from scipy.linalg import pinv

from procrustes.utils import ProcrustesResult, compute_error, setup_input_arrays

__all__ = [
    "generic",
]


def generic(
    a: np.ndarray,
    b: np.ndarray,
    pad: bool = True,
    translate: bool = False,
    scale: bool = False,
    unpad_col: bool = False,
    unpad_row: bool = False,
    check_finite: bool = True,
    weight: Optional[np.ndarray] = None,
) -> ProcrustesResult:
    r"""Perform generic one-sided Procrustes.

    Given matrix :math:`\mathbf{A}_{m \times n}` and a reference matrix :math:`\mathbf{B}_{m \times
    n}`, find the transformation matrix :math:`\mathbf{T}_{n \times n}` that makes
    :math:`\mathbf{AT}` as close as possible to :math:`\mathbf{B}`. In other words,

    .. math::
       \underbrace{\text{min}}_{\mathbf{T}} \quad \|\mathbf{A} \mathbf{T} - \mathbf{B}\|_{F}^2

    This Procrustes method requires the :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices to
    have the same shape, which is gauranteed with the default ``pad`` argument for any given
    :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices. In preparing the :math:`\mathbf{A}` and
    :math:`\mathbf{B}` matrices, the (optional) order of operations is: **1)** unpad zero
    rows/columns, **2)** translate the matrices to the origin, **3)** weight entries of
    :math:`\mathbf{A}`, **4)** scale the matrices to have unit norm, **5)** pad matrices with zero
    rows/columns so they have the same shape.

    Parameters
    ----------
    a : ndarray
        The 2D-array :math:`\mathbf{A}` which is going to be transformed.
    b : ndarray
        The 2D-array :math:`\mathbf{B}` representing the reference matrix.
    pad : bool, optional
        Add zero rows (at the bottom) and/or columns (to the right-hand side) of matrices
        :math:`\mathbf{A}` and :math:`\mathbf{B}` so that they have the same shape.
    translate : bool, optional
        If True, both arrays are centered at origin (columns of the arrays will have mean zero).
    scale : bool, optional
        If True, both arrays are normalized with respect to the Frobenius norm, i.e.,
        :math:`\text{Tr}\left[\mathbf{A}^\dagger\mathbf{A}\right] = 1` and
        :math:`\text{Tr}\left[\mathbf{B}^\dagger\mathbf{B}\right] = 1`.
    unpad_col : bool, optional
        If True, zero columns (with values less than 1.0e-8) on the right-hand side of the intial
        :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices are removed.
    unpad_row : bool, optional
        If True, zero rows (with values less than 1.0e-8) at the bottom of the intial
        :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices are removed.
    check_finite : bool, optional
        If True, convert the input to an array, checking for NaNs or Infs.
    weight : ndarray, optional
        The 1D-array representing the weights of each row of :math:`\mathbf{A}`. This defines the
        elements of the diagonal matrix :math:`\mathbf{W}` that is multiplied by :math:`\mathbf{A}`
        matrix, i.e., :math:`\mathbf{A} \rightarrow \mathbf{WA}`.

    Returns
    -------
    res : ProcrustesResult
        The Procrustes result represented as a class:`utils.ProcrustesResult` object.

    Notes
    -----
    The optimal transformation matrix is obtained by solving the least-squares equations,

    .. math::
        \mathbf{X}_\text{opt} = {(\mathbf{A}^{\top}\mathbf{A})}^{-1} \mathbf{A}^{\top} \mathbf{B}

    If :math:`m < n`, the transformation matrix :math:`\mathbf{T}_\text{opt}` is not unique,
    because the system of equations is underdetermined (i.e., there are fewer equations than
    unknowns).

    """
    # check inputs
    new_a, new_b = setup_input_arrays(
        a,
        b,
        unpad_col,
        unpad_row,
        pad,
        translate,
        scale,
        check_finite,
        weight,
    )
    # compute the generic solution
    try:
        a_inv = pinv(np.dot(new_a.T, new_a))
    # add little bit of random noise when the matrix is ill conditioned
    except np.linalg.LinAlgError:
        # conver new_a to float if it is not
        new_a = new_a.astype(float)
        new_a += 2e-14 * np.random.random_sample((new_a.shape[0], new_a.shape[1])) - 1e-14
        a_inv = pinv(np.dot(new_a.T, new_a))
    # a_inv = pinv(np.dot(new_a.T, new_a))

    array_x = np.linalg.multi_dot([a_inv, new_a.T, new_b])
    # compute one-sided error
    e_opt = compute_error(new_a, new_b, array_x)
    return ProcrustesResult(error=e_opt, new_a=new_a, new_b=new_b, t=array_x, s=None)
