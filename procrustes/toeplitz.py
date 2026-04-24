# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2025 The QC-Devs Community
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
"""Toeplitz and Hankel Procrustes Module."""

from typing import Optional

import numpy as np

from procrustes.utils import ProcrustesResult, compute_error, setup_input_arrays

__all__ = [
    "toeplitz",
    "hankel",
]


def _build_toeplitz_operator(a: np.ndarray) -> np.ndarray:
    r"""Build the linear operator M such that vec(A @ T) = M @ t for any Toeplitz T.

    A Toeplitz matrix of size n x n is fully determined by its first column c (length n)
    and first row r (length n), giving 2n-1 unique parameters. We stack them as:
        t = [c[0], c[1], ..., c[n-1], r[1], r[2], ..., r[n-1]]
    where c[0] == r[0] (the main diagonal value).

    This function builds M (shape: m*n x (2n-1)) such that:
        vec(A @ T) = M @ t

    Parameters
    ----------
    a : ndarray
        The 2D-array :math:`\mathbf{A}_{m \times n}`.

    Returns
    -------
    op_m : ndarray
        The operator matrix of shape (m * n, 2n - 1).
    """
    m, n = a.shape
    num_params = 2 * n - 1
    op_m = np.zeros((m * n, num_params), dtype=a.dtype)

    for col_idx in range(n):
        for row_idx in range(n):
            # Diagonal index: 0 = main diagonal, >0 = upper, <0 = lower.
            # Maps (row, col) position in T to its parameter index in the flat vector t.
            diag = col_idx - row_idx
            if diag == 0:
                param_idx = 0           # main diagonal: shared entry t[0]
            elif diag < 0:
                param_idx = -diag       # sub-diagonal: first column entry t[-diag]
            else:
                param_idx = n - 1 + diag  # super-diagonal: first row entry t[n-1+diag]
            op_m[col_idx * m : (col_idx + 1) * m, param_idx] += a[:, row_idx]

    return op_m


def _build_hankel_operator(a: np.ndarray) -> np.ndarray:
    r"""Build the linear operator M such that vec(A @ H) = M @ h for any Hankel H.

    A Hankel matrix of size n x n is fully determined by its first column c (length n)
    and last row r (length n), giving 2n-1 unique parameters. We stack them as:
        h = [c[0], c[1], ..., c[n-1], r[1], r[2], ..., r[n-1]]
    where c[n-1] == r[0] (the anti-diagonal corner value).

    In a Hankel matrix: H[i, j] = h[i + j] (using 0-based indexing into the parameter vector).

    Parameters
    ----------
    a : ndarray
        The 2D-array :math:`\mathbf{A}_{m \times n}`.

    Returns
    -------
    op_m : ndarray
        The operator matrix of shape (m * n, 2n - 1).
    """
    m, n = a.shape
    num_params = 2 * n - 1
    op_m = np.zeros((m * n, num_params), dtype=a.dtype)

    for col_idx in range(n):
        for row_idx in range(n):
            # H[row_idx, col_idx] = h[row_idx + col_idx] by the Hankel anti-diagonal property.
            param_idx = row_idx + col_idx
            op_m[col_idx * m : (col_idx + 1) * m, param_idx] += a[:, row_idx]

    return op_m


def _params_to_toeplitz(params: np.ndarray, n: int) -> np.ndarray:
    r"""Construct an n x n Toeplitz matrix from its parameter vector.

    Parameters
    ----------
    params : ndarray
        1D array of length 2n-1. The first n entries are the first column
        (top to bottom), and entries n..2n-2 are the rest of the first row
        (left to right, excluding the corner which is params[0]).
    n : int
        Size of the square Toeplitz matrix.

    Returns
    -------
    t_mat : ndarray
        The n x n Toeplitz matrix.
    """
    t_mat = np.zeros((n, n), dtype=params.dtype)
    for i in range(n):
        for j in range(n):
            # Same diagonal mapping as in _build_toeplitz_operator.
            diag = j - i
            if diag == 0:
                t_mat[i, j] = params[0]
            elif diag < 0:
                t_mat[i, j] = params[-diag]
            else:
                t_mat[i, j] = params[n - 1 + diag]
    return t_mat


def _params_to_hankel(params: np.ndarray, n: int) -> np.ndarray:
    r"""Construct an n x n Hankel matrix from its parameter vector.

    Parameters
    ----------
    params : ndarray
        1D array of length 2n-1. H[i, j] = params[i + j].
    n : int
        Size of the square Hankel matrix.

    Returns
    -------
    h_mat : ndarray
        The n x n Hankel matrix.
    """
    h_mat = np.zeros((n, n), dtype=params.dtype)
    for i in range(n):
        for j in range(n):
            h_mat[i, j] = params[i + j]
    return h_mat


def toeplitz(
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
    r"""Perform one-sided Toeplitz Procrustes.

    Given a matrix :math:`\mathbf{A}_{m \times n}` and a reference matrix :math:`\mathbf{B}_{m
    \times n}`, find the Toeplitz transformation matrix :math:`\mathbf{T}_{n \times n}` that
    makes :math:`\mathbf{A}\mathbf{T}` as close as possible to :math:`\mathbf{B}`. In other words,

    .. math::
       \underbrace{\min}_{\mathbf{T} \in \mathcal{T}_n} \|\mathbf{A}\mathbf{T} - \mathbf{B}\|_{F}^2

    where :math:`\mathcal{T}_n` denotes the set of all :math:`n \times n` Toeplitz matrices.

    A Toeplitz matrix has constant values along each diagonal:

    .. math::
       T_{ij} = t_{j-i}

    so it is fully determined by :math:`2n-1` parameters.

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
        If True, zero columns (with values less than 1.0e-8) on the right-hand side of the initial
        :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices are removed.
    unpad_row : bool, optional
        If True, zero rows (with values less than 1.0e-8) at the bottom of the initial
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
        The Procrustes result represented as a :class:`utils.ProcrustesResult` object.

    Notes
    -----
    The optimal Toeplitz matrix is found by reformulating the problem as a linear least-squares
    problem. Since a Toeplitz matrix :math:`\mathbf{T}` of size :math:`n \times n` is determined
    by :math:`2n-1` parameters (its first column and first row), we can write:

    .. math::
       \text{vec}(\mathbf{A}\mathbf{T}) = \mathbf{M} \mathbf{t}

    where :math:`\mathbf{t}` is the parameter vector and :math:`\mathbf{M}` is a structured matrix
    derived from :math:`\mathbf{A}`. The optimal :math:`\mathbf{t}` is then found via
    least-squares (using an SVD-based solver):

    .. math::
       \mathbf{t}^{\text{opt}} = \arg\min_{\mathbf{t}}
       \|\mathbf{M}\mathbf{t} - \text{vec}(\mathbf{B})\|_2^2

    References
    ----------
    .. [1] Yang, J., & Deng, Y. (2013). "Procrustes Problems for General, Triangular, and
           Symmetric Toeplitz Matrices."
           Journal of Applied Mathematics, 2013.
           http://dx.doi.org/10.1155/2013/696019

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import toeplitz as scipy_toeplitz
    >>> from procrustes import toeplitz
    >>> n = 4
    >>> A = np.random.rand(n, n)
    >>> T_true = scipy_toeplitz([1, 2, 3, 4], [1, 0.5, 0.25, 0.1])
    >>> B = A @ T_true
    >>> result = toeplitz(A, B)
    >>> print(result.error)   # should be near zero
    >>> print(result.t)       # should match T_true

    """
    new_a, new_b = setup_input_arrays(
        a, b, unpad_col, unpad_row, pad, translate, scale, check_finite, weight
    )
    if new_a.shape != new_b.shape:
        raise ValueError(
            f"Shape of A and B does not match: {new_a.shape} != {new_b.shape} "
            "Check pad, unpad_col, and unpad_row arguments."
        )

    _, n = new_a.shape
    # Build operator M so that vec(A @ T) = M @ t, then solve for t via least-squares.
    op_m = _build_toeplitz_operator(new_a)
    b_vec = new_b.flatten(order="F")  # column-major vectorization to match vec convention
    params, _, _, _ = np.linalg.lstsq(op_m, b_vec, rcond=None)
    t_opt = _params_to_toeplitz(params, n)
    error = compute_error(new_a, new_b, t_opt)

    return ProcrustesResult(error=error, new_a=new_a, new_b=new_b, t=t_opt, s=None)


def hankel(
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
    r"""Perform one-sided Hankel Procrustes.

    Given a matrix :math:`\mathbf{A}_{m \times n}` and a reference matrix :math:`\mathbf{B}_{m
    \times n}`, find the Hankel transformation matrix :math:`\mathbf{H}_{n \times n}` that
    makes :math:`\mathbf{A}\mathbf{H}` as close as possible to :math:`\mathbf{B}`. In other words,

    .. math::
       \underbrace{\min}_{\mathbf{H} \in \mathcal{H}_n} \|\mathbf{A}\mathbf{H} - \mathbf{B}\|_{F}^2

    where :math:`\mathcal{H}_n` denotes the set of all :math:`n \times n` Hankel matrices.

    A Hankel matrix has constant values along each anti-diagonal:

    .. math::
       H_{ij} = h_{i+j}

    so it is fully determined by :math:`2n-1` parameters.

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
        If True, zero columns (with values less than 1.0e-8) on the right-hand side of the initial
        :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices are removed.
    unpad_row : bool, optional
        If True, zero rows (with values less than 1.0e-8) at the bottom of the initial
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
        The Procrustes result represented as a :class:`utils.ProcrustesResult` object.

    Notes
    -----
    The optimal Hankel matrix is found by reformulating the problem as a linear least-squares
    problem. Since a Hankel matrix :math:`\mathbf{H}` of size :math:`n \times n` is determined
    by :math:`2n-1` parameters, we can write:

    .. math::
       \text{vec}(\mathbf{A}\mathbf{H}) = \mathbf{M} \mathbf{h}

    where :math:`\mathbf{h}` is the parameter vector and :math:`\mathbf{M}` is a structured matrix
    derived from :math:`\mathbf{A}`. The optimal :math:`\mathbf{h}` is then found via
    least-squares (using an SVD-based solver):

    .. math::
       \mathbf{h}^{\text{opt}} = \arg\min_{\mathbf{h}}
       \|\mathbf{M}\mathbf{h} - \text{vec}(\mathbf{B})\|_2^2

    References
    ----------
    .. [1] Yang, J., & Deng, Y. (2013). "Procrustes Problems for General, Triangular, and
           Symmetric Toeplitz Matrices."
           Journal of Applied Mathematics, 2013.
           http://dx.doi.org/10.1155/2013/696019

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import hankel as scipy_hankel
    >>> from procrustes import hankel
    >>> n = 4
    >>> A = np.random.rand(n, n)
    >>> H_true = scipy_hankel([1, 2, 3, 4], [4, 5, 6, 7])
    >>> B = A @ H_true
    >>> result = hankel(A, B)
    >>> print(result.error)   # should be near zero
    >>> print(result.t)       # should match H_true

    """
    new_a, new_b = setup_input_arrays(
        a, b, unpad_col, unpad_row, pad, translate, scale, check_finite, weight
    )
    if new_a.shape != new_b.shape:
        raise ValueError(
            f"Shape of A and B does not match: {new_a.shape} != {new_b.shape} "
            "Check pad, unpad_col, and unpad_row arguments."
        )

    _, n = new_a.shape
    # Build operator M so that vec(A @ H) = M @ h, then solve for h via least-squares.
    op_m = _build_hankel_operator(new_a)
    b_vec = new_b.flatten(order="F")  # column-major vectorization to match vec convention
    params, _, _, _ = np.linalg.lstsq(op_m, b_vec, rcond=None)
    t_opt = _params_to_hankel(params, n)
    error = compute_error(new_a, new_b, t_opt)

    return ProcrustesResult(error=error, new_a=new_a, new_b=new_b, t=t_opt, s=None)
