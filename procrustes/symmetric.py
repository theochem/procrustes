# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2024 The QC-Devs Community
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
"""Symmetric Procrustes Module."""

from typing import Optional

import numpy as np
import scipy

from procrustes.utils import ProcrustesResult, _zero_padding, compute_error, setup_input_arrays

__all__ = [
    "symmetric",
]


def symmetric(
    a: np.ndarray,
    b: np.ndarray,
    pad: bool = True,
    translate: bool = False,
    scale: bool = False,
    unpad_col: bool = False,
    unpad_row: bool = False,
    check_finite: bool = True,
    weight: Optional[np.ndarray] = None,
    lapack_driver: str = "gesvd",
) -> ProcrustesResult:
    r"""Perform symmetric Procrustes.

    Given a matrix :math:`\mathbf{A}_{m \times n}` and a reference matrix :math:`\mathbf{B}_{m
    \times n}` with :math:`m \geqslant n`, find the symmetrix transformation matrix
    :math:`\mathbf{X}_{n \times n}` that makes :math:`\mathbf{AX}` as close as possible to
    :math:`\mathbf{B}`. In other words,

    .. math::
       \underbrace{\text{min}}_{\left\{\mathbf{X} \left| \mathbf{X} = \mathbf{X}^\dagger
                        \right. \right\}} \|\mathbf{A} \mathbf{X} - \mathbf{B}\|_{F}^2

    This Procrustes method requires the :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices to
    have the same shape with :math:`m \geqslant n`, which is guaranteed with the default ``pad``
    argument for any given :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices.
    In preparing the :math:`\mathbf{A}` and
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
    lapack_driver : {'gesvd', 'gesdd'}, optional
        Whether to use the more efficient divide-and-conquer approach ('gesdd') or the more robust
        general rectangular approach ('gesvd') to compute the singular-value decomposition with
        `scipy.linalg.svd`.

    Returns
    -------
    res : ProcrustesResult
        The Procrustes result represented as a class:`utils.ProcrustesResult` object.

    Notes
    -----
    The optimal symmetrix matrix is obtained by,

    .. math::
       \mathbf{X}_{\text{opt}} = \arg
       \underbrace{\text{min}}_{\left\{\mathbf{X} \left| \mathbf{X} = \mathbf{X}^\dagger
                        \right. \right\}} \|\mathbf{A} \mathbf{X} - \mathbf{B}\|_{F}^2 =
       \underbrace{\text{min}}_{\left\{\mathbf{X} \left| \mathbf{X} = \mathbf{X}^\dagger
                        \right. \right\}}
                \text{Tr}\left[\left(\mathbf{A}\mathbf{X} - \mathbf{B} \right)^\dagger
                         \left(\mathbf{A}\mathbf{X} - \mathbf{B} \right)\right]

    Considering the singular value decomposition of :math:`\mathbf{A}`,

    .. math::
       \mathbf{A}_{m \times n} = \mathbf{U}_{m \times m}
                                 \mathbf{\Sigma}_{m \times n}
                                 \mathbf{V}_{n \times n}^\dagger

    where :math:`\mathbf{\Sigma}_{m \times n}` is a rectangular diagonal matrix with non-negative
    singular values :math:`\sigma_i = [\mathbf{\Sigma}]_{ii}` listed in descending order, define

    .. math::
       \mathbf{C}_{m \times n} = \mathbf{U}_{m \times m}^\dagger
                                 \mathbf{B}_{m \times n} \mathbf{V}_{n \times n}

    with elements denoted by :math:`c_{ij}`.
    Then we compute the symmetric matrix :math:`\mathbf{Y}_{n \times n}` with

    .. math::
       [\mathbf{Y}]_{ij} = \begin{cases}
              0 && i \text{ and } j > \text{rank} \left(\mathbf{A}\right) \\
              \frac{\sigma_i c_{ij} + \sigma_j c_{ji}}{\sigma_i^2 +
              \sigma_j^2} && \text{otherwise} \end{cases}

    It is worth noting that the first part of this definition only applies in the unusual case where
    :math:`\mathbf{A}` has rank less than :math:`n`. The :math:`\mathbf{X}_\text{opt}` is given by

    .. math::
       \mathbf{X}_\text{opt} = \mathbf{V Y V}^{\dagger}

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[5., 2., 8.],
    ...               [2., 2., 3.],
    ...               [1., 5., 6.],
    ...               [7., 3., 2.]])
    >>> b = np.array([[ 52284.5, 209138. , 470560.5],
    ...               [ 22788.5,  91154. , 205096.5],
    ...               [ 46139.5, 184558. , 415255.5],
    ...               [ 22788.5,  91154. , 205096.5]])
    >>> res = symmetric(a, b, pad=True, translate=True, scale=True)
    >>> res.t   # symmetric transformation array
    array([[0.0166352 , 0.06654081, 0.14971682],
          [0.06654081, 0.26616324, 0.59886729],
          [0.14971682, 0.59886729, 1.34745141]])
    >>> res.error   # error
    4.483083428047388e-31

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

    # if number of rows is less than column, the arrays are made square
    if (new_a.shape[0] < new_a.shape[1]) or (new_b.shape[0] < new_b.shape[1]):
        new_a, new_b = _zero_padding(new_a, new_b, "square")

    # if new_a.shape[0] < new_a.shape[1]:
    #     raise ValueError(f"Shape of A {new_a.shape}=(m, n) needs to satisfy m >= n.")
    #
    # if new_b.shape[0] < new_b.shape[1]:
    #     raise ValueError(f"Shape of B {new_b.shape}=(m, n) needs to satisfy m >= n.")

    # compute SVD of A & matrix C
    u, s, vt = scipy.linalg.svd(new_a, lapack_driver=lapack_driver)
    c = np.dot(np.dot(u.T, new_b), vt.T)

    # compute intermediate matrix Y
    n = new_a.shape[1]
    y = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if s[i] ** 2 + s[j] ** 2 != 0:
                y[i, j] = (s[i] * c[i, j] + s[j] * c[j, i]) / (s[i] ** 2 + s[j] ** 2)

    # compute optimum symmetric transformation matrix X
    x = np.dot(np.dot(vt.T, y), vt)
    error = compute_error(new_a, new_b, x)

    return ProcrustesResult(error=error, new_a=new_a, new_b=new_b, t=x, s=None)
