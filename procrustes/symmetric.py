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
"""
Symmetric Procrustes Module.
"""

from procrustes.utils import singular_value_decomposition, _get_input_arrays, \
    error
import numpy as np


def symmetric(A, B, remove_zero_col=True, remove_zero_row=True,
              pad_mode='row-col', translate=False, scale=False,
              check_finite=True):
    r"""
    Symmetric right-sided procrustes transformation.

    The symmetric procrustes analysis requires two 2d-arrays with the same number of rows, so
    the array with the smaller number of rows will automatically be padded with zero rows.

    Parameters
    ----------
    A : ndarray
        The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
    B : ndarray
        The 2d-array :math:`\mathbf{B}_{m \times n}` representing the reference.
    remove_zero_col : bool, optional
        If True, the zero columns on the right side will be removed.
        Default=True.
    remove_zero_row : bool, optional
        If True, the zero rows on the top will be removed.
        Default=True.
    pad_mode : str, optional
      Zero padding mode when the sizes of two arrays differ. Default='row-col'.
      'row': The array with fewer rows is padded with zero rows so that both have the same
           number of rows.
      'col': The array with fewer columns is padded with zero columns so that both have the
           same number of columns.
      'row-col': The array with fewer rows is padded with zero rows, and the array with fewer
           columns is padded with zero columns, so that both have the same dimensions.
           This does not necessarily result in square arrays.
      'square': The arrays are padded with zero rows and zero columns so that they are both
           squared arrays. The dimension of square array is specified based on the highest
           dimension, i.e. :math:`\text{max}(n_a, m_a, n_b, m_b)`.'
    translate : bool, optional
        If True, both arrays are translated to be centered at origin.
        Default=False.
    scale : bool, optional
        If True, both arrays are column normalized to unity. Default=False.
    check_finite : bool, optional
        If true, convert the input to an array, checking for NaNs or Infs.
        Default=True.

    Returns
    -------
    A : ndarray
        The transformed ndarray A.
    B : ndarray
        The transformed ndarray B.
    U_opt : ndarray
        The optimum symmetric transformation array.
    e_opt : float
        One-sided orthogonal Procrustes error.

    Notes
    -----
    Given matrix :math:`\mathbf{A}_{m \times n}` and a reference :math:`\mathbf{B}_{m \times n}`,
    with :math:`m \geqslant n`, find the symmetric matrix :math:`\mathbf{X}_{n \times n}` for which
    :math:`\mathbf{AX}` is as close as possible to :math:`\mathbf{B}_{m \times n}`. I.e.,

    .. math::
       \underbrace{\text{min}}_{\left\{\mathbf{X} \left| \mathbf{X} = \mathbf{X}^\dagger
                                \right. \right\}} \|\mathbf{A} \mathbf{X} - \mathbf{B}\|_{F}^2 =
       \underbrace{\text{min}}_{\left\{\mathbf{X} \left| \mathbf{X} = \mathbf{X}^\dagger
                                \right. \right\}}
                \text{Tr}\left[\left(\mathbf{A}\mathbf{X} - \mathbf{B} \right)^\dagger
                         \left(\mathbf{A}\mathbf{X} - \mathbf{B} \right)\right]

    Considering the singular value decomposition of :math:`\mathbf{A}_{m \times n}` as

    .. math::
       \mathbf{A}_{m \times n} = \mathbf{U}_{m \times m} \begin{bmatrix}
                                 \mathbf{\Sigma}_{m \times m} \\
                                 \mathbf{0}_{m \times (n - m)} \end{bmatrix}
                                 \mathbf{V}_{n \times n}^\dagger

    where :math:`\mathbf{\Sigma}_{n \times n}` is a square diagonal matrix with nonnegative elements
    denoted by :math:`\sigma_i` listed in decreasing order, define

    .. math::
       \mathbf{C}_{m \times n} = \mathbf{U}_{m \times m}^\dagger
                                 \mathbf{A}_{m \times n}^0 \mathbf{V}_{n \times n}

    Then the elements of the optimal matrix :math:`\mathbf{X}_{n \times n}` are

    .. math::
       x_{ij} = \begin{cases}
              0 && i \text{ and } j > \text{rank} \left(\mathbf{B}\right) \\
              \frac{\sigma_i c_{ij} + \sigma_j c_{ji}}{\sigma_i^2 + \sigma_j^2} && \text{otherwise}
              \end{cases}

    Notice that the first part of this constrain only works in the unusual case where
    :math:`\mathbf{B}` has rank less than :math:`n`.

    References
    ----------
    1. Higham, Nicholas J. The Symmetric Procrustes problem.
       BIT Numerical Mathematics, 28 (1), 133-143, 1988.

    Examples
    --------
    >>> import numpy as np
    >>> array_a = np.array([[5., 2., 8.],
                            [2., 2., 3.],
                            [1., 5., 6.],
                            [7., 3., 2.]])
    >>> array_b = np.array([[ 52284.5, 209138. , 470560.5],
                            [ 22788.5,  91154. , 205096.5],
                            [ 46139.5, 184558. , 415255.5],
                            [ 22788.5,  91154. , 205096.5]])
    >>> new_a, new_b, array_x, error_opt = symmetric(array_a, array_b, translate=True, scale=True)
    >>> array_x # symmetric transformation array
    array([[0.0166352 , 0.06654081, 0.14971682],
          [0.06654081, 0.26616324, 0.59886729],
          [0.14971682, 0.59886729, 1.34745141]])
    >>> error_opt # error
    4.483083428047388e-31

    """
    # check inputs
    A, B = _get_input_arrays(A, B, remove_zero_col, remove_zero_row,
                             pad_mode, translate, scale, check_finite)

    # compute SVD of A
    n = A.shape[1]
    u, s, vt = singular_value_decomposition(A)

    # add zeros to the eigenvalue array so it has length n
    if len(s) < A.shape[1]:
        s = np.concatenate((s, np.zeros(n - len(s))))
    c = np.dot(np.dot(u.T, B), vt.T)

    # create the intermediate array Y and the optimum symmetric transformation array X
    y = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if s[i] ** 2 + s[j] ** 2 == 0:
                y[i, j] = 0
            else:
                y[i, j] = (s[i] * c[i, j] + s[j] * c[j, i]) / \
                          (s[i] ** 2 + s[j] ** 2)
    X_opt = np.dot(np.dot(vt.T, y), vt)
    e_opt = error(A, B, X_opt)

    return A, B, X_opt, e_opt
