# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2020 The Procrustes Development Team
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

import numpy as np
from procrustes.utils import error, setup_input_arrays


def symmetric(array_a, array_b, remove_zero_col=True, remove_zero_row=True,
              pad_mode='row-col', translate=False, scale=False, check_finite=True):
    r"""
    Symmetric right-sided Procrustes transformation.

    The symmetric Procrustes requires both matrices to have the number of rows
    greater than equal to the number of columns. Further, it is assumed that
    each matrix has the same dimension, if not padding will occur.

    Parameters
    ----------
    array_a : ndarray
        The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
    array_b : ndarray
        The 2d-array :math:`\mathbf{B}_{m \times n}` representing the reference.
    remove_zero_col : bool, optional
        If True, the zero columns on the right side will be removed.
        Default=True.
    remove_zero_row : bool, optional
        If True, the zero rows on the bottom will be removed.
        Default=True.
    pad_mode : str, optional
        Specifying how to pad the arrays, listed below. Default="row-col".

            - "row"
                The array with fewer rows is padded with zero rows so that both have the same
                number of rows.
            - "col"
                The array with fewer columns is padded with zero columns so that both have the
                same number of columns.
            - "row-col"
                The array with fewer rows is padded with zero rows, and the array with fewer
                columns is padded with zero columns, so that both have the same dimensions.
                This does not necessarily result in square arrays.
            - "square"
                The arrays are padded with zero rows and zero columns so that they are both
                squared arrays. The dimension of square array is specified based on the highest
                dimension, i.e. :math:`\text{max}(n_a, m_a, n_b, m_b)`.
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
    new_a : ndarray
        The transformed ndarray array_a.
    new_b : ndarray
        The transformed ndarray array_b.
    array_x : ndarray
        The optimum symmetric transformation array.
    e_opt : float
        One-sided Procrustes error.

    Raises
    ------
    ValueError :
        If their matrix dimension (m, n) don't satisfy :math:`m \geq n` after padding.

    Notes
    -----
    Given matrix :math:`\mathbf{A}_{m \times n}` and a reference
    :math:`\mathbf{B}_{m \times n}`,
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
    >>> array_a = np.array([[5., 2., 8.], \
                            [2., 2., 3.], \
                            [1., 5., 6.], \
                            [7., 3., 2.]])
    >>> array_b = np.array([[ 52284.5, 209138. , 470560.5], \
                            [ 22788.5,  91154. , 205096.5], \
                            [ 46139.5, 184558. , 415255.5], \
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
    new_a, new_b = setup_input_arrays(array_a, array_b, remove_zero_col, remove_zero_row,
                                      pad_mode, translate, scale, check_finite)
    if new_a.shape[0] < new_a.shape[1]:
        raise ValueError("Array A with size (m, n) needs m >= to n.")
    if new_b.shape[0] < new_b.shape[1]:
        raise ValueError("Array B with size (m, n) needs m >= to n.")

    # compute SVD of  new_a
    array_n = new_a.shape[1]
    array_u, array_s, array_vt = np.linalg.svd(new_a)

    # add zeros to the eigenvalue array so it has length n
    if len(array_s) < new_a.shape[1]:
        array_s = np.concatenate((array_s, np.zeros(array_n - len(array_s))))
    array_c = np.dot(np.dot(array_u.T, new_b), array_vt.T)

    # create the intermediate array Y and the optimum symmetric transformation array X
    array_y = np.zeros((array_n, array_n))
    for i in range(array_n):
        for j in range(array_n):
            if array_s[i] ** 2 + array_s[j] ** 2 == 0:
                array_y[i, j] = 0
            else:
                array_y[i, j] = (array_s[i] * array_c[i, j] + array_s[j] * array_c[j, i]) / (
                            array_s[i] ** 2 + array_s[j] ** 2)
    array_x = np.dot(np.dot(array_vt.T, array_y), array_vt)
    e_opt = error(new_a, new_b, array_x)

    return new_a, new_b, array_x, e_opt
