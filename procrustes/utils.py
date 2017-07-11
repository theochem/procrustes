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
Utility Module.
"""


import numpy as np


def zero_padding(array_a, array_b, mode):
    r"""
    Return arrays padded with rowm and/or columns of zero.

    Parameters
    ----------
    array_a : ndarray
        The 2D array :math:`\mathbf{A}_{n_1 \times m_1}`.
    array_b : ndarray
        The 2D array :math:`\mathbf{B}_{n_2 \times m_2}`.
    mode : str
        One of the following values specifying how to padd arrays:

        **'row'**
             The array with fewer rows is padded with zero rows so that both have the same
             number of rows.
        **'col'**
             The array with fewer columns is padded with zero columns so that both have the
             same number of columns.
        **'row-col'**
             The array with fewer rows is padded with zero rows, and the array with fewer
             columns is padded with zero columns, so that both have the same dimensions.
             This does not necessarily result in square arrays.
        'square'
             The arrays are padded with zero rows and zero columns so that they are both
             squared arrays. The dimension of square array is specified based on the highest
             dimension, i.e. :math:`\text{max}(n_1, m_1, n_2, m_2)`.

    Returns
    -------
    array : ndarray
        Padded array_a and array_b arrays.
    """

    # sanity checks
    if not isinstance(array_a, np.ndarray) or not isinstance(array_b, np.ndarray):
        raise ValueError('Arguments array_a & array_b should be numpy arrays.')
    if array_a.ndim != 2 or array_b.ndim != 2:
        raise ValueError('Arguments array_a & array_b should be 2D arrays.')

    if array_a.shape == array_b.shape:
        # special case of square arrays, mode is set to None so that array_a & array_b are returned.
        mode = None

    if mode == 'square':
        # calculate desired dimension of square array
        (n1, m1), (n2, m2) = array_a.shape, array_b.shape
        dim = max(n1, n2, m1, m2)
        # padding rows to have both arrays have dim rows
        if n1 < dim:
            array_a = np.pad(
                array_a, [[0, dim - n1], [0, 0]], 'constant', constant_values=0)
        if n2 < dim:
            array_b = np.pad(
                array_b, [[0, dim - n2], [0, 0]], 'constant', constant_values=0)
        # padding columns to have both arrays have dim columns
        if m1 < dim:
            array_a = np.pad(
                array_a, [[0, 0], [0, dim - m1]], 'constant', constant_values=0)
        if m2 < dim:
            array_b = np.pad(
                array_b, [[0, 0], [0, dim - m2]], 'constant', constant_values=0)

    if mode == 'row' or mode == 'row-col':
        # padding rows to have both arrays have the same number of rows
        diff = array_a.shape[0] - array_b.shape[0]
        if diff < 0:
            array_a = np.pad(
                array_a, [[0, -diff], [0, 0]], 'constant', constant_values=0)
        else:
            array_b = np.pad(array_b, [[0, diff], [0, 0]],
                             'constant', constant_values=0)

    if mode == 'col' or mode == 'row-col':
        # padding columns to have both arrays have the same number of columns
        diff = array_a.shape[1] - array_b.shape[1]
        if diff < 0:
            array_a = np.pad(
                array_a, [[0, 0], [0, -diff]], 'constant', constant_values=0)
        else:
            array_b = np.pad(array_b, [[0, 0], [0, diff]],
                             'constant', constant_values=0)

    return array_a, array_b


def translate_array(array_a, array_b=None):
    """
    Return translated array_a and translation vector.

    Parameters
    ----------
    array_a : ndarray
        The 2D array to translate.
    array_b : ndarray
        The 2D array to translate array_a based on (default=None).

    Returns
    -------
    ndarray, ndarray
        If array_b is None, array_a is translated to origin using its centroid.
        If array_b is given, array_a is translated to centroid of array_b (the centroid of
        translated array_a will centroid with the centroid array_b).
    """

    # The mean is strongly affected by outliers and is not a robust estimator for central location
    # see https://docs.python.org/3.6/library/statistics.html?highlight=mean#statistics.mean
    centroid = compute_centroid(array_a)
    if array_b is not None:
        # translation vector to b centroid
        centroid -= compute_centroid(array_b)
    return array_a - centroid, -centroid


def scale_array(array_a, array_b=None):
    """
    Return scaled array_a and scaling vector.

    Parameters
    ----------
    array_a : ndarray
        The 2D array to scale
    array_b : ndarray, default=None
        The 2D array to scale array_a based on.

    Returns
    -------
    ndarray, ndarray
        If array_b is None, array_a is scaled to match norm of unit sphere using array_a's
        Frobenius norm.
        If array_b is given, array_a is scaled to match array_b's norm (the norm of array_a
        will be equal norm of array_b).
    """

    # scaling factor to match unit sphere
    scale = 1. / frobenius_norm(array_a)
    if array_b is not None:
        # scaling factor to match array_b norm
        scale *= frobenius_norm(array_b)
    return array_a * scale, scale


def singular_value_decomposition(array):
    """
    Return Singular Value Decomposition (SVD) factorization of an array.

    Parameters
    -----------
    array: ndarray
        The 2d-array :math:`\mathbf{A}_{m \times n}`.

    Returns
    -------
    u : ndarray
        Unitary matrix \mathbf{U}_{m \times m}.
    s : ndarray
        The singular values of matrix sorted in descending order.
    v : ndarray
        Unitary matrix \mathbf{V}_{n \times n}.

    See Also
    --------
    `numpy.linalg.svd() <https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.linalg.svd.html>`_
    """

    return np.linalg.svd(array)


def eigenvalue_decomposition(array, two_sided_single=False):
    """
    Compute the eigenvalue decomposition of array
    Decomposes array :math:`A` such that :math:`A = U S {U}^\intercal`.

    Parameters
    ------------
    array: ndarray
       A 2D array who's eigenvalue decomposition is to be calculated.

    two_sided_single : bool
        Set to True when dealing with two-sided single transformation procrustes problems,
        such as two_sided single transformation orthogonal / permutation. When true, array of
        eigenvectors is rearranged according to rows rather than columns, allowing the analysis
        to proceed.

    Returns
    -------
    s : ndarray
        1D array of the eigenvalues of array, sorted from greatest to least.
    v : ndarray
        2D array of eigenvectors of array, sorted according to S.
    """

    # Test whether eigenvalue decomposition is possible on the input array
    if is_diagonalizable(array) is False:
        raise ValueError(
            'The input array is not diagonalizable. The analysis cannot continue.')

    # Find eigenvalues and array of eigenvectors
    s, v = np.linalg.eigh(array)
    # Sort the eigenvalues from greatest to least
    idx = s.argsort()[::-1]

    s = s[idx]
    if two_sided_single:
        # For the two-sided single-transformation problems, we permute rows by idx
        v = v[idx]
    else:
        # For all other given problems, we permute columns by idx
        v = v[:, idx]

    return s, v


def hide_zero_padding(array):
    """
    Return array with zero-padded rows (at the bottom) and columns (on the right) removed.

    Parameters
    ----------
    array: ndarray
        The 2d or 1d array which may or may not contain zero-padd rows and/or columns.
        It is assumed that zero-padded rows are located at the bottom and zero-padded columns
        are located on the right. I.e. all relevant information is contained in upper-left
        block. The rows (from bottom) and columns (from right) are checked until a non-zero
        row and column is reached.

    Returns
    -------
    array : ndarray
        The 2d or 1d array with no padded row and/or columns of zero.
    """

    # check zero rows from bottom to up
    count = array.shape[0] - 1
    while count >= 0:
        # check is stopped when the 1st non-zero row is found
        if np.any(abs(array[count]) > 1.e-8):
            break
        array = np.delete(array, count, axis=0)
        count -= 1

    # for 2d arrays, check zero columns from right to left
    if array.ndim == 2:
        count = array.shape[1] - 1
        while count >= 0:
            # check is stopped when the 1st non-zero column is found
            if np.any(abs(array[:, count]) > 1.e-8):
                break
            array = np.delete(array, count, axis=1)
            count -= 1
    return array


def is_diagonalizable(array):
    """
    Check whether the given array is diagonalizable.

    Parameters
    ------------
    array: ndarray
        A square array for which the diagonalizability is checked.

    Returns
    ---------
    diagonalizable : bool
        True if the array is diagonalizable, otherwise False.
    """

    array = hide_zero_padding(array)

    if array.shape[0] != array.shape[1]:
        raise ValueError(
            'Argument arrays should be a square array! shape={0}'.format(array.shape))

    # SVD decomposition of array
    u, s, vt = singular_value_decomposition(array)
    rank_u = np.linalg.matrix_rank(u)
    rank_a = np.linalg.matrix_rank(array)
    diagonalizable = True
    # If the ranks of u and a are not equal, the eigenvectors cannot span the dimension
    # of the vector space, and the array cannot be diagonalized.
    if rank_u != rank_a:
        diagonalizable = False
    return diagonalizable


def compute_centroid(array):
    """
    Return mean value of array columns.

    Parameters
    ----------
    array: ndarray
        A 2d-array.

    Returns
    -------
    col_mean : float
        the mean value of array columns.

    See Also
    --------
    `np.mean() <https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html>`_
    """

    col_mean = np.mean(array, axis=0)
    return col_mean


def frobenius_norm(array):
    """
    Return the Forbenius norm of array.

    Parameters
    ----------
    array: ndarray
        A 2d-array.

    Returns
    -------
    f_norm : float
        the Forbenius norm of the array.

    See Also
    --------
    `np.linalg.norm() <https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.linalg.norm.html>`_
    """

    f_norm = np.linalg.norm(array)
    return f_norm
