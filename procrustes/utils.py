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
"""Utility Module."""

import copy

import numpy as np

import itertools as it


def zero_padding(array_a, array_b, pad_mode='row-col'):
    r"""
    Return arrays padded with rows and/or columns of zero.

    Parameters
    ----------
    array_a : ndarray
        The 2d-array :math:`\mathbf{A}_{n_a \times m_a}`.
    array_b : ndarray
        The 2d-array :math:`\mathbf{B}_{n_b \times m_b}`.
    pad_mode : str
        Specifying how to padd arrays. Options:

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
             dimension, i.e. :math:`\text{max}(n_a, m_a, n_b, m_b)`.

    Returns
    -------
    padded_a, padded_b : ndarray, ndarray
        Padded array_a and array_b arrays.
    """

    # sanity checks
    if not isinstance(array_a, np.ndarray) or not isinstance(array_b, np.ndarray):
        raise ValueError('Arguments array_a & array_b should be numpy arrays.')
    if array_a.ndim != 2 or array_b.ndim != 2:
        raise ValueError('Arguments array_a & array_b should be 2D arrays.')

    if array_a.shape == array_b.shape:
        # special case of square arrays, mode is set to None so that array_a & array_b are returned.
        pad_mode = None

    if pad_mode == 'square':
        # calculate desired dimension of square array
        (n1, m1), (n2, m2) = array_a.shape, array_b.shape
        dim = max(n1, n2, m1, m2)
        # padding rows to have both arrays have dim rows
        if n1 < dim:
            array_a = np.pad(array_a, [[0, dim - n1], [0, 0]], 'constant', constant_values=0)
        if n2 < dim:
            array_b = np.pad(array_b, [[0, dim - n2], [0, 0]], 'constant', constant_values=0)
        # padding columns to have both arrays have dim columns
        if m1 < dim:
            array_a = np.pad(array_a, [[0, 0], [0, dim - m1]], 'constant', constant_values=0)
        if m2 < dim:
            array_b = np.pad(array_b, [[0, 0], [0, dim - m2]], 'constant', constant_values=0)

    if pad_mode == 'row' or pad_mode == 'row-col':
        # padding rows to have both arrays have the same number of rows
        diff = array_a.shape[0] - array_b.shape[0]
        if diff < 0:
            array_a = np.pad(array_a, [[0, -diff], [0, 0]], 'constant', constant_values=0)
        else:
            array_b = np.pad(array_b, [[0, diff], [0, 0]], 'constant', constant_values=0)

    if pad_mode == 'col' or pad_mode == 'row-col':
        # padding columns to have both arrays have the same number of columns
        diff = array_a.shape[1] - array_b.shape[1]
        if diff < 0:
            array_a = np.pad(array_a, [[0, 0], [0, -diff]], 'constant', constant_values=0)
        else:
            array_b = np.pad(array_b, [[0, 0], [0, diff]], 'constant', constant_values=0)

    return array_a, array_b


def translate_array(array_a, array_b=None):
    """
    Return translated array_a and translation vector.

    Parameters
    ----------
    array_a : ndarray
        The 2d-array to translate.
    array_b : ndarray, default=None
        The 2d-array to translate array_a based on.

    Returns
    -------
    ndarray, ndarray
        If array_b is None, array_a is translated to origin using its centroid.
        If array_b is given, array_a is translated to centroid of array_b (the centroid of
        translated array_a will centroid with the centroid array_b).
    """
    # The mean is strongly affected by outliers and is not a robust estimator for central location
    # see https://docs.python.org/3.6/library/statistics.html?highlight=mean#statistics.mean
    centroid = np.mean(array_a, axis=0)
    if array_b is not None:
        # translation vector to b centroid
        centroid -= np.mean(array_b, axis=0)
    return array_a - centroid, -centroid


def scale_array(array_a, array_b=None):
    """
    Return scaled array_a and scaling vector.

    Parameters
    ----------
    array_a : ndarray
        The 2d-array to scale
    array_b : ndarray, default=None
        The 2d-array to scale array_a based on.

    Returns
    -------
    ndarray, ndarray
        If array_b is None, array_a is scaled to match norm of unit sphere using array_a's
        Frobenius norm.
        If array_b is given, array_a is scaled to match array_b's norm (the norm of array_a
        will be equal norm of array_b).
    """
    # scaling factor to match unit sphere
    scale = 1. / np.linalg.norm(array_a)
    if array_b is not None:
        # scaling factor to match array_b norm
        scale *= np.linalg.norm(array_b)
    return array_a * scale, scale


def singular_value_decomposition(array):
    r"""
    Return singular value decomposition (SVD) factorization of an array.

    .. math::
      \mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\dagger

    Parameters
    ----------
    array: ndarray
        The 2d-array :math:`\mathbf{A}_{m \times n}` to factorize.

    Returns
    -------
    u : ndarray
        Unitary matrix :math:`\mathbf{U}_{m \times m}`.
    s : ndarray
        The singular values of matrix sorted in descending order.
    v : ndarray
        Unitary matrix :math:`\mathbf{V}_{n \times n}`.
    """
    return np.linalg.svd(array)


def eigendecomposition(A, permute_rows=False):
    r"""
    Compute the eigenvalue decomposition of an array.

    .. math::
      \mathbf{A} = \mathbf{U} \mathbf{S} \mathbf{U}^\dagger

    Parameters
    ----------
    array: ndarray
       The 2D array to decompose.
    permute_rows : bool, default = False
        If True, permute rows of eigenvectors according to the greatest
        to least eigenvalues. Otherwise, permute columns.

    Returns
    -------
    s : ndarray
        The 1D array of the eigenvalues, sorted from greatest to least.
    V : ndarray
        The 2D array of eigenvectors, sorted according to greatest to
        least eigenvalues.

    """
    # find eigenvalues & eigenvectors
    s, V = np.linalg.eigh(A)
    # get index of sorted eigenvalues from largest to smallest
    idx = s.argsort()[::-1]
    # Return permuted eigenvalues & eigenvectors
    return s[idx], V[idx] if permute_rows else V[:, idx]


def hide_zero_padding(A, remove_zero_col=True, remove_zero_row=True, tol=1.0e-8):
    r"""
    Return array with zero-padded rows (bottom) and columns (right) removed.

    Parameters
    ----------
    A : ndarray
    remove_zero_col : bool, optional
        If True, the zero columns on the right side will be removed.
        Default=True.
    remove_zero_row : bool, optional
        If True, the zero rows on the top will be removed. Default=True.

    Returns
    -------
    new_A : ndarray

    """
    # Input checking
    if A.ndim > 2:
        raise TypeError("Matrix inputs must be 1- or 2- dimensional arrays")
    # Check zero rows from bottom to top
    if remove_zero_row:
        n = A.shape[0]
        tmpA = A[..., np.newaxis] if A.ndim == 1 else A
        for v in tmpA[::-1]:
            if any(abs(i) > tol for i in v):
                break
            n -= 1
        A = A[:n]
    # Cut off zero rows
    if remove_zero_col:
        if A.ndim == 2:
            # Check zero columns from right to left
            m = A.shape[1]
            for v in A.T[::-1]:
                if any(abs(i) > tol for i in v):
                    break
                m -= 1
            # Cut off zero columns
            A = A[:, :m]
    return A


def is_diagonalizable(array):
    """
    Check whether the given array is diagonalizable.

    Parameters
    ----------
    array: ndarray
        A square array for which the diagonalizability is checked.

    Returns
    -------
    diagonalizable : bool
        True if the array is diagonalizable, otherwise False.
    """
    # check array is square
    array = hide_zero_padding(array)
    if array.shape[0] != array.shape[1]:
        raise ValueError('Argument array should be a square array! shape={0}'.format(array.shape))
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


def error(A, B, U, V=None):
    r"""
    Return the single- or double- sided Procrustes error.

    The single sided error is defined as

    .. math::
       \text{Tr}\left[\left(\mathbf{AU} - \mathbf{B}\right)^\dagger
                       \left(\mathbf{AU} - \mathbf{B}\right)\right]

    The double sided Procrustes error is defined as

    .. math::
       \text{Tr}\left[
            \left(\mathbf{U}_1^\dagger \mathbf{A}\mathbf{U}_2 - \mathbf{B}\right)^\dagger
            \left(\mathbf{U}_1^\dagger \mathbf{A}\mathbf{U}_2 - \mathbf{B}\right)\right]

    Parameters
    ----------
    a : npdarray
        The array being transformed.
    b : npdarray
        The reference array.
    u : ndarray
        The 2D array representing the transformation :math:`\mathbf{U}`.
    v : ndarray, optional
        The 2D array representing the transformation :math:`\mathbf{V}`.
        If provided, will compute the double-sided Procrustes error.
        Otherwise, will compute the single-sided Procrustes error.

    Returns
    -------
    error : float

    """
    E = np.dot(A, U) if V is None else np.dot(np.dot(U.T, A), V)
    E -= B
    return np.trace(np.dot(E.T, E))


def optimal_heuristic(perm, A, B, ref_error, k_opt=3):
    r"""
    Perform k-opt local search with every possible valid combination of the swapping mechanism which
    also includes 2-opt heuristic.

    Parameters
    ----------
    perm : np.ndarray
        The permutation array which remains to be processed with k-opt local search.
    A : np.ndarray
        The array to be permuted.
    B : np.ndarray
        The reference array.
    ref_error : float
        The reference error value.
    k_opt : int, optional
        Order of local search. Default=3.

    Returns
    -------
    perm : np.ndarray
        The permutation array after optimal heuristic search.
    kopt_error : float
        The error distance of two arrays with the updated permutation array.

    """
    if k_opt < 2:
        raise ValueError("K_opt value must be a integer greater than 2.")
    num_row = perm.shape[0]
    kopt_error = ref_error
    # all the possible row-wise permutations
    for comb in it.combinations(np.arange(num_row), r=k_opt):
        for comb_perm in it.permutations(comb, r=k_opt):
            if comb_perm != comb:
                perm_kopt = copy.deepcopy(perm)
                perm_kopt[comb, :] = perm_kopt[comb_perm, :]
                e_kopt_new = error(A, B, perm_kopt, perm_kopt)
                if e_kopt_new < kopt_error:
                    perm = perm_kopt
                    kopt_error = e_kopt_new
                    if kopt_error == 0:
                        break
    return perm, kopt_error


def _get_input_arrays(A, B, remove_zero_col, remove_zero_row,
                      pad_mode, translate, scale, check_finite):
    r"""Check and process array inputs to Procrustes transformation routines."""
    _check_arraytypes(A, B)
    if check_finite:
        A = np.asarray_chkfinite(A)
        B = np.asarray_chkfinite(B)
    A = hide_zero_padding(A, remove_zero_col, remove_zero_row)
    B = hide_zero_padding(B, remove_zero_col, remove_zero_row)
    if translate:
        A, _ = translate_array(A)
        B, _ = translate_array(B)
    if scale:
        A, _ = scale_array(A)
        B, _ = scale_array(B)
    return zero_padding(A, B, pad_mode)


def _check_arraytypes(*args):
    r"""Check array input types to Procrustes transformation routines."""
    if any(not isinstance(x, np.ndarray) for x in args):
        raise TypeError("Matrix inputs must be NumPy arrays")
    if any(x.ndim != 2 for x in args):
        raise TypeError("Matrix inputs must be 2-dimensional arrays")


def _check_rank(A):
    r"""Check whether the given array is diagonalizable."""
    A = hide_zero_padding(A)
    U, _, _ = np.linalg.svd(A)
    if np.linalg.matrix_rank(U) != np.linalg.matrix_rank(A):
        raise np.linalg.LinAlgError("Matrix cannot be diagonalized")
