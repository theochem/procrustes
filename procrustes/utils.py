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
"""
Utility Module.

Functions
---------
error : Calculates the squared distance between transformed matrices and reference matrix.
setup_input_arrays : Setups up the arrays for all Procrustes methods.  It checks if the
                    inputs are all numpy arrays, and two-dimensional.  It does zero-padding to make
                    sure all arrays are of the same matrix dimensions and translates/scales the
                    arrays if specified.

"""
from copy import deepcopy
import itertools as it

import numpy as np

__all__ = ["error", "setup_input_arrays", "kopt_heuristic_single", "kopt_heuristic_double"]


def _zero_padding(array_a, array_b, pad_mode="row-col"):
    r"""
    Return arrays padded with rows and/or columns of zero.

    Parameters
    ----------
    array_a : ndarray
        The 2d-array :math:`\mathbf{A}_{n_a \times m_a}`.
    array_b : ndarray
        The 2d-array :math:`\mathbf{B}_{n_b \times m_b}`.
    pad_mode : str
        Specifying how to pad the arrays. Should be one of

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

    Returns
    -------
    padded_a : ndarray
        Padded array_a.
    padded_b : ndarray
        Padded array_b.

    """
    # sanity checks
    if not isinstance(array_a, np.ndarray) or not isinstance(array_b, np.ndarray):
        raise ValueError("Arguments array_a & array_b should be numpy arrays.")
    if array_a.ndim != 2 or array_b.ndim != 2:
        raise ValueError("Arguments array_a & array_b should be 2D arrays.")

    if array_a.shape == array_b.shape and array_a.shape[0] == array_a.shape[1]:
        # special case of square arrays, mode is set to None so that array_a & array_b are returned.
        pad_mode = None

    if pad_mode == "square":
        # calculate desired dimension of square array
        (a_n1, a_m1), (a_n2, a_m2) = array_a.shape, array_b.shape
        dim = max(a_n1, a_n2, a_m1, a_m2)
        # padding rows to have both arrays have dim rows
        if a_n1 < dim:
            array_a = np.pad(array_a, [[0, dim - a_n1], [0, 0]], "constant", constant_values=0)
        if a_n2 < dim:
            array_b = np.pad(array_b, [[0, dim - a_n2], [0, 0]], "constant", constant_values=0)
        # padding columns to have both arrays have dim columns
        if a_m1 < dim:
            array_a = np.pad(array_a, [[0, 0], [0, dim - a_m1]], "constant", constant_values=0)
        if a_m2 < dim:
            array_b = np.pad(array_b, [[0, 0], [0, dim - a_m2]], "constant", constant_values=0)

    if pad_mode in ["row", "row-col"]:
        # padding rows to have both arrays have the same number of rows
        diff = array_a.shape[0] - array_b.shape[0]
        if diff < 0:
            array_a = np.pad(array_a, [[0, -diff], [0, 0]], "constant", constant_values=0)
        else:
            array_b = np.pad(array_b, [[0, diff], [0, 0]], "constant", constant_values=0)

    if pad_mode in ["col", "row-col"]:
        # padding columns to have both arrays have the same number of columns
        diff = array_a.shape[1] - array_b.shape[1]
        if diff < 0:
            array_a = np.pad(array_a, [[0, 0], [0, -diff]], "constant", constant_values=0)
        else:
            array_b = np.pad(array_b, [[0, 0], [0, diff]], "constant", constant_values=0)

    return array_a, array_b


def _translate_array(array_a, array_b=None):
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
    array_a : ndarray
        If array_b is None, array_a is translated to origin using its centroid.
        If array_b is given, array_a is translated to centroid of array_b (the centroid of
        translated array_a will centroid with the centroid array_b).
    centroid : float
        If array_b is given, the centroid is returned.

    """
    # The mean is strongly affected by outliers and is not a robust estimator for central location
    # see https://docs.python.org/3.6/library/statistics.html?highlight=mean#statistics.mean
    centroid = np.mean(array_a, axis=0)
    if array_b is not None:
        # translation vector to b centroid
        centroid -= np.mean(array_b, axis=0)
    return array_a - centroid, -centroid


def _scale_array(array_a, array_b=None):
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
    scaled_a, ndarray
        If array_b is None, array_a is scaled to match norm of unit sphere using array_a"s
        Frobenius norm.
        If array_b is given, array_a is scaled to match array_b"s norm (the norm of array_a
        will be equal norm of array_b).
    scale : float
        The scaling factor to match array_b norm.

    """
    # scaling factor to match unit sphere
    scale = 1. / np.linalg.norm(array_a)
    if array_b is not None:
        # scaling factor to match array_b norm
        scale *= np.linalg.norm(array_b)
    return array_a * scale, scale


def _hide_zero_padding(array_a, remove_zero_col=True, remove_zero_row=True, tol=1.0e-8):
    r"""
    Return array with zero-padded rows (bottom) and columns (right) removed.

    Parameters
    ----------
    array_a : ndarray
        The initial array.
    remove_zero_col : bool, optional
        If True, the zero columns on the right side will be removed. Default=True.
    remove_zero_row : bool, optional
        If True, the zero rows on the top will be removed. Default=True.
    tol : float
        Tolerance value.

    Returns
    -------
    new_A : ndarray
        Array, with either near zero columns and/or zero rows are removed.

    """
    # Input checking
    if array_a.ndim > 2:
        raise TypeError("Matrix inputs must be 1- or 2- dimensional arrays")
    # Check zero rows from bottom to top
    if remove_zero_row:
        num_row = array_a.shape[0]
        tmp_a = array_a[..., np.newaxis] if array_a.ndim == 1 else array_a
        for array_v in tmp_a[::-1]:
            if any(abs(i) > tol for i in array_v):
                break
            num_row -= 1
        array_a = array_a[:num_row]
    # Cut off zero rows
    if remove_zero_col:
        if array_a.ndim == 2:
            # Check zero columns from right to left
            col_m = array_a.shape[1]
            for array_v in array_a.T[::-1]:
                if any(abs(i) > tol for i in array_v):
                    break
                col_m -= 1
            # Cut off zero columns
            array_a = array_a[:, :col_m]
    return array_a


def error(array_a, array_b, array_u, array_v=None):
    r"""
    Return the single- or double- sided Procrustes/norm-squared error.

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
    array_a : npdarray
        The 2D array :math:`A` being transformed.
    array_b : npdarray
        The 2D reference array :math:`B`.
    array_u : ndarray
        The 2D array representing the transformation :math:`\mathbf{U}`.
    array_v : ndarray, optional
        The 2D array representing the transformation :math:`\mathbf{V}`. If provided, it will
        compute the double-sided Procrustes error. Otherwise, will compute the single-sided
        Procrustes error.

    Returns
    -------
    error : float
        The squared value of the distance between transformed array and reference.

    """
    array_e = np.dot(array_a, array_u) if array_v is None \
        else np.dot(np.dot(array_u.T, array_a), array_v)
    array_e -= array_b
    return np.trace(np.dot(array_e.T, array_e))


def setup_input_arrays(array_a, array_b, remove_zero_col, remove_zero_row,
                       pad_mode, translate, scale, check_finite):
    r"""
    Check and process array inputs for the Procrustes transformation routines.

    Usually, the precursor step before all Procrustes methods.

    Parameters
    ----------
    array_a : npdarray
        The 2D array :math:`A` being transformed.
    array_b : npdarray
        The 2D reference array :math:`B`.
    remove_zero_col : bool, optional
        If True, the zero columns on the right side will be removed. Default=True.
    remove_zero_row : bool, optional
        If True, the zero rows on the top will be removed. Default=True.
    pad_mode : str
        Specifying how to pad the arrays. Should be one of

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
    translate : bool
        If true, then translate both arrays :math:`A, B` to the origin.
    scale :
        If true, then both arrays :math:`A, B` are scaled/normalized.
    check_finite : bool
        If true, then checks if both arrays :math:`A, B` are numpy arrays and two-dimensional.

    Returns
    -------
    (ndarray, ndarray) :
        Returns the padded arrays, in that they have the same matrix dimensions.

    """
    _check_arraytypes(array_a, array_b)
    if check_finite:
        array_a = np.asarray_chkfinite(array_a)
        array_b = np.asarray_chkfinite(array_b)
    # Sometimes arrays already have zero padding that messes up zero padding below.
    array_a = _hide_zero_padding(array_a, remove_zero_col, remove_zero_row)
    array_b = _hide_zero_padding(array_b, remove_zero_col, remove_zero_row)
    if translate:
        array_a, _ = _translate_array(array_a)
        array_b, _ = _translate_array(array_b)
    if scale:
        array_a, _ = _scale_array(array_a)
        array_b, _ = _scale_array(array_b)
    return _zero_padding(array_a, array_b, pad_mode)


def _check_arraytypes(*args):
    r"""Check array input types to Procrustes transformation routines."""
    if any(not isinstance(arr_x, np.ndarray) for arr_x in args):
        raise TypeError("Matrix inputs must be NumPy arrays")
    if any(x.ndim != 2 for x in args):
        raise TypeError("Matrix inputs must be 2-dimensional arrays")


def kopt_heuristic_single(array_a, array_b, ref_error, perm=None, kopt_k=3, kopt_tol=1.e-8):
    r"""K-opt heuristic to improve the accuracy for two-sided permutation with one transformation.

    Perform k-opt local search with every possible valid combination of the swapping mechanism.

    Parameters
    ----------
    array_a : ndarray
        The array to be permuted.
    array_b : ndarray
        The reference array.
    ref_error : float
        The reference error value.
    perm : ndarray, optional
        The permutation array which remains to be processed with k-opt local search. Default is the
        identity matrix with the same shape of array_a.
    kopt_k : int, optional
        Defines the oder of k-opt heuristic local search. For example, kopt_k=3 leads to a local
        search of 3 items and kopt_k=2 only searches for two items locally. Default=3.
    kopt_tol : float, optional
        Tolerance value to check if k-opt heuristic converges. Default=1.e-8.

    Returns
    -------
    perm : ndarray
        The permutation array after optimal heuristic search.
    kopt_error : float
        The error distance of two arrays with the updated permutation array.
    """
    if kopt_k < 2:
        raise ValueError("Kopt_k value must be a integer greater than 2.")
    # if perm is not specified, use the identity matrix as default
    if perm is None:
        perm = np.identity(np.shape(array_a)[0])
    num_row = perm.shape[0]
    kopt_error = ref_error
    # all the possible row-wise permutations
    for comb in it.combinations(np.arange(num_row), r=kopt_k):
        for comb_perm in it.permutations(comb, r=kopt_k):
            if comb_perm != comb:
                perm_kopt = deepcopy(perm)
                perm_kopt[comb, :] = perm_kopt[comb_perm, :]
                e_kopt_new = error(array_a, array_b, perm_kopt, perm_kopt)
                if e_kopt_new < kopt_error:
                    perm = perm_kopt
                    kopt_error = e_kopt_new
                    if kopt_error <= kopt_tol:
                        break
    return perm, kopt_error


def kopt_heuristic_double(array_m, array_n, ref_error,
                          perm_p=None, perm_q=None,
                          kopt_k=3, kopt_tol=1.e-8):
    r"""
    K-opt kopt for regular two-sided permutation Procrustes to improve the accuracy.

    Perform k-opt local search with every possible valid combination of the swapping mechanism for
    regular 2-sided permutation Procrustes.

    Parameters
    ----------
    array_m : ndarray
        The array to be permuted.
    array_n : ndarray
        The reference array.
    ref_error : float
        The reference error value.
    perm_p : ndarray, optional
        The left permutation array which remains to be processed with k-opt local search. Default
        is the identity matrix with the same shape of array_m.
    perm_q : ndarray, optional
        The right permutation array which remains to be processed with k-opt local search. Default
        is the identity matrix with the same shape of array_m.
    kopt_k : int, optional
        Defines the oder of k-opt heuristic local search. For example, kopt_k=3 leads to a local
        search of 3 items and kopt_k=2 only searches for two items locally. Default=3.
    kopt_tol : float, optional
        Tolerance value to check if k-opt heuristic converges. Default=1.e-8.

    Returns
    -------
    perm_kopt_p : ndarray
        The left permutation array after optimal heuristic search.
    perm_kopt_q : ndarray
        The right permutation array after optimal heuristic search.
    kopt_error : float
        The error distance of two arrays with the updated permutation array.
    """
    if kopt_k < 2:
        raise ValueError("Kopt_k value must be a integer greater than 2.")
    # if perm_p is not specified, use the identity matrix as default
    if perm_p is None:
        perm_p = np.identity(np.shape(array_m)[0])
    # if perm_p is not specified, use the identity matrix as default
    if perm_q is None:
        perm_q = np.identity(np.shape(array_m)[0])

    num_row_left = perm_p.shape[0]
    num_row_right = perm_q.shape[0]
    kopt_error = ref_error
    # the left hand side permutation
    # pylint: disable=too-many-nested-blocks
    for comb_left in it.combinations(np.arange(num_row_left), r=kopt_k):
        for comb_perm_left in it.permutations(comb_left, r=kopt_k):
            if comb_perm_left != comb_left:
                perm_kopt_left = deepcopy(perm_p)
                # the right hand side permutation
                for comb_right in it.combinations(np.arange(num_row_right), r=kopt_k):
                    for comb_perm_right in it.permutations(comb_right, r=kopt_k):
                        if comb_perm_right != comb_right:
                            perm_kopt_right = deepcopy(perm_q)
                            perm_kopt_right[comb_right, :] = perm_kopt_right[comb_perm_right, :]
                            e_kopt_new_right = error(array_n, array_m, perm_p.T, perm_kopt_right)
                            if e_kopt_new_right < kopt_error:
                                perm_q = perm_kopt_right
                                kopt_error = e_kopt_new_right
                                if kopt_error <= kopt_tol:
                                    break

                perm_kopt_left[comb_left, :] = perm_kopt_left[comb_perm_left, :]
                e_kopt_new_left = error(array_n, array_m, perm_kopt_left.T, perm_q)
                if e_kopt_new_left < kopt_error:
                    perm_p = perm_kopt_left
                    kopt_error = e_kopt_new_left
                    if kopt_error <= kopt_tol:
                        break

    return perm_p, perm_q, kopt_error
