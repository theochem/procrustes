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
"""Utility Module."""

import numpy as np

__all__ = [
    "compute_error",
    "setup_input_arrays",
    "ProcrustesResult",
]


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


def _translate_array(array_a, array_b=None, weight=None):
    """
    Return translated array_a and translation vector.

    Columns of both arrays will have mean zero.

    Parameters
    ----------
    array_a : ndarray
        The 2d-array to translate.
    array_b : ndarray, default=None
        The 2d-array to translate array_a based on.
    weight : ndarray
        The weight vector. Default=None.

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
    if weight is not None:
        if weight.ndim != 1:
            raise ValueError("The weight should be a 1d row vector.")
        if not (weight >= 0).all():
            raise ValueError("The elements of the weight should be non-negative.")

    centroid_a = np.average(array_a, axis=0, weights=weight)
    if array_b is not None:
        # translation vector to b centroid
        centroid_a -= np.average(array_b, axis=0, weights=weight)
    return array_a - centroid_a, -1 * centroid_a


def _scale_array(array_a, array_b=None):
    """
    Return scaled/normalized array_a and scaling vector.

    Parameters
    ----------
    array_a : ndarray
        The 2d-array to scale
    array_b : ndarray, default=None
        The 2d-array to scale array_a based on.

    Returns
    -------
    scaled_a, ndarray
        If array_b is None, array_a is normalized using the Frobenius norm.
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
        If True, zero columns (values less than 1e-8) on the right side will be removed.
        Default=True.
    remove_zero_row : bool, optional
        If True, zero rows (values less than 1e-8) on the bottom will be removed.
        Default=True.
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


def compute_error(array_a, array_b, array_u, array_v=None):
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
                       pad_mode, translate, scale, check_finite, weight):
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
        If True, zero columns (values less than 1e-8) on the right side will be removed.
        Default=True.
    remove_zero_row : bool, optional
        If True, zero rows (values less than 1e-8) on the bottom will be removed. Default=True.
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
        If true, then translate both arrays :math:`A, B` to the origin, ie columns of the arrays
        will have mean zero.
    scale :
        If True, both arrays are normalized to one with respect to the Frobenius norm, ie
        :math:`Tr(A^T A) = 1`.
    check_finite : bool
        If true, then checks if both arrays :math:`A, B` are numpy arrays and two-dimensional.
    weight : A list of ndarray or ndarray
        A list of the weight arrays or one numpy array. When only on numpy array provided,
        it is assumed that the two arrays :math:`A` and :math:`B` share the same weight matrix.
        Default=None.

    Returns
    -------
    (ndarray, ndarray) :
        Returns the padded arrays, in that they have the same matrix dimensions.

    """
    array_a = _setup_input_array_lower(array_a, None, check_finite, translate,
                                       scale, remove_zero_col, remove_zero_row, weight)
    array_b = _setup_input_array_lower(array_b, None, check_finite, translate,
                                       scale, remove_zero_col, remove_zero_row, weight)
    return _zero_padding(array_a, array_b, pad_mode)


def setup_input_arrays_multi(array_list, array_ref, remove_zero_col, remove_zero_row,
                             pad_mode, translate, scale, check_finite, weight):
    r"""
    Check and process array inputs for the Procrustes transformation routines.

    Parameters
    ----------
    array_list : List
        A list of 2D arrays that being transformed.
    array_ref : ndarray
        The 2D reference array :math:`B`.
    remove_zero_col : bool, optional
        If True, zero columns (values less than 1e-8) on the right side will be removed.
        Default=True.
    remove_zero_row : bool, optional
        If True, zero rows (values less than 1e-8) on the bottom will be removed. Default=True.
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
        If true, then translate both arrays :math:`A, B` to the origin, ie columns of the arrays
        will have mean zero.
    scale :
        If True, both arrays are normalized to one with respect to the Frobenius norm, ie
        :math:`Tr(A^T A) = 1`.
    check_finite : bool
        If true, then checks if both arrays :math:`A, B` are numpy arrays and two-dimensional.
    weight : A list of ndarray or ndarray
        A list of the weight arrays or one numpy array. When only on numpy array provided,
        it is assumed that the two arrays :math:`A` and :math:`B` share the same weight matrix.
        Default=None.

    Returns
    -------
    List of arrays :
        Returns the padded arrays, in that they have the same matrix dimensions.
    """
    array_list_new = [_setup_input_array_lower(array_a=arr,
                                               array_ref=array_ref,
                                               check_finite=check_finite,
                                               translate=translate,
                                               scale=scale,
                                               remove_zero_col=remove_zero_col,
                                               remove_zero_row=remove_zero_row,
                                               weight=weight)
                      for arr in array_list]
    arr_shape = np.array([arr.shape for arr in array_list_new])
    array_b = np.ones(np.max(arr_shape, axis=0), dtype=int)
    array_list_new = [_zero_padding(arr, array_b, pad_mode=pad_mode) for arr in array_list_new]
    return array_list_new


def _setup_input_array_lower(array_a, array_ref, check_finite, translate,
                             scale, remove_zero_col, remove_zero_row, weight):
    """Pre-processing the matrices with translation, scaling."""
    _check_arraytypes(array_a)
    if check_finite:
        array_a = np.asarray_chkfinite(array_a)
        # Sometimes arrays already have zero padding that messes up zero padding below.
    array_a = _hide_zero_padding(array_a, remove_zero_col, remove_zero_row)
    if translate:
        array_a, _ = _translate_array(array_a, array_ref, weight)
    # scale the matrix when translate is False, but weight is True
    else:
        if weight is not None:
            array_a = np.dot(np.diag(weight), array_a)

    if scale:
        array_a, _ = _scale_array(array_a, array_ref)
    return array_a


def _check_arraytypes(*args):
    r"""Check array input types to Procrustes transformation routines."""
    if any(not isinstance(arr_x, np.ndarray) for arr_x in args):
        raise TypeError("Matrix inputs must be NumPy arrays")
    if any(x.ndim != 2 for x in args):
        raise TypeError("Matrix inputs must be 2-dimensional arrays")


class ProcrustesResult(dict):
    r"""Represents the Procrustes analysis result.

    Attributes
    ----------
    new_a : ndarray
        The translated/scaled numpy ndarray :math:`\mathbf{A}`.
    new_b : ndarray
        The translated/scaled numpy ndarray :math:`\mathbf{B}`.
    array_u : ndarray
        The right hand side optimum transformation matrix.
    array_p : ndarray
        The left hand side transformation matrix for two-sided Procrustes problem with
        two transformation.
    array_q : ndarray
        The right hand side transformation matrix for two-sided Procrustes problem with
        two transformation.
    error : float
        Two-sided permutation Procrustes error.

    """

    # modification on https://github.com/scipy/scipy/blob/v1.4.1/scipy/optimize/optimize.py#L77-L132
    def __getattr__(self, name):
        """Deal with attributes which it doesn't explicitly manage."""
        try:
            return self[name]
        # Not using raise from makes the traceback inaccurate, because the message implies there
        # is a bug in the exception-handling code itself, which is a separate situation than
        # wrapping an exception
        # W0707 from http://pylint.pycqa.org/en/latest/technical_reference/features.html
        except KeyError as ke_info:
            raise AttributeError(name) from ke_info

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        """Return a human friendly representation."""
        if self.keys():
            max_len = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(max_len) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        """Provide basic customization of module attribute access with a list."""
        return list(self.keys())
