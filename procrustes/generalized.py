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
"""Generalized Procrustes Module."""

import warnings
from typing import List, Optional, Tuple

import numpy as np

from procrustes import orthogonal
from procrustes.utils import _check_arraytypes

__all__ = [
    "generalized",
]


def generalized(
    array_list: List[np.ndarray],
    ref: Optional[np.ndarray] = None,
    tol: float = 1.0e-7,
    n_iter: int = 200,
    check_finite: bool = True,
    handle_missing: bool = False,
) -> Tuple[List[np.ndarray], float]:
    r"""Generalized Procrustes Analysis.

    Parameters
    ----------
    array_list : List
        The list of 2D-array which is going to be transformed. Can contain NaN values or
        masked arrays to indicate missing values when `handle_missing=True`.
    ref : ndarray, optional
        The reference array to initialize the first iteration. If None, the first array in
        `array_list` will be used.
    tol: float, optional
        Tolerance value to stop the iterations.
    n_iter: int, optional
        Number of total iterations.
    check_finite : bool, optional
        If true, convert the input to an array, checking for NaNs or Infs. When
        `handle_missing=True`, this parameter is ignored as NaN values are expected.
    handle_missing : bool, optional
        If True, handle missing values (NaN or masked) using the Albers-Gower algorithm.
        Missing values are iteratively estimated during the alignment process. Default is False
        for backward compatibility.

    Returns
    -------
    array_aligned : List
        A list of transformed arrays with generalized Procrustes analysis. If `handle_missing=True`,
        missing values in the original arrays are filled with estimated values.
    new_distance_gpa: float
        The distance for matching all the transformed arrays with generalized Procrustes analysis.

    Notes
    -----
    Given a set of matrices, :math:`\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k` with
    :math:`k > 2`,  the objective is to minimize in order to superimpose pairs of matrices.

    .. math::
        \min \quad = \sum_{i<j}^{j} {\left\| \mathbf{A}_i \mathbf{T}_i  -
         \mathbf{A}_j \mathbf{T}_j \right\| }^2

    This function implements the Equation (20) and the corresponding algorithm in Gower's paper.

    When `handle_missing=True`, the implementation follows the Albers-Gower algorithm for handling
    missing values in Procrustes analysis [1]_. Missing values are indicated by NaN or numpy masked
    arrays. The algorithm iteratively:

    1. Estimates missing values using current transformations
    2. Performs orthogonal Procrustes alignment on observed values
    3. Updates the consensus configuration

    References
    ----------
    .. [1] Albers, C. J., & Gower, J. C. (2010). A general approach to handling missing values
           in Procrustes analysis. Advances in Data Analysis and Classification, 4(4), 223-237.

    """
    # check input arrays
    _check_arraytypes(*array_list)

    # check finite (skip if handling missing values)
    if check_finite and not handle_missing:
        array_list = [np.asarray_chkfinite(arr) for arr in array_list]

    if n_iter <= 0:
        raise ValueError("Number of iterations should be a positive number.")

    # handle missing values using Albers-Gower algorithm
    if handle_missing:
        return _generalized_with_missing(array_list, ref, tol, n_iter)

    # original implementation (no missing values)
    if ref is None:
        # the first array will be used to build the initial ref
        array_aligned = [array_list[0]] + [
            _orthogonal(arr, array_list[0]) for arr in array_list[1:]
        ]
        ref = np.mean(array_aligned, axis=0)
    else:
        array_aligned = [None] * len(array_list)
        ref = ref.copy()

    distance_gpa = np.inf
    for _ in np.arange(n_iter):
        # align to ref
        array_aligned = [_orthogonal(arr, ref) for arr in array_list]
        # the mean
        new_ref = np.mean(array_aligned, axis=0)
        # todo: double check if the error is defined in the right way
        # the error
        new_distance_gpa = np.square(ref - new_ref).sum()
        if distance_gpa != np.inf and np.abs(new_distance_gpa - distance_gpa) < tol:
            break
        ref = new_ref
        distance_gpa = new_distance_gpa
    return array_aligned, new_distance_gpa


def _orthogonal(arr_a: np.ndarray, arr_b: np.ndarray) -> np.ndarray:
    """Orthogonal Procrustes transformation and returns the transformed array."""
    res = orthogonal(arr_a, arr_b, translate=False, scale=False, unpad_col=False, unpad_row=False)
    return np.dot(res["new_a"], res["t"])


def _get_mask(arr: np.ndarray) -> np.ndarray:
    """Get boolean mask indicating missing values (True for observed, False for missing)."""
    if np.ma.is_masked(arr):
        # For masked arrays, invert the mask (True for observed values)
        return ~np.ma.getmaskarray(arr)
    else:
        # For regular arrays, check for NaN
        return ~np.isnan(arr)


def _extract_data(arr: np.ndarray) -> np.ndarray:
    """Extract data from array, converting masked arrays to regular arrays with NaN."""
    if np.ma.is_masked(arr):
        return np.ma.filled(arr, np.nan)
    return arr.copy()


def _compute_column_means(array_list: List[np.ndarray]) -> List[float]:
    """Compute column means of observed values across all arrays."""
    n_cols = array_list[0].shape[1]
    col_means = []

    # Compute overall mean for fallback
    all_observed = []
    for arr in array_list:
        arr_mask = _get_mask(arr)
        arr_data = _extract_data(arr)
        all_observed.extend(arr_data[arr_mask])
    overall_mean = np.mean(all_observed) if all_observed else 0.0

    for col_idx in range(n_cols):
        col_values = []
        for arr in array_list:
            mask = _get_mask(arr)
            data = _extract_data(arr)
            observed_vals = data[mask[:, col_idx], col_idx]
            col_values.extend(observed_vals)
        if col_values:
            col_means.append(np.mean(col_values))
        else:
            col_means.append(overall_mean)

    return col_means


def _initialize_missing_values(array_list: List[np.ndarray]) -> List[np.ndarray]:
    """Initialize missing values using column means of observed values across all arrays."""
    n_cols = array_list[0].shape[1]
    col_means = _compute_column_means(array_list)

    # Fill missing values using precomputed column means
    initialized_arrays = []
    for arr in array_list:
        arr_copy = _extract_data(arr)
        mask = _get_mask(arr)
        for col_idx in range(n_cols):
            if not mask[:, col_idx].all():  # if column has missing values
                arr_copy[~mask[:, col_idx], col_idx] = col_means[col_idx]
        initialized_arrays.append(arr_copy)

    return initialized_arrays


def _weighted_mean(arrays: List[np.ndarray], masks: List[np.ndarray]) -> np.ndarray:
    """Compute element-wise weighted mean considering only observed values."""
    if not arrays:
        raise ValueError("At least one array is required.")

    # Stack arrays and masks for vectorized computation
    arrays_stacked = np.stack(arrays)  # shape: (k, n, m)
    masks_stacked = np.stack(masks)  # shape: (k, n, m)

    # Compute overall mean of all observed values for fallback
    all_observed = arrays_stacked[masks_stacked]
    fallback_mean = np.mean(all_observed) if all_observed.size > 0 else 0.0

    # For each position, compute mean of observed values
    # Set missing values to 0 for sum calculation, but only count observed
    observed_sum = np.sum(arrays_stacked * masks_stacked, axis=0)
    observed_count = np.sum(masks_stacked, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(
            observed_count > 0,
            observed_sum / observed_count,
            fallback_mean
        )

    return result


def _orthogonal_with_mask(arr_a: np.ndarray, arr_b: np.ndarray, mask_a: np.ndarray) -> np.ndarray:
    """
    Perform weighted orthogonal Procrustes considering only observed values.

    This computes the optimal orthogonal transformation by solving:
    min ||W * (A*Q - B)||^2 where W is a diagonal weight matrix based on mask_a.
    """
    # Create weight matrix from mask (observed = 1, missing = 0)
    weights = mask_a.astype(float).flatten()

    # If no observed values, return identity transformation
    if not weights.any():
        return arr_a

    # Weight the matrices element-wise
    weighted_a = arr_a * mask_a
    weighted_b = arr_b * mask_a

    # Compute weighted cross-product matrix
    cross_product = np.dot(weighted_a.T, weighted_b)

    # SVD to find optimal rotation
    try:
        u, _, vt = np.linalg.svd(cross_product)
        q_opt = np.dot(u, vt)

        # Apply transformation
        return np.dot(arr_a, q_opt)
    except np.linalg.LinAlgError:
        # If SVD fails, issue a warning and return original array
        warnings.warn(
            "SVD failed during orthogonal Procrustes transformation. "
            "Returning original array without transformation."
        )
        return arr_a


def _generalized_with_missing(
    array_list: List[np.ndarray],
    ref: Optional[np.ndarray],
    tol: float,
    n_iter: int,
) -> Tuple[List[np.ndarray], float]:
    """
    Generalized Procrustes Analysis with missing value handling.

    Implements the Albers-Gower algorithm for GPA with missing values.
    """
    # Extract masks and data
    masks = [_get_mask(arr) for arr in array_list]
    arrays_filled = _initialize_missing_values(array_list)

    # Initialize reference
    if ref is None:
        # Use first array to build initial reference
        array_aligned = [arrays_filled[0]] + [
            _orthogonal_with_mask(arr, arrays_filled[0], mask)
            for arr, mask in zip(arrays_filled[1:], masks[1:])
        ]
        ref = _weighted_mean(array_aligned, masks)
    else:
        array_aligned = [arr.copy() for arr in arrays_filled]
        # Handle missing values in ref if present
        ref_mask = _get_mask(ref)
        ref_filled = _extract_data(ref)
        if not ref_mask.all():
            # Fill missing values in ref with mean of observed values
            ref_mean = np.mean(ref_filled[ref_mask]) if ref_mask.any() else 0.0
            ref_filled[~ref_mask] = ref_mean
        ref = ref_filled

    distance_gpa = np.inf

    for _ in range(n_iter):
        # Step 1: Align each array to reference using only observed values
        array_aligned_new = []
        for arr, mask in zip(arrays_filled, masks):
            aligned = _orthogonal_with_mask(arr, ref, mask)
            array_aligned_new.append(aligned)

        # Step 2: Update reference as weighted mean of aligned arrays
        new_ref = _weighted_mean(array_aligned_new, masks)

        # Step 3: Update missing values in arrays with values from aligned arrays
        arrays_filled_new = []
        for arr_orig, arr_aligned, mask in zip(array_list, array_aligned_new, masks):
            arr_updated = _extract_data(arr_orig)
            # Fill missing values with estimates from aligned array
            arr_updated[~mask] = arr_aligned[~mask]
            arrays_filled_new.append(arr_updated)

        arrays_filled = arrays_filled_new

        # Compute convergence criterion
        new_distance_gpa = np.sum((ref - new_ref) ** 2)

        # Check convergence
        if distance_gpa != np.inf and np.abs(new_distance_gpa - distance_gpa) < tol:
            array_aligned = array_aligned_new
            break

        ref = new_ref
        distance_gpa = new_distance_gpa
        array_aligned = array_aligned_new

    # Compute actual GPA error: sum of squared distances between aligned arrays and reference
    final_ref = _weighted_mean(array_aligned, masks)
    gpa_error = sum(np.sum((arr - final_ref) ** 2) for arr in array_aligned)

    return array_aligned, gpa_error
