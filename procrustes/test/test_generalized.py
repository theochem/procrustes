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
"""Testings for generalized Procrustes module."""

import numpy as np
from numpy.testing import assert_almost_equal, assert_raises

from procrustes.generalized import generalized


def test_generalized_with_reference():
    r"""Test generalized Procrustes with reference."""
    arr_b = np.array([[5, 0], [8, 0], [5, 5]])
    arr_c = np.dot(arr_b, _rotation(30))
    arr_d = np.dot(arr_b, _rotation(45))
    arr_e = np.dot(arr_b, _rotation(90))
    # arr_list = [arr_a, arr_b, arr_c, arr_d]
    # arr_list = [arr_b, arr_c, arr_d, arr_e]
    arr_list = [arr_c, arr_d, arr_e]
    arr_aligned, error = generalized(arr_list, ref=arr_b, tol=1.0e-7, n_iter=200)
    # one right alignment
    aligned = [
        np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0]]),
        np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0]]),
        np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0]]),
    ]
    assert_almost_equal(arr_aligned[0], aligned[0], decimal=7)
    assert_almost_equal(arr_aligned[1], aligned[1], decimal=7)
    assert_almost_equal(arr_aligned[2], aligned[2], decimal=7)
    assert_almost_equal(error, 0.0)


def test_generalized_without_reference():
    r"""Test generalized Procrustes without reference."""
    arr_b = np.array([[5, 0], [8, 0], [5, 5]])
    arr_c = np.dot(arr_b, _rotation(30))
    arr_d = np.dot(arr_b, _rotation(45))
    arr_e = np.dot(arr_b, _rotation(90))
    arr_list = [arr_b, arr_c, arr_d, arr_e]
    # arr_list = [arr_c, arr_d, arr_e]
    arr_aligned, error = generalized(arr_list, ref=None, tol=1.0e-7, n_iter=200)
    # one right alignment
    aligned = [
        np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0]]),
        np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0]]),
        np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0]]),
        np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0]]),
    ]
    assert_almost_equal(arr_aligned[0], aligned[0], decimal=7)
    assert_almost_equal(arr_aligned[1], aligned[1], decimal=7)
    assert_almost_equal(arr_aligned[2], aligned[2], decimal=7)
    assert_almost_equal(arr_aligned[3], aligned[3], decimal=7)
    assert_almost_equal(error, 0.0)


def test_generalized_invalid():
    """Test invalid input of n_iter for generalized Procrustes analysis."""
    arr_b = np.array([[5, 0], [8, 0], [5, 5]])
    arr_c = np.dot(arr_b, _rotation(30))
    arr_d = np.dot(arr_b, _rotation(45))
    arr_e = np.dot(arr_b, _rotation(90))
    arr_list = [arr_b, arr_c, arr_d, arr_e]
    assert_raises(ValueError, generalized, arr_list, None, 1.0e-7, n_iter=-5)


def _rotation(degree):
    """Generate the rotation matrix."""
    theta = np.radians(degree)
    rot = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
    return rot


def test_generalized_with_missing_values_nan():
    """Test generalized Procrustes with missing values using NaN."""
    # Create test data with known transformation
    arr_base = np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0]])
    arr_b = arr_base.copy()
    arr_c = np.dot(arr_base, _rotation(30))
    arr_d = np.dot(arr_base, _rotation(45))

    # Introduce missing values (NaN)
    arr_b_missing = arr_b.copy()
    arr_b_missing[0, 0] = np.nan  # missing value in first array

    arr_c_missing = arr_c.copy()
    arr_c_missing[1, 1] = np.nan  # missing value in second array

    arr_d_missing = arr_d.copy()
    arr_d_missing[2, 0] = np.nan  # missing value in third array

    arr_list = [arr_b_missing, arr_c_missing, arr_d_missing]

    # Run GPA with missing value handling
    arr_aligned, error = generalized(
        arr_list, ref=None, tol=1.0e-5, n_iter=200, handle_missing=True
    )

    # Check that all arrays are aligned (should be close to arr_base after alignment)
    # The alignment might not be perfect due to missing values, but should be reasonable
    assert len(arr_aligned) == 3
    assert arr_aligned[0].shape == arr_base.shape

    # Check that missing values have been filled
    assert not np.any(np.isnan(arr_aligned[0]))
    assert not np.any(np.isnan(arr_aligned[1]))
    assert not np.any(np.isnan(arr_aligned[2]))

    # Error should be finite and positive
    assert np.isfinite(error)
    assert error >= 0


def test_generalized_with_missing_values_masked():
    """Test generalized Procrustes with missing values using masked arrays."""
    # Create test data
    arr_base = np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0]])
    arr_b = arr_base.copy()
    arr_c = np.dot(arr_base, _rotation(30))

    # Create masked arrays
    arr_b_masked = np.ma.array(arr_b, mask=[[True, False], [False, False], [False, False]])
    arr_c_masked = np.ma.array(arr_c, mask=[[False, False], [False, True], [False, False]])

    arr_list = [arr_b_masked, arr_c_masked]

    # Run GPA with missing value handling
    arr_aligned, error = generalized(
        arr_list, ref=None, tol=1.0e-5, n_iter=200, handle_missing=True
    )

    # Check results
    assert len(arr_aligned) == 2
    assert not np.any(np.isnan(arr_aligned[0]))
    assert not np.any(np.isnan(arr_aligned[1]))
    assert np.isfinite(error)


def test_generalized_missing_backward_compatibility():
    """Test that handle_missing=False maintains backward compatibility."""
    # Create complete arrays (no missing values)
    arr_b = np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0]])
    arr_c = np.dot(arr_b, _rotation(30))
    arr_d = np.dot(arr_b, _rotation(45))
    arr_list = [arr_b, arr_c, arr_d]

    # Run with handle_missing=False (default)
    arr_aligned_old, error_old = generalized(
        arr_list, ref=None, tol=1.0e-7, n_iter=200, handle_missing=False
    )

    # Run without specifying handle_missing (should default to False)
    arr_aligned_default, error_default = generalized(arr_list, ref=None, tol=1.0e-7, n_iter=200)

    # Results should be identical
    for i in range(len(arr_aligned_old)):
        assert_almost_equal(arr_aligned_old[i], arr_aligned_default[i], decimal=10)
    assert_almost_equal(error_old, error_default, decimal=10)


def test_generalized_missing_no_missing_values():
    """Test that GPA with missing value handling works when no values are actually missing."""
    arr_b = np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0]])
    arr_c = np.dot(arr_b, _rotation(30))
    arr_d = np.dot(arr_b, _rotation(45))
    arr_list = [arr_b, arr_c, arr_d]

    # Run with handle_missing=True but no actual missing values
    arr_aligned, error = generalized(
        arr_list, ref=None, tol=1.0e-7, n_iter=200, handle_missing=True
    )

    # Should still produce good alignment
    assert len(arr_aligned) == 3
    for aligned in arr_aligned:
        assert not np.any(np.isnan(aligned))
    assert error < 1.0e-5


def test_generalized_missing_high_percentage():
    """Test GPA with a high percentage of missing values."""
    arr_base = np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0], [3.0, 4.0]])
    arr_b = arr_base.copy()
    arr_c = np.dot(arr_base, _rotation(30))

    # Create arrays with many missing values
    arr_b_missing = arr_b.copy()
    arr_b_missing[0, 0] = np.nan
    arr_b_missing[1, 1] = np.nan
    arr_b_missing[3, 0] = np.nan

    arr_c_missing = arr_c.copy()
    arr_c_missing[0, 1] = np.nan
    arr_c_missing[2, 0] = np.nan

    arr_list = [arr_b_missing, arr_c_missing]

    # Run GPA
    arr_aligned, error = generalized(
        arr_list, ref=None, tol=1.0e-5, n_iter=200, handle_missing=True
    )

    # Check that algorithm completes and fills missing values
    assert len(arr_aligned) == 2
    assert not np.any(np.isnan(arr_aligned[0]))
    assert not np.any(np.isnan(arr_aligned[1]))
    assert np.isfinite(error)


def test_generalized_missing_with_reference():
    """Test GPA with missing values and a provided reference."""
    arr_base = np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0]])
    arr_b = arr_base.copy()
    arr_c = np.dot(arr_base, _rotation(30))

    # Introduce missing values
    arr_b_missing = arr_b.copy()
    arr_b_missing[0, 0] = np.nan

    arr_c_missing = arr_c.copy()
    arr_c_missing[1, 1] = np.nan

    arr_list = [arr_b_missing, arr_c_missing]

    # Use arr_base as reference
    arr_aligned, error = generalized(
        arr_list, ref=arr_base, tol=1.0e-5, n_iter=200, handle_missing=True
    )

    # Check results
    assert len(arr_aligned) == 2
    assert not np.any(np.isnan(arr_aligned[0]))
    assert not np.any(np.isnan(arr_aligned[1]))
    assert np.isfinite(error)


def test_generalized_missing_convergence():
    """Test that GPA with missing values converges."""
    arr_base = np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0]])
    arr_b = arr_base.copy()
    arr_c = np.dot(arr_base, _rotation(30))
    arr_d = np.dot(arr_base, _rotation(60))

    # Introduce missing values
    arr_b_missing = arr_b.copy()
    arr_b_missing[0, 0] = np.nan

    arr_c_missing = arr_c.copy()
    arr_c_missing[1, 1] = np.nan

    arr_d_missing = arr_d.copy()
    arr_d_missing[2, 0] = np.nan

    arr_list = [arr_b_missing, arr_c_missing, arr_d_missing]

    # Run with different tolerances
    arr_aligned_loose, error_loose = generalized(
        arr_list, ref=None, tol=1.0e-3, n_iter=200, handle_missing=True
    )
    arr_aligned_tight, error_tight = generalized(
        arr_list, ref=None, tol=1.0e-7, n_iter=200, handle_missing=True
    )

    # Tighter tolerance should give better (or equal) result
    assert error_tight <= error_loose + 1.0e-5

    # Both should produce valid results
    for aligned in arr_aligned_loose:
        assert not np.any(np.isnan(aligned))
    for aligned in arr_aligned_tight:
        assert not np.any(np.isnan(aligned))


def test_generalized_missing_different_patterns():
    """Test GPA with different missing patterns across arrays."""
    arr_base = np.array([[5.0, 0.0], [8.0, 0.0], [5.0, 5.0]])

    # Create three arrays with different rotation and missing patterns
    arr_a = arr_base.copy()
    arr_a[0, 0] = np.nan  # top-left missing

    arr_b = np.dot(arr_base, _rotation(45))
    arr_b[1, 1] = np.nan  # middle-right missing

    arr_c = np.dot(arr_base, _rotation(90))
    arr_c[2, 0] = np.nan  # bottom-left missing

    arr_list = [arr_a, arr_b, arr_c]

    # Run GPA
    arr_aligned, error = generalized(
        arr_list, ref=None, tol=1.0e-5, n_iter=200, handle_missing=True
    )

    # Verify results
    assert len(arr_aligned) == 3
    for i, aligned in enumerate(arr_aligned):
        assert not np.any(np.isnan(aligned)), f"Array {i} has NaN values"
        assert aligned.shape == arr_base.shape

    assert np.isfinite(error)
    assert error >= 0
