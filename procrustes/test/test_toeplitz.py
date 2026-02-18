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
r"""Tests for Toeplitz and Hankel Procrustes module."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.linalg import hankel as scipy_hankel
from scipy.linalg import toeplitz as scipy_toeplitz

from procrustes.generic import generic
from procrustes.toeplitz import hankel, toeplitz

# Helper utilities


def _is_toeplitz(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """Return True if *matrix* is Toeplitz (constant along each diagonal)."""
    n = matrix.shape[0]
    for k in range(-(n - 1), n):
        diag = np.diag(matrix, k)
        if not np.allclose(diag, diag[0], atol=tol):
            return False
    return True


def _is_hankel(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """Return True if *matrix* is Hankel (constant along each anti-diagonal)."""
    n = matrix.shape[0]
    for k in range(2 * n - 1):
        # collect elements where i + j == k
        vals = [matrix[i, k - i] for i in range(n) if 0 <= k - i < n]
        if not np.allclose(vals, vals[0], atol=tol):
            return False
    return True

# Toeplitz tests


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_toeplitz_exact_recovery(n):
    r"""When B = A @ T_true with T_true Toeplitz, the solver should recover T_true exactly."""
    rng = np.random.default_rng(42 + n)
    mat_a = rng.uniform(-5.0, 5.0, (n, n))
    col = rng.uniform(-3.0, 3.0, n)
    row = rng.uniform(-3.0, 3.0, n)
    row[0] = col[0]  # Toeplitz constraint: T[0,0] shared
    t_true = scipy_toeplitz(col, row)
    mat_b = mat_a @ t_true

    result = toeplitz(mat_a, mat_b)

    assert_almost_equal(result.error, 0.0, decimal=6)
    assert_almost_equal(result.t, t_true, decimal=6)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_toeplitz_output_is_toeplitz(n):
    r"""The transformation matrix returned by toeplitz() must be a Toeplitz matrix."""
    rng = np.random.default_rng(100 + n)
    mat_a = rng.uniform(-5.0, 5.0, (n, n))
    mat_b = rng.uniform(-5.0, 5.0, (n, n))

    result = toeplitz(mat_a, mat_b)

    assert _is_toeplitz(result.t), "Returned transformation matrix is not Toeplitz."


@pytest.mark.parametrize("n", [3, 4, 5])
def test_toeplitz_error_leq_generic(n):
    r"""Toeplitz error should be >= generic (unconstrained) error for the same inputs."""
    rng = np.random.default_rng(200 + n)
    mat_a = rng.uniform(-5.0, 5.0, (n, n))
    mat_b = rng.uniform(-5.0, 5.0, (n, n))

    res_toeplitz = toeplitz(mat_a, mat_b)
    res_generic = generic(mat_a, mat_b)

    # Constrained optimum can only be worse than or equal to unconstrained
    assert res_toeplitz.error >= res_generic.error - 1e-8


def test_toeplitz_raises_shape_mismatch():
    r"""toeplitz() should raise ValueError when shapes cannot be reconciled."""
    # A is (3,3), B is (3,5) -> after pad=False they differ in columns -> ValueError
    mat_a = np.random.rand(3, 3)
    mat_b = np.random.rand(3, 5)
    with pytest.raises(ValueError):
        toeplitz(mat_a, mat_b, pad=False)

# Hankel tests


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_hankel_exact_recovery(n):
    r"""When B = A @ H_true with H_true Hankel, the solver should recover H_true exactly."""
    rng = np.random.default_rng(42 + n)
    mat_a = rng.uniform(-5.0, 5.0, (n, n))
    col = rng.uniform(-3.0, 3.0, n)
    row = rng.uniform(-3.0, 3.0, n)
    row[0] = col[-1]  # Hankel constraint: H[n-1,0] shared
    h_true = scipy_hankel(col, row)
    mat_b = mat_a @ h_true

    result = hankel(mat_a, mat_b)

    assert_almost_equal(result.error, 0.0, decimal=6)
    assert_almost_equal(result.t, h_true, decimal=6)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_hankel_output_is_hankel(n):
    r"""The transformation matrix returned by hankel() must be a Hankel matrix."""
    rng = np.random.default_rng(400 + n)
    mat_a = rng.uniform(-5.0, 5.0, (n, n))
    mat_b = rng.uniform(-5.0, 5.0, (n, n))

    result = hankel(mat_a, mat_b)

    assert _is_hankel(result.t), "Returned transformation matrix is not Hankel."


@pytest.mark.parametrize("n", [3, 4, 5])
def test_hankel_error_leq_generic(n):
    r"""Hankel error should be >= generic (unconstrained) error for the same inputs."""
    rng = np.random.default_rng(500 + n)
    mat_a = rng.uniform(-5.0, 5.0, (n, n))
    mat_b = rng.uniform(-5.0, 5.0, (n, n))

    res_hankel = hankel(mat_a, mat_b)
    res_generic = generic(mat_a, mat_b)

    assert res_hankel.error >= res_generic.error - 1e-8


def test_hankel_raises_shape_mismatch():
    r"""hankel() should raise ValueError when shapes cannot be reconciled."""
    mat_a = np.random.rand(3, 3)
    mat_b = np.random.rand(3, 5)
    with pytest.raises(ValueError):
        hankel(mat_a, mat_b, pad=False)

# Rectangular A matrix tests


@pytest.mark.parametrize("m, n", [(5, 3), (6, 4), (8, 3)])
def test_toeplitz_rectangular_output_is_toeplitz(m, n):
    r"""Toeplitz() must return a Toeplitz T when A is rectangular (m > n)."""
    rng = np.random.default_rng(300 + m + n)
    mat_a = rng.uniform(-5.0, 5.0, (m, n))
    mat_b = rng.uniform(-5.0, 5.0, (m, n))

    result = toeplitz(mat_a, mat_b)

    assert result.t.shape == (n, n), f"Expected T shape ({n}, {n}), got {result.t.shape}"
    assert _is_toeplitz(result.t), "Returned transformation matrix is not Toeplitz."


@pytest.mark.parametrize("m, n", [(5, 3), (6, 4), (8, 3)])
def test_hankel_rectangular_output_is_hankel(m, n):
    r"""Hankel() must return a Hankel T when A is rectangular (m > n)."""
    rng = np.random.default_rng(600 + m + n)
    mat_a = rng.uniform(-5.0, 5.0, (m, n))
    mat_b = rng.uniform(-5.0, 5.0, (m, n))

    result = hankel(mat_a, mat_b)

    assert result.t.shape == (n, n), f"Expected T shape ({n}, {n}), got {result.t.shape}"
    assert _is_hankel(result.t), "Returned transformation matrix is not Hankel."


# ProcrustesResult structure tests


def test_toeplitz_result_fields():
    r"""Ensure that the result from toeplitz() has the expected fields."""
    mat_a = np.random.rand(4, 4)
    mat_b = np.random.rand(4, 4)
    result = toeplitz(mat_a, mat_b)
    assert hasattr(result, "error")
    assert hasattr(result, "new_a")
    assert hasattr(result, "new_b")
    assert hasattr(result, "t")
    assert result.s is None


def test_hankel_result_fields():
    r"""Ensure that the result from hankel() has the expected fields."""
    mat_a = np.random.rand(4, 4)
    mat_b = np.random.rand(4, 4)
    result = hankel(mat_a, mat_b)
    assert hasattr(result, "error")
    assert hasattr(result, "new_a")
    assert hasattr(result, "new_b")
    assert hasattr(result, "t")
    assert result.s is None
