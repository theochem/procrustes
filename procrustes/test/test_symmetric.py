# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2021 The QC-Devs Community
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
"""Testings for symmetric Procrustes module."""

import numpy as np
from numpy.testing import assert_almost_equal
from procrustes import symmetric
import pytest
from scipy.optimize import minimize
from scipy.stats import ortho_group


@pytest.mark.parametrize("m, n, add_cols, add_rows", np.random.randint(50, 100, (5, 4)))
def test_symmetric_with_unpadding(m, n, add_cols, add_rows):
    r"""Test symmetric without translation and scaling."""
    # define arbitrary array and generate random symmetric transformation with rank 1.
    array_a = np.random.uniform(-10.0, 10.0, (m, n))
    rand_array = np.random.uniform(-10., 10., (n,))
    sym_array = np.outer(rand_array.T, rand_array)
    # define array_b by transforming array_a and padding with zero
    array_b = np.dot(array_a, sym_array)
    array_b = np.concatenate((array_b, np.zeros((m, add_cols))), axis=1)
    array_b = np.concatenate((array_b, np.zeros((add_rows, n + add_cols))), axis=0)
    # compute procrustes transformation
    res = symmetric(array_a, array_b, unpad_col=True, unpad_row=True)
    # check transformation is symmetric & error is zero
    assert_almost_equal(res["t"], res["t"].T, decimal=6)
    assert_almost_equal(res["error"], 0.0, decimal=6)
    assert_almost_equal(res["new_a"].dot(res["t"]), res["new_b"], decimal=6)


@pytest.mark.parametrize("m, n", np.random.randint(50, 100, (25, 2)))
def test_symmetric_scaled_shifted_transformed(m, n):
    r"""Test symmetric with translation and scaling."""
    # define an arbitrary array_a, translation matrix & symmetric matrix
    array_a = np.random.uniform(-10.0, 10.0, (m, n))
    # Define shift to the rows of the matrices. This repeats a random matrix of size n, m times.
    shift = np.tile(np.random.uniform(-10.0, 10.0, (n,)), (m, 1))
    # Generate random array with rank with which ever is the smallest.
    rand_array = np.random.uniform(-10., 10., (m, n))
    sym_array = np.dot(rand_array.T, rand_array)
    # define array_b by scaling, translating and transforming array_a
    array_b = 614.5 * array_a + shift
    array_b = np.dot(array_b, sym_array)
    # compute procrustes transformation
    res = symmetric(array_a, array_b, translate=True, scale=True)
    # check transformation is symmetric & error is zero
    assert_almost_equal(res["t"], res["t"].T, decimal=6)
    assert_almost_equal(res["new_a"].dot(res["t"]), res["new_b"], decimal=6)
    assert_almost_equal(res["error"], 0.0, decimal=6)


@pytest.mark.parametrize("m, n", np.random.randint(5, 10, (25, 2)))
def test_symmetric_with_small_values(m, n):
    r"""Test symmetric by arrays with small values."""
    if n <= m:
        n = m
    # define an arbitrary array_a, translation matrix & symmetric matrix
    array_a = np.random.uniform(0.0, 1., (m, n))
    shift = np.random.uniform(0.0, 1., (m, n))
    # Random positive Semidefinite symmetric matrix.
    rand_array = np.random.uniform(-1., 1., (n, n))
    sym_array = (rand_array + rand_array.T) / 2.0
    # define array_b by scaling, translating and transforming array_a
    array_b = 6.61e-4 * array_a + shift
    array_b = np.dot(array_b, sym_array)
    # compute procrustes transformation
    res = symmetric(array_a, array_b, translate=True, scale=True)
    # check transformation is symmetric & error is zero
    assert_almost_equal(res["t"], res["t"].T, decimal=6)
    assert_almost_equal(res["error"], 0.0, decimal=5)


def test_not_full_rank_case():
    r"""Test symmetric with not full rank case."""
    # Define a random matrix and symmetric matrix
    array_a = np.array([[10, 83], [52, 58], [58, 44]])
    sym_array = np.array([[0.38895636, 0.30523869], [0.30523869, 0.30856369]])
    array_b = np.dot(array_a, sym_array)
    # compute procrustes transformation
    res = symmetric(array_a, array_b)
    # check transformation is symmetric & error is zero
    assert_almost_equal(res["t"], res["t"].T, decimal=6)
    assert_almost_equal(res["error"], 0.0, decimal=6)


def test_having_zero_singular_case():
    r"""Test symmetric that has zero singular values."""
    # Define a matrix with not full singular values, e.g. 5 and 0 are singular values.
    sing_mat = np.array([[5., 0.], [0., 0.], [0., 0.], [0., 0.]])
    array_a = ortho_group.rvs(4).dot(sing_mat).dot(ortho_group.rvs(2))

    sym_array = np.array([[0.38895636, 0.30523869], [0.30523869, 0.30856369]])
    array_b = np.dot(array_a, sym_array)
    # compute procrustes transformation
    res = symmetric(array_a, array_b)
    # check transformation is symmetric & error is zero
    assert_almost_equal(res["t"], res["t"].T, decimal=6)
    assert_almost_equal(res["error"], 0.0, decimal=6)


# def test_fat_rectangular_matrices_raises_error_no_padding():
#     r"""Test Symmetric Procrustes raises error when improper padding."""
#     # Generate Random Rectangular Matrices
#     nrow = 3
#     ncol = np.random.randint(nrow + 1, nrow + 4)
#     array_a, array_b = np.random.random((nrow, ncol)), np.random.random((nrow, ncol))
#     np.testing.assert_raises(ValueError, symmetric, array_a, array_b)
#
#     array_a = np.random.random((nrow, nrow))
#     np.testing.assert_raises(ValueError, symmetric, array_a, array_b)


class TestAgainstNumerical:
    r"""
    Testing Procrustes over symmetric matrices against numerical optimization methods.

    Note that there is a unique solution to ||AX - B|| if and only if rank(A) = n. This must be
    guaranteed for numerical optimization to be exact.

    """

    @staticmethod
    def _vector_to_matrix(vec, nsize):
        r"""Given a vector, change it to a matrix."""
        mat = np.zeros((nsize, nsize))
        mat[np.triu_indices(nsize)] = vec
        mat = mat + mat.T - np.diag(np.diag(mat))
        return mat

    def _objective_func(self, vec, array_a, array_b, nsize):
        mat = self._vector_to_matrix(vec, nsize)
        diff = array_a.dot(mat) - array_b
        return np.trace(diff.T.dot(diff))

    def _optimize(self, array_a, array_b, ncol):
        init = np.random.random(int(ncol * (ncol + 1) / 2.))
        results = minimize(self._objective_func, init, args=(array_a, array_b, ncol),
                           method="slsqp", options={"eps": 1e-8, 'ftol': 1e-11, "maxiter": 1000})
        return self._vector_to_matrix(results["x"], ncol), results["fun"]

    @pytest.mark.parametrize("ncol", [2, 10, 15])
    def test_random_tall_rectangular_matrices(self, ncol):
        r"""Test Symmetric Procrustes with random tall matrices."""
        # Generate Random Rectangular Matrices with small singular values
        nrow = np.random.randint(ncol, ncol + 10)
        array_a, array_b = np.random.random((nrow, ncol)), np.random.random((nrow, ncol))

        desired, desired_func = self._optimize(array_a, array_b, ncol)
        res = symmetric(array_a, array_b, unpad_col=True, unpad_row=True)

        assert np.abs(res["error"] - desired_func) < 1e-5
        assert np.all(np.abs(res["t"] - desired) < 1e-3)

    @pytest.mark.parametrize("nrow", [2, 10, 15])
    def test_fat_rectangular_matrices_with_square_padding(self, nrow):
        r"""Test Symmetric Procrustes with random wide matrices."""
        # Square padding is needed or else it returns an error.
        # Generate Random Rectangular Matrices
        ncol = np.random.randint(nrow + 1, nrow + 10)
        array_a, array_b = np.random.random((nrow, ncol)), np.random.random((nrow, ncol))

        _, desired_func = self._optimize(array_a, array_b, ncol)
        res = symmetric(array_a, array_b, pad=True)

        # No uniqueness in solution, thus check that the optimal values are the same.
        assert np.abs(res["error"] - desired_func) < 1e-5
