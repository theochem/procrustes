# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2022 The QC-Devs Community
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
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from scipy.stats import ortho_group

from procrustes import symmetric
from procrustes.test.common import minimize_one_transformation


@pytest.mark.parametrize("m, n, add_cols, add_rows", np.random.randint(50, 100, (5, 4)))
def test_symmetric_with_unpadding(m, n, add_cols, add_rows):
    r"""Test symmetric without translation and scaling."""
    # define arbitrary array and generate random symmetric transformation with rank 1.
    array_a = np.random.uniform(-10.0, 10.0, (m, n))
    rand_array = np.random.uniform(-10.0, 10.0, (n,))
    sym_array = np.outer(rand_array.T, rand_array)
    # define array_b by transforming array_a and padding with zero
    array_b = np.dot(array_a, sym_array)
    array_b = np.concatenate((array_b, np.zeros((m, add_cols))), axis=1)
    array_b = np.concatenate((array_b, np.zeros((add_rows, n + add_cols))), axis=0)
    # compute procrustes transformation
    res = symmetric(array_a, array_b, unpad_col=True, unpad_row=True)
    # check transformation is symmetric & error is zero
    assert_almost_equal(res.t, res.t.T, decimal=6)
    assert_almost_equal(res.error, 0.0, decimal=6)
    assert_equal(res.s, None)
    assert_almost_equal(res.new_a.dot(res.t), res.new_b, decimal=6)


@pytest.mark.parametrize("m, n", np.random.randint(50, 100, (25, 2)))
def test_symmetric_scaled_shifted_transformed(m, n):
    r"""Test symmetric with translation and scaling."""
    # define an arbitrary array_a, translation matrix & symmetric matrix
    array_a = np.random.uniform(-10.0, 10.0, (m, n))
    # Define shift to the rows of the matrices. This repeats a random matrix of size n, m times.
    shift = np.tile(np.random.uniform(-10.0, 10.0, (n,)), (m, 1))
    # Generate random array with rank with which ever is the smallest.
    rand_array = np.random.uniform(-10.0, 10.0, (m, n))
    sym_array = np.dot(rand_array.T, rand_array)
    # define array_b by scaling, translating and transforming array_a
    array_b = 614.5 * array_a + shift
    array_b = np.dot(array_b, sym_array)
    # compute procrustes transformation
    res = symmetric(array_a, array_b, translate=True, scale=True)
    # check transformation is symmetric & error is zero
    assert_almost_equal(res.t, res.t.T, decimal=6)
    assert_almost_equal(res.new_a.dot(res.t), res.new_b, decimal=6)
    assert_equal(res.s, None)
    assert_almost_equal(res.error, 0.0, decimal=6)


@pytest.mark.parametrize("m, n", np.random.randint(50, 100, (25, 2)))
def test_symmetric_with_small_values(m, n):
    r"""Test symmetric by arrays with small values."""
    # define an arbitrary array_a, translation matrix & symmetric matrix
    array_a = np.random.uniform(0.0, 1.0e-6, (m, n))
    # Define shift to the rows of the matrices. This repeats a random matrix of size n, m times.
    shift = np.tile(np.random.uniform(-1.0e-6, 1.0e-6, (n,)), (m, 1))
    # Random symmetric matrix.
    rand_array = np.random.uniform(-1.0, 1.0, (n, n))
    sym_array = (rand_array + rand_array.T) / 2.0
    # define array_b by scaling, translating and transforming array_a
    array_b = 6.61e-4 * array_a + shift
    array_b = np.dot(array_b, sym_array)
    # compute procrustes transformation
    res = symmetric(array_a, array_b, translate=True, scale=True)
    # check transformation is symmetric & error is zero
    assert_equal(res.s, None)
    assert_almost_equal(res.t, res.t.T, decimal=6)
    assert_almost_equal(res.error, 0.0, decimal=5)


def test_not_full_rank_case():
    r"""Test symmetric with not full rank case."""
    # define a random matrix and symmetric matrix
    array_a = np.array([[10, 83], [52, 58], [58, 44]])
    sym_array = np.array([[0.38895636, 0.30523869], [0.30523869, 0.30856369]])
    array_b = np.dot(array_a, sym_array)
    # compute procrustes transformation & check results
    res = symmetric(array_a, array_b)
    assert_equal(res.s, None)
    assert_almost_equal(res.t, res.t.T, decimal=6)
    assert_almost_equal(res.error, 0.0, decimal=6)


@pytest.mark.parametrize("m", np.random.randint(50, 100, (25,)))
# @pytest.mark.parametrize("m, n", np.random.randint(50, 100, (25, 2)))
def test_having_zero_eigenvalues_case(m):
    r"""Test symmetric that has zero singular values."""
    # define a singular matrix (i.e. some eigenvalues are hard zeros)
    numb_nonzero_eigs = int(np.random.randint(1, m - 1))
    sing_mat = list(np.random.uniform(-10.0, 10.0, (numb_nonzero_eigs,)))
    sing_mat += [0.0] * (m - numb_nonzero_eigs)
    sing_mat = np.diag(sing_mat)
    array_a = ortho_group.rvs(m).dot(sing_mat).dot(ortho_group.rvs(m))
    # generate random symmetric matrix & define matrix b
    rand_array = np.random.uniform(-1.0, 1.0, (m, m))
    sym_array = (rand_array + rand_array.T) / 2.0
    array_b = np.dot(array_a, sym_array)
    # compute procrustes transformation & check results
    res = symmetric(array_a, array_b)
    assert_equal(res.s, None)
    assert_almost_equal(res.t, res.t.T, decimal=6)
    assert_almost_equal(res.error, 0.0, decimal=6)


@pytest.mark.parametrize("ncol", np.random.randint(2, 15, (3,)))
def test_random_tall_rectangular_matrices(ncol):
    r"""Test Symmetric Procrustes with random tall matrices."""
    # generate random floats in [0.0, 1.0) interval
    nrow = np.random.randint(ncol, ncol + 10)
    array_a, array_b = np.random.random((nrow, ncol)), np.random.random((nrow, ncol))
    # minimize objective function to find transformation matrix
    desired, desired_func = minimize_one_transformation(array_a, array_b, ncol)
    # compute transformation & check results
    res = symmetric(array_a, array_b, unpad_col=True, unpad_row=True)
    assert_equal(res.s, None)
    assert_almost_equal(np.abs(res.error - desired_func), 0.0, decimal=5)
    assert_almost_equal(np.abs(res.t - desired), 0.0, decimal=3)


@pytest.mark.parametrize("nrow", np.random.randint(2, 15, (3,)))
def test_fat_rectangular_matrices_with_square_padding(nrow):
    r"""Test Symmetric Procrustes with random wide matrices."""
    # generate random rectangular matrices
    ncol = np.random.randint(nrow + 1, nrow + 10)
    array_a, array_b = np.random.random((nrow, ncol)), np.random.random((nrow, ncol))
    # minimize objective function to find transformation matrix
    _, desired_func = minimize_one_transformation(array_a, array_b, ncol)
    res = symmetric(array_a, array_b, pad=True)
    # check results (solution is not uniqueness)
    assert_almost_equal(np.abs(res.error - desired_func), 0.0, decimal=5)
    assert_equal(res.s, None)


@pytest.mark.parametrize("nrow", np.random.randint(2, 15, (3,)))
def test_fat_rectangular_matrices_with_square_padding_with_lapack_driver(nrow):
    r"""Test Symmetric Procrustes with random wide matrices and non-default lapack driver."""
    # generate random rectangular matrices
    ncol = np.random.randint(nrow + 1, nrow + 10)
    array_a, array_b = np.random.random((nrow, ncol)), np.random.random((nrow, ncol))
    # minimize objective function to find transformation matrix
    _, desired_func = minimize_one_transformation(array_a, array_b, ncol)
    res = symmetric(array_a, array_b, pad=True, lapack_driver="gesdd")
    # check results (solution is not uniqueness)
    assert_almost_equal(np.abs(res.error - desired_func), 0.0, decimal=5)
    assert_equal(res.s, None)
