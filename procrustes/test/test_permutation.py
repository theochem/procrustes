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
"""Testings for permutation module."""
# pylint: disable=too-many-lines


import itertools

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_raises

from procrustes.permutation import (
    _approx_permutation_2sided_1trans_normal1,
    _approx_permutation_2sided_1trans_normal2,
    _approx_permutation_2sided_1trans_umeyama,
    permutation,
    permutation_2sided,
)


def generate_random_permutation_matrix(n):
    r"""Generate a random permutation matrix."""
    arr = np.arange(0, n)
    np.random.shuffle(arr)
    perm = np.zeros((n, n))
    perm[np.arange(0, n), arr] = 1.0
    return perm


@pytest.mark.parametrize("n", np.random.randint(50, 100, (25)))
def test_permutation_one_sided_square_matrices_rows_permuted(n):
    r"""Test one-sided permutation Procrustes with square matrices and permuted rows."""
    array_a = np.random.uniform(-10.0, 10.0, (n, n))
    perm = generate_random_permutation_matrix(n)
    # permuted array_b
    array_b = np.dot(array_a, perm)
    # procrustes with no translate and scale
    res = permutation(array_a, array_b)
    assert_almost_equal(res.t, perm, decimal=6)
    assert_almost_equal(res.error, 0.0, decimal=6)


@pytest.mark.parametrize("m, n, ncols, nrows", np.random.randint(50, 100, (25, 4)))
def test_permutation_one_sided_columns_pad(m, n, ncols, nrows):
    r"""Test one-sided permutation by permuted columns along with padded zeros."""
    array_a = np.random.uniform(-10.0, 10.0, (m, n))
    perm = generate_random_permutation_matrix(n)
    # permuted array_b
    array_b = np.dot(array_a, perm)
    # padded array b with zero row and columns
    array_b = np.concatenate((array_b, np.zeros((m, ncols))), axis=1)
    array_b = np.concatenate((array_b, np.zeros((nrows, n + ncols))), axis=0)
    if m < n:
        array_a = np.concatenate((array_a, np.zeros((n - m, n))), axis=0)
    # procrustes with no translate and scale
    res = permutation(array_a, array_b, unpad_col=True, unpad_row=True)
    # Test that the unpadded b is the same as the original b.
    assert_almost_equal(res.new_b, np.dot(array_a, perm), decimal=6)
    # Test that the permutation and the error are the same/zero.
    assert_almost_equal(res.t, perm, decimal=6)
    assert_almost_equal(res.error, 0.0, decimal=6)


@pytest.mark.parametrize("m, n", np.random.randint(50, 100, (25, 2)))
def test_permutation_one_sided_with_translate_scale(m, n):
    r"""Test permutation one_sided by translated and scaled arrays."""
    array_a = np.random.uniform(-10.0, 10.0, (m, n))
    # array_b is scaled, translated, and permuted array_a
    perm = generate_random_permutation_matrix(n)
    # obtain random translation/shift array and permute the array.
    shift = np.random.uniform(-10.0, 10.0, (n,))
    array_b = 3.78 * array_a + shift
    array_b = np.dot(array_b, perm)
    # permutation procrustes
    res = permutation(array_a, array_b, translate=True, scale=True)
    assert_almost_equal(res.t, perm, decimal=6)
    assert_almost_equal(res.error, 0.0, decimal=6)


def test_2sided_1trans_initial_guess_normal1_positive():
    r"""Test 2sided-perm initial normal1 guess by positive arrays."""
    # define a random array
    a = np.array([[1, 5, 8, 4], [0, 12, 7, 2], [3, 6, 9, 4], [2, 7, 8, 5]])
    # Build the new matrix array_b
    array_b = np.array([[1, 12, 9, 5], [8, 7, 6, 8], [5, 2, 4, 7], [4, 0, 3, 2]])
    weight_p = np.power(2, -0.5)
    weight = np.empty(a.shape)
    for row in range(4):
        weight[row, :] = np.power(weight_p, row)
    array_b = np.multiply(array_b, weight)
    # Check
    array_new = _approx_permutation_2sided_1trans_normal1(a)
    assert_almost_equal(array_b, array_new, decimal=6)


def test_2sided_1trans_initial_guess_normal1_negative():
    r"""Test 2sided-perm initial normal1 guess by negative arrays."""
    # Define a random array
    array_a = np.array([[1, 5, -8, 4], [0, 12, 7, 2], [3, -6, 9, 4], [2, -7, 8, -5]])
    # Build the new matrix array_b
    array_b = np.array([[1, 12, 9, -5], [-8, 7, -6, 8], [5, 2, 4, -7], [4, 0, 3, 2]])
    weight_p = np.power(2, -0.5)
    weight = np.empty(array_a.shape)
    for row in range(4):
        weight[row, :] = np.power(weight_p, row)
    array_b = np.multiply(array_b, weight)
    # Check
    array_new = _approx_permutation_2sided_1trans_normal1(array_a)
    assert_almost_equal(array_b, array_new, decimal=6)


def test_2sided_1trans_initial_guess_normal2_positive():
    r"""Test 2sided-perm initial normal2 guess by positive arrays."""
    # Define a random array
    array_a = np.array(
        [
            [32, 14, 3, 63, 50],
            [24, 22, 1, 56, 4],
            [94, 16, 28, 75, 81],
            [19, 72, 42, 90, 54],
            [71, 85, 10, 96, 58],
        ]
    )
    array_b = np.array(
        [
            [32, 22, 28, 90, 58],
            [90, 90, 32, 22, 90],
            [63, 56, 94, 72, 96],
            [58, 32, 58, 58, 22],
            [50, 24, 81, 54, 85],
            [22, 58, 90, 28, 32],
            [14, 4, 75, 42, 71],
            [28, 28, 22, 32, 28],
            [3, 1, 16, 19, 10],
        ]
    )
    # Build the new matrix array_b
    weight_p = np.power(2, -0.5)
    weight = np.zeros([9, 5])
    weight[0, :] = 1
    for col in range(1, array_a.shape[1]):
        weight[2 * col - 1, :] = np.power(weight_p, col)
        weight[2 * col, :] = np.power(weight_p, col)
    array_b = np.multiply(array_b, weight)
    # Check
    array_new = _approx_permutation_2sided_1trans_normal2(array_a)
    assert_almost_equal(array_b, array_new, decimal=6)


def test_2sided_1trans_initial_guess_normal2_negative():
    r"""Test 2sided-perm initial normal2 guess by negative arrays."""
    # Define a random matrix array_a
    array_a = np.array([[3, -1, 4, -1], [-1, 5, 7, 6], [4, 7, -9, 3], [-1, 6, 3, 2]])
    array_b = np.array(
        [
            [3, 5, -9, 2],
            [-9, -9, 5, 5],
            [4, 7, 7, 6],
            [5, 2, 3, -9],
            [-1, 6, 4, 3],
            [2, 3, 2, 3],
            [-1, -1, 3, -1],
        ]
    )
    # Build the new matrix array_b
    weight_p = np.power(2, -0.5)
    weight = np.zeros([7, 4])
    weight[0, :] = 1
    for col in range(1, array_a.shape[1]):
        weight[2 * col - 1, :] = np.power(weight_p, col)
        weight[2 * col, :] = np.power(weight_p, col)
    array_b = np.multiply(array_b, weight)
    # Check
    array_new = _approx_permutation_2sided_1trans_normal2(array_a)
    assert_almost_equal(array_b, array_new, decimal=6)


def test_2sided_1trans_initial_guess_umeyama():
    r"""Test 2sided-perm initial umeyama guess by positive arrays."""
    a = np.array([[0, 5, 8, 6], [5, 0, 5, 1], [8, 5, 0, 2], [6, 1, 2, 0]])
    b = np.array([[0, 1, 8, 4], [1, 0, 5, 2], [8, 5, 0, 5], [4, 2, 5, 0]])
    u_umeyama = np.array(
        [
            [0.909, 0.818, 0.973, 0.893],
            [0.585, 0.653, 0.612, 0.950],
            [0.991, 0.524, 0.892, 0.601],
            [0.520, 0.931, 0.846, 0.618],
        ]
    )
    array_u = _approx_permutation_2sided_1trans_umeyama(a=b, b=a)
    assert_almost_equal(u_umeyama, array_u, decimal=3)


@pytest.mark.parametrize("n", np.random.randint(50, 100, (10,)))
def test_permutation_2sided_1trans_umeyama(n):
    r"""Test 2sided-permutation with single transform with umeyama guess."""
    # define a random matrix
    array_a = np.random.uniform(-10.0, 10.0, (n, n))
    array_a = (array_a + array_a.T) / 2.0
    # define array_b by permuting array_a
    perm = generate_random_permutation_matrix(n)
    array_b = np.dot(perm.T, np.dot(array_a, perm))
    # Check
    res = permutation_2sided(array_a, array_b, single=True, method="approx-umeyama")
    res = permutation_2sided(res.new_a, res.new_b, single=True, method="nmf", guess_p2=res.t)
    assert_almost_equal(res.t, perm, decimal=6)
    assert_almost_equal(res.error, 0, decimal=6)


@pytest.mark.parametrize("n", np.random.randint(3, 6, (5,)))
def test_permutation_2sided_1trans_small_matrices_umeyama_all_permutations(n):
    r"""Test 2sided-perm single transform with Umeyama guess for all permutations."""
    # define a random matrix
    a = np.random.uniform(-10.0, 10.0, (n, n))
    # check with all possible permutation matrices
    for comb in itertools.permutations(np.arange(n)):
        p = np.zeros((n, n))
        p[np.arange(n), comb] = 1
        # get array_b by permutation
        b = np.dot(p.T, np.dot(a, p))
        res = permutation_2sided(a, b, single=True, method="approx-umeyama")
        res = permutation_2sided(res.new_a, res.new_b, single=True, method="nmf", guess_p2=res.t)
        assert_almost_equal(res.t, p, decimal=6)
        assert_almost_equal(res.error, 0, decimal=6)


@pytest.mark.parametrize("n", np.random.randint(50, 500, (10,)))
def test_permutation_2sided_1trans_symmetric_umeyama_translate_scale(n):
    r"""Test 2sided-perm with Umeyama guess with symmetric arrays with translation and scale."""
    # define a random, symmetric matrix
    a = np.random.uniform(-10, 10.0, (n, n))
    a = np.dot(a, a.T)
    # define array_b by scale-translate array_a and permuting
    shift = np.random.uniform(-10.0, 10.0, n)
    perm = generate_random_permutation_matrix(n)
    b = np.dot(perm.T, np.dot((14.7 * a + shift), perm))
    res = permutation_2sided(a, b, single=True, method="approx-umeyama", translate=True, scale=True)
    assert_almost_equal(res.t, perm, decimal=6)
    assert_almost_equal(res.s, perm.T, decimal=6)
    assert_almost_equal(res.error, 0, decimal=6)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_permutation_2sided_single_transform_umeyama_translate_scale_all_permutations(n):
    r"""Test 2-sided single transform permutation Umeyama guess for all permutations."""
    # define a random matrix
    array_a = np.random.uniform(-10.0, 10.0, (n, n))
    # check with all possible permutation matrices
    for comb in itertools.permutations(np.arange(n)):
        # Compute the permutation matrix
        perm = np.zeros((n, n))
        perm[np.arange(n), comb] = 1
        # Compute the translated, scaled matrix
        shift = np.random.uniform(-10.0, 10.0, n)
        array_b = np.dot(perm.T, np.dot(60 * array_a + shift, perm))

        res = permutation_2sided(
            array_a, array_b, single=True, method="approx-umeyama", translate=True, scale=True
        )
        assert_almost_equal(res.t, perm, decimal=6)
        assert_almost_equal(res.s, perm.T, decimal=6)
        assert_almost_equal(res.error, 0, decimal=6)


@pytest.mark.parametrize("n, ncol, nrow", np.random.randint(50, 100, (10, 3)))
def test_permutation_2sided_single_transform_umeyama_translate_scale_zero_padding(n, ncol, nrow):
    r"""Test permutation two-sided umeyama guess with translation, scale and padding."""
    a = np.random.uniform(-10.0, 10.0, (n, n))
    p = generate_random_permutation_matrix(n)
    b = np.dot(p.T, np.dot(20 * a + 8, p))
    # pad both of the matrices with zeros
    a = np.concatenate((a, np.zeros((n, ncol))), axis=1)
    a = np.concatenate((a, np.zeros((nrow, n + ncol))), axis=0)
    b = np.concatenate((b, np.zeros((n, ncol))), axis=1)
    b = np.concatenate((b, np.zeros((nrow, n + ncol))), axis=0)
    res = permutation_2sided(
        a,
        b,
        single=True,
        method="approx-umeyama",
        unpad_col=True,
        unpad_row=True,
        translate=True,
        scale=True,
    )
    assert_almost_equal(res.t, p, decimal=6)
    assert_almost_equal(res.error, 0, decimal=6)


@pytest.mark.parametrize("n", np.random.randint(50, 100, (10,)))
def test_permutation_2sided_1trans_umeyama_approx(n):
    r"""Test 2sided-perm, single transform with "umeyama_approx" mode."""
    # define a random, symmetric matrix
    array_a = np.random.uniform(-10.0, 10.0, (n, n))
    array_a = (array_a + array_a.T) / 2.0
    # define array_b by permuting array_a
    perm = generate_random_permutation_matrix(n)
    array_b = np.dot(perm.T, np.dot(array_a, perm))
    res = permutation_2sided(array_a, array_b, single=True, method="approx-umeyama-svd")
    assert_almost_equal(res.t, perm, decimal=6)
    assert_almost_equal(res.error, 0, decimal=6)


def test_permutation_2sided_4by4_umeyama_approx_loop():
    r"""Test 2sided-perm with "umeyama_approx" mode by 4by4 arrays for all permutations."""
    # define a random matrix
    array_a = np.array([[4, 5, 3, 3], [5, 7, 3, 5], [3, 3, 2, 2], [3, 5, 2, 5]])
    array_a = (array_a + array_a.T) / 2.0
    # check with all possible permutation matrices
    for comb in itertools.permutations(np.arange(4)):
        perm = np.zeros((4, 4))
        perm[np.arange(4), comb] = 1
        # get array_b by permutation
        array_b = np.dot(perm.T, np.dot(array_a, perm))
        # Check
        res = permutation_2sided(array_a, array_b, single=True, method="approx-umeyama")
        assert_almost_equal(res.t, perm, decimal=6)
        assert_almost_equal(res.error, 0, decimal=6)


@pytest.mark.parametrize("n", np.random.randint(50, 100, (10,)))
def test_permutation_2sided_one_transform_symmetric_umeyama_approx_translate_scale(n):
    r"""Test 2sided-perm with "umeyama_approx" by symmetric with translation and scaling."""
    # define a random, symmetric matrix
    array_a = np.random.uniform(-10.0, 10.0, (n, n))
    array_a = (array_a + array_a.T) / 2.0
    # define array_b by scale-translate array_a and permuting
    shift = np.random.uniform(-10.0, 10.0, n)
    perm = generate_random_permutation_matrix(n)
    array_b = np.dot(perm.T, np.dot((14.7 * array_a + shift), perm))
    # Check
    res = permutation_2sided(
        array_a, array_b, single=True, method="approx-umeyama", translate=True, scale=True
    )
    assert_almost_equal(res.t, perm, decimal=6)
    assert_almost_equal(res.error, 0, decimal=6)


@pytest.mark.parametrize("n, ncol, nrow", np.random.randint(50, 100, (10, 3)))
def test_permutation_2sided_single_transform_umeyama_approx_trans_scale_zero_padding(n, ncol, nrow):
    r"""Test 2sided-perm single transf with "umeyama_approx" by arrays with translate, scaling."""
    # define a random, symmetric matrix
    array_a = np.random.uniform(-10.0, 10.0, (n, n))
    array_a = (array_a + array_a.T) / 2.0
    # check with all possible permutation matrices
    perm = generate_random_permutation_matrix(n)
    # Compute the translated, scaled matrix padded with zeros
    array_b = np.dot(perm.T, np.dot(20 * array_a + 9, perm))
    # pad the matrices with zeros
    array_b = np.concatenate((array_b, np.zeros((n, ncol))), axis=1)
    array_b = np.concatenate((array_b, np.zeros((nrow, n + ncol))), axis=0)
    # Check
    res = permutation_2sided(
        array_a,
        array_b,
        single=True,
        method="approx-umeyama",
        unpad_col=True,
        unpad_row=True,
        translate=True,
        scale=True,
    )
    assert_almost_equal(res.t, perm, decimal=6)
    assert_almost_equal(res.error, 0, decimal=6)


@pytest.mark.parametrize("n", np.random.randint(50, 100, (10,)))
def test_permutation_2sided_1trans_normal1(n):
    r"""Test 2sided-perm with "normal1"."""
    # define a random, symmetric matrix
    a = np.random.uniform(-10.0, 10.0, (n, n))
    a = (a + a.T) / 2.0
    # define array_b by permuting array_a
    p = generate_random_permutation_matrix(n)
    b = np.dot(p.T, np.dot(a, p))
    res = permutation_2sided(a, b, single=True, method="approx-normal1")
    assert_almost_equal(res.t, p, decimal=6)
    assert_almost_equal(res.error, 0, decimal=6)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_permutation_2sided_1trans_normal1_loop(n):
    r"""Test 2sided-perm with "normal1" by small arrays with all permutations."""
    # define a random matrix & symmetrize it
    a = np.random.uniform(-10.0, 10.0, (n, n))
    a = (a + a.T) / 2.0
    # check with all possible permutation matrices
    for comb in itertools.permutations(np.arange(n)):
        p = np.zeros((n, n))
        p[np.arange(n), comb] = 1
        b = np.dot(p.T, np.dot(a, p))
        res = permutation_2sided(a, b, single=True, method="approx-normal1")
        assert_almost_equal(res.t, p, decimal=6)
        assert_almost_equal(res.s, p.T, decimal=6)
        assert_almost_equal(res.error, 0.0, decimal=6)


@pytest.mark.parametrize("n", np.random.randint(50, 100, (10,)))
def test_permutation_2sided_1trans_normal1_translate_scale(n):
    r"""Test 2sided-perm with "normal1" with translation and scaling."""
    # define a random, symmetric matrix
    a = np.random.uniform(-10.0, 10.0, (n, n))
    a = np.dot(a, a.T)
    p = generate_random_permutation_matrix(n)
    b = np.dot(p.T, np.dot((14.7 * a + 3.14), p))
    res = permutation_2sided(a, b, single=True, method="approx-umeyama", translate=True, scale=True)
    res = permutation_2sided(
        res.new_a, res.new_b, single=True, method="nmf", guess_p2=res.t, translate=True, scale=True
    )
    assert_almost_equal(res.t, p, decimal=6)
    assert_almost_equal(res.s, p.T, decimal=6)
    assert_almost_equal(res.error, 0.0, decimal=6)


@pytest.mark.parametrize("n, ncol, nrow, ncol2, nrow2", np.random.randint(50, 100, (10, 5)))
def test_permutation_2sided_1trans_umeyama_nmf_translate_scale_pad(n, ncol, nrow, ncol2, nrow2):
    r"""Test "normal1" by arrays by translation and scaling and zero padding."""
    # define a random matrix & symmetrize it
    a = np.random.uniform(-10.0, 10.0, (n, n))
    a = (a + a.T) / 2.0
    p = generate_random_permutation_matrix(n)
    b = np.dot(p.T, np.dot(a, p))
    # pad the matrices with zeros
    a = np.concatenate((a, np.zeros((n, ncol))), axis=1)
    a = np.concatenate((a, np.zeros((nrow, n + ncol))), axis=0)
    b = np.concatenate((b, np.zeros((n, ncol2))), axis=1)
    b = np.concatenate((b, np.zeros((nrow2, n + ncol2))), axis=0)
    res = permutation_2sided(
        a,
        b,
        single=True,
        method="approx-umeyama",
        unpad_col=True,
        unpad_row=True,
        translate=True,
        scale=True,
    )
    res = permutation_2sided(
        res.new_a,
        res.new_b,
        single=True,
        method="nmf",
        guess_p2=res.t,
        unpad_col=True,
        unpad_row=True,
        translate=True,
        scale=True,
    )
    assert_almost_equal(res.t, p, decimal=6)
    assert_almost_equal(res.s, p.T, decimal=6)
    assert_almost_equal(res.error, 0, decimal=6)


@pytest.mark.parametrize("n", np.random.randint(50, 100, (10,)))
def test_permutation_2sided_single_transform_normal2(n):
    r"""Test 2sided-perm with "normal2"."""
    # define a random, symmetric matrix
    a = np.random.uniform(-10.0, 10.0, (n, n))
    a = (a + a.T) / 2.0
    # define b by permuting a
    p = generate_random_permutation_matrix(n)
    b = np.dot(p.T, np.dot(a, p))
    res = permutation_2sided(a, b, single=True, method="approx-normal2")
    assert_almost_equal(res.t, p, decimal=6)
    assert_almost_equal(res.error, 0, decimal=6)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_permutation_2sided_1trans_small_normal2_loop(n):
    r"""Test 2sided-perm with "normal2" by small arrays over all permutations."""
    # define a random symmetric matrix
    a = np.random.uniform(-10.0, 10.0, (n, n))
    a = (a + a.T) / 2.0
    # check all possible permutation matrices
    for comb in itertools.permutations(np.arange(n)):
        # generate permutation matrix & compute b
        p = np.zeros((n, n))
        p[np.arange(n), comb] = 1
        b = np.dot(p.T, np.dot(a, p))
        res = permutation_2sided(
            a, b, single=True, method="approx-normal2", translate=True, scale=True
        )
        assert_almost_equal(res.t, p, decimal=6)
        assert_almost_equal(res.s, p.T, decimal=6)
        assert_almost_equal(res.error, 0, decimal=6)


@pytest.mark.parametrize("n", np.random.randint(50, 100, (10,)))
def test_permutation_2sided_1trans_umeyama_nmf_translate_scale(n):
    r"""Test 2sided-perm single transform with "normal2" with translation and scaling."""
    # generate random symmetric matrix.
    a = np.random.uniform(-10.0, 10.0, (n, n))
    a = np.dot(a, a.T) / 2.0
    # define array_b by scale-translate array_a and permuting
    p = generate_random_permutation_matrix(n)
    shift = np.random.uniform(-10.0, 10.0, n)
    b = np.dot(p.T, np.dot((14.7 * a + shift), p))
    res = permutation_2sided(a, b, single=True, method="approx-umeyama", translate=True, scale=True)
    res = permutation_2sided(res.new_a, res.new_b, single=True, method="nmf", guess_p2=res.t)
    assert_almost_equal(res.t, p, decimal=6)
    assert_almost_equal(res.s, p.T, decimal=6)
    assert_almost_equal(res.error, 0, decimal=6)


def test_permutation_2sided_invalid_method():
    r"""Test 2sided-perm with invalid mode argument."""
    # define a random matrix
    a = np.arange(16).reshape(4, 4)
    # define array_b by permuting array_a
    p = np.array(
        [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]]
    )
    b = np.dot(p.T, np.dot(a, p))
    assert_raises(ValueError, permutation_2sided, a, b, single=True, method="nature")
    assert_raises(ValueError, permutation_2sided, a, b, single=False, method="boo")


def test_permutation_2sided_2trans_regular():
    r"""Test regular 2sided-perm by practical example."""
    # Example taken from page 64 in parallel solution of
    # svd-related problems, with applications
    # vummath.ma.man.ac.uk/~higham/links/theses/papad93.pdf
    b = np.array(
        [
            [32, 14, 3, 63, 50],
            [24, 22, 1, 56, 4],
            [94, 16, 28, 75, 81],
            [19, 72, 42, 90, 54],
            [71, 85, 10, 96, 58],
        ]
    )
    a = np.array(
        [
            [58, 96, 85, 10, 71],
            [81, 75, 16, 28, 94],
            [4, 56, 22, 1, 24],
            [54, 90, 72, 42, 19],
            [50, 63, 14, 3, 32],
        ]
    )
    p1 = np.array(
        [[0, 0, 0, 0, 1], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [1, 0, 0, 0, 0]]
    )
    p2 = np.array(
        [[0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]]
    )
    result = permutation_2sided(a, b, single=False, method="flip-flop")
    assert_almost_equal(result.s, p1, decimal=6)
    assert_almost_equal(result.t, p2, decimal=6)
    assert_almost_equal(result.error, 0.0, decimal=6)


def test_permutation_2sided_2trans_flipflop():
    r"""Test regular 2sided-perm by 4by4 random arrays."""
    # define a random matrix
    a = np.array(
        [
            [0.74163916, 0.82661152, 0.26856538, 0.23777467],
            [0.06530971, 0.28429819, 0.44244327, 0.79478503],
            [0.83645105, 0.49704302, 0.34292989, 0.01406331],
            [0.04351473, 0.85459821, 0.00663386, 0.62464223],
        ]
    )
    p = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    b = np.dot(np.dot(p, a), p.T)
    result = permutation_2sided(b, a, single=False, method="flip-flop")
    assert_almost_equal(result.s, p.T, decimal=6)
    assert_almost_equal(result.t, p, decimal=6)
    assert_almost_equal(result.error, 0.0, decimal=6)


@pytest.mark.parametrize("n", np.random.randint(3, 6, (3,)))
def test_permutation_2sided_2trans_kopt(n):
    r"""Test regular 2sided permutation with kopt."""
    a = np.random.uniform(-10.0, 10.0, (n, n))
    p1 = generate_random_permutation_matrix(n)
    p2 = generate_random_permutation_matrix(n)
    b = p2.dot(a.dot(p1))
    result = permutation_2sided(b, a, single=False, method="k-opt", options={"k": n})
    assert_almost_equal(result.s, p2, decimal=6)
    assert_almost_equal(result.t, p1.T, decimal=6)
    assert_almost_equal(result.error, 0, decimal=6)


def test_permutation_2sided_2trans_flipflop_rectangular():
    r"""Test regular 2sided-perm by rectangular 4by2 random arrays."""
    a = np.array([[6, 8], [10, 8], [5, 8], [5, 7]])
    p = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    q = np.array([[0, 1], [1, 0]])
    b = np.linalg.multi_dot([p, a, q])
    result = permutation_2sided(a, b, single=False, method="flip-flop")
    assert_almost_equal(result.s, p, decimal=6)
    assert_almost_equal(result.t, q, decimal=6)
    assert_almost_equal(result.error, 0.0, decimal=6)


def test_permutation_2sided_2trans_rectangular_negative():
    r"""Test regular 2sided-perm by unsquared negative 6by4 random arrays."""
    # build random matrix by seed 999
    np.random.seed(999)
    a = np.random.randint(-5, 6, size=(6, 4)).astype(float)
    p1 = np.random.permutation(np.eye(6, 6))
    p2 = np.random.permutation(np.eye(4, 4))
    b = np.linalg.multi_dot([p1, a, p2])
    result = permutation_2sided(b, a, single=False, method="flip-flop")
    assert_almost_equal(result.s, p1, decimal=6)
    assert_almost_equal(result.t, p2, decimal=6)
    assert_almost_equal(result.error, 0.0, decimal=6)


@pytest.mark.parametrize("n", np.random.randint(10, 100, (10,)))
def test_permutation_2sided_1trans_directed(n):
    r"""Test 2sided-perm with single transform and directed."""
    # A random array
    a = np.random.uniform(-10.0, 10.0, (n, n))
    p = generate_random_permutation_matrix(n)
    b = np.dot(p.T, np.dot(a, p))
    # Procrustes with no translate and scale
    res = permutation_2sided(a, b, single=True, method="approx-umeyama")
    res = permutation_2sided(res.new_a, res.new_b, single=True, method="nmf", guess_p2=res.t)
    assert_almost_equal(res.t, p, decimal=6)
    assert_almost_equal(res.s, p.T, decimal=6)
    assert_almost_equal(res.error, 0.0, decimal=6)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_permutation_2sided_1trans_directed_all_permutations(n):
    r"""Test 2sided-perm with "directed" over all permutations."""
    # define a random matrix
    a = np.random.uniform(-10.0, 10.0, (n, n))
    # check with all possible permutation matrices
    for comb in itertools.permutations(np.arange(n)):
        perm = np.zeros((n, n))
        perm[np.arange(n), comb] = 1
        b = np.dot(perm.T, np.dot(a, perm))
        res = permutation_2sided(a, b, single=True, method="approx-umeyama")
        res = permutation_2sided(res.new_a, res.new_b, single=True, method="nmf", guess_p2=res.t)
        assert_almost_equal(res.t, perm, decimal=6)
        assert_almost_equal(res.s, perm.T, decimal=6)
        assert_almost_equal(res.error, 0, decimal=6)


@pytest.mark.parametrize("n", np.random.randint(50, 100, (10,)))
def test_permutation_2sided_1trans_directed_translate_scale(n):
    r"""Test 2sided-perm single transform with "directed" and translation, and scaling."""
    a = np.random.uniform(-10.0, 10.0, (n, n))
    p = generate_random_permutation_matrix(n)
    b = np.dot(p.T, np.dot(15.3 * a + 5.45, p))
    res = permutation_2sided(a, b, single=True, method="approx-umeyama", translate=True, scale=True)
    # res = permutation_2sided(res.new_a, res.new_b, single=True, method="nmf", guess_p2=res.t,
    #                          translate=True, scale=True)
    assert_almost_equal(res.t, p, decimal=6)
    assert_almost_equal(res.s, p.T, decimal=6)
    assert_almost_equal(res.error, 0.0, decimal=6)


@pytest.mark.parametrize("n", np.random.randint(50, 100, (10,)))
def test_permutation_2sided_1trans_directed_translate_scale_padding(n):
    r"""Test 2sided-perm single transform directed with translation, scaling and zero paddings."""
    a = np.random.uniform(-10.0, 10.0, (n, n))
    p = generate_random_permutation_matrix(n)
    b = np.dot(p.T, np.dot(15.3 * a + 5.45, p))
    # pad the matrices with zeros
    a = np.concatenate((a, np.zeros((n, 3))), axis=1)
    a = np.concatenate((a, np.zeros((10, n + 3))), axis=0)
    b = np.concatenate((b, np.zeros((n, 2))), axis=1)
    b = np.concatenate((b, np.zeros((6, n + 2))), axis=0)
    res = permutation_2sided(
        a,
        b,
        single=True,
        method="approx-umeyama",
        unpad_col=True,
        unpad_row=True,
        translate=True,
        scale=True,
    )
    assert_almost_equal(res.t, p, decimal=6)
    assert_almost_equal(res.s, p.T, decimal=6)
    assert_almost_equal(res.error, 0.0, decimal=6)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_permutation_2sided_1trans_with_kopt_all_permutations(n):
    r"""Test 2sided-perm single transform with kopt over all permutations."""
    # define a random matrix
    a = np.random.uniform(-10.0, 10.0, (n, n))
    # check with all possible permutation matrices
    for comb in itertools.permutations(np.arange(n)):
        p = np.zeros((n, n))
        p[np.arange(n), comb] = 1
        # get array_b by permutation
        b = np.dot(p.T, np.dot(a, p))
        result = permutation_2sided(a, b, single=True, method="k-opt", options={"k": n})
        assert_almost_equal(result.t, p, decimal=6)
        assert_almost_equal(result.error, 0, decimal=6)


@pytest.mark.parametrize("n", np.random.randint(3, 8, (3,)))
def test_permutation_2sided_explicit_translate_scale(n):
    r"""Test 2-sided permutation with explicit method by 4by4 method."""
    # define a random matrix
    a = np.random.uniform(-10.0, 10.0, (n, n))
    # define array_b by scale-translate array_a and permuting
    perm = generate_random_permutation_matrix(n)
    b = np.dot(perm.T, np.dot((14.7 * a + 2.14), perm))
    # check
    result = permutation_2sided(
        a, b, single=True, method="k-opt", translate=True, scale=True, options={"k": n}
    )
    assert_almost_equal(result.t, perm, decimal=6)
    assert_almost_equal(result.error, 0, decimal=6)


@pytest.mark.parametrize("n, ncol, nrow", np.random.randint(5, 10, (5, 3)))
def test_permutation_2sided_single_kopt_translate_scale_zero_padding(n, ncol, nrow):
    r"""Test 2sided perm, single transform with kopt with translation, scaling and zero padding."""
    # define a random matrix
    a = np.random.uniform(-10.0, 10.0, (n, n))
    perm = generate_random_permutation_matrix(n)
    b = np.dot(perm.T, np.dot(20 * a + 9, perm))
    # pad the matrices with zeros
    b = np.concatenate((b, np.zeros((n, ncol))), axis=1)
    b = np.concatenate((b, np.zeros((nrow, n + ncol))), axis=0)
    result = permutation_2sided(
        a,
        b,
        single=True,
        method="k-opt",
        unpad_col=True,
        unpad_row=True,
        translate=True,
        scale=True,
        options={"k": n},
    )
    assert_almost_equal(result.t, perm, decimal=6)
    assert_almost_equal(result.error, 0, decimal=6)


def test_permutation_2sided_invalid_input_kopt_single_transform():
    r"""Test 2-sided permutation with invalid inputs to kopt and single transform."""
    # define a random matrix and symmetric matrix
    a = np.array([[4, 5, 3, 3], [5, 7, 3, 5], [3, 3, 2, 2], [3, 5, 2, 5]])
    p = np.array(
        [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]]
    )
    b = np.dot(p.T, np.dot(a, p))
    # check
    assert_raises(TypeError, permutation_2sided, a, b, single="invalid")
    assert_raises(TypeError, permutation_2sided, a, b, single=True, kopt=20.1)
    assert_raises(ValueError, permutation_2sided, a, np.eye(20), single=True, pad=False)


# def test_permutation_2sided_dominators_zero():
#     """Test two-sided permutations which has zeros in the dominator in updating step."""
#     a = np.array([[6, 3, 0, 0],
#                   [3, 6, 1, 0],
#                   [0, 1, 6, 2],
#                   [0, 0, 2, 6]])
#     b = np.array([[6, 3, 0, 0, 0, 0, 0],
#                   [3, 6, 1, 0, 0, 0, 0],
#                   [0, 1, 6, 1, 0, 1, 1],
#                   [0, 0, 1, 6, 2, 0, 0],
#                   [0, 0, 0, 2, 6, 0, 0],
#                   [0, 0, 1, 0, 0, 6, 0],
#                   [0, 0, 1, 0, 0, 0, 6]])
#     res = permutation_2sided(a, b, single=True, method="approx-normal1", pad=True,
#                              unpad_col=False, unpad_row=False, scale=False)
#     res = permutation_2sided(res.new_a, res.new_b, single=True, method="nmf", guess_p2=res.t,
#                              unpad_col=False, unpad_row=False, scale=False)
#     perm = np.array([[1, 0, 0, 0, 0, 0, 0],
#                      [0, 1, 0, 0, 0, 0, 0],
#                      [0, 0, 0, 1, 0, 0, 0],
#                      [0, 0, 0, 0, 1, 0, 0],
#                      [0, 0, 1, 0, 0, 0, 0],
#                      [0, 0, 0, 0, 0, 1, 0],
#                      [0, 0, 0, 0, 0, 0, 1]])
#     assert_almost_equal(res["t"], perm)
