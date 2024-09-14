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
"""Testings for softassign module."""

import itertools
import warnings

import numpy as np
from numpy.testing import assert_almost_equal, assert_raises

from procrustes import softassign


def test_softassign_4by4():
    r"""Test softassign by a 4by4 matrix."""
    # define a random matrix
    array_a = np.array([[4, 5, 3, 3], [5, 7, 3, 5], [3, 3, 2, 2], [3, 5, 2, 5]])
    # define array_b by permuting array_a
    perm = np.array(
        [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]]
    )
    array_b = np.dot(perm.T, np.dot(array_a, perm))
    # Check
    res = softassign(array_a, array_b, unpad_col=False, unpad_row=False)
    assert_almost_equal(res["t"], perm, decimal=6)
    assert_almost_equal(res["error"], 0, decimal=6)


def test_softassign_4by4_loop():
    r"""Test softassign by a 4by4 matrix with all possible permutation matrices."""
    # define a random symmetric matrix
    array_a = np.array([[4, 5, 3, 3], [5, 7, 3, 5], [3, 3, 2, 2], [3, 5, 2, 5]])
    # check with all possible permutation matrices
    for comb in itertools.permutations(np.arange(4)):
        perm = np.zeros((4, 4))
        perm[np.arange(4), comb] = 1
        # get array_b by permutation
        array_b = np.dot(perm.T, np.dot(array_a, perm))
        # Check
        res = softassign(array_a, array_b, unpad_col=False, unpad_row=False)
        assert_almost_equal(res["t"], perm, decimal=6)
        assert_almost_equal(res["error"], 0, decimal=6)


def test_softassign_4by4_loop_negative():
    r"""Test softassign by a 4by4 negative matrix with all possible permutation matrices."""
    # define a random matrix
    array_a = np.array([[4, 5, -3, 3], [5, 7, 3, -5], [-3, 3, 2, 2], [3, -5, 2, 5]])
    # check with all possible permutation matrices
    for comb in itertools.permutations(np.arange(4)):
        # Compute the permutation matrix
        perm = np.zeros((4, 4))
        perm[np.arange(4), comb] = 1
        if not np.allclose(perm, np.eye(4)):
            # Compute the translated, scaled matrix padded with zeros
            array_b = np.dot(perm.T, np.dot(array_a, perm))
            # Check
            res = softassign(array_a, array_b, unpad_col=False, unpad_row=False)
            assert_almost_equal(res["t"], perm, decimal=6)
            assert_almost_equal(res["error"], 0, decimal=6)


def test_softassign_4by4_translate_scale():
    r"""Test softassign by 4by4 matrix with translation and scaling."""
    # define a random matrix
    array_a = np.array([[5.0, 2.0, 1.0], [4.0, 6.0, 1.0], [1.0, 6.0, 3.0]])
    array_a = np.dot(array_a, array_a.T)
    # define array_b by scale-translate array_a and permuting
    perm = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    array_b = np.dot(perm.T, np.dot((14.7 * array_a + 3.14), perm))
    # Check
    res = softassign(array_a, array_b, translate=True, scale=True, unpad_col=False, unpad_row=False)
    assert_almost_equal(res["t"], perm, decimal=6)
    assert_almost_equal(res["error"], 0, decimal=6)


def test_softassign_4by4_translate_scale_loop():
    r"""Test softassign by 4by4 matrix with all permutations with translation and scaling."""
    # define a random matrix
    array_a = np.array([[4, 5, -3, 3], [5, 7, 3, -5], [-3, 3, 2, 2], [3, -5, 2, 5]])
    # check with all possible permutation matrices
    for comb in itertools.permutations(np.arange(4)):
        # Compute the permutation matrix
        perm = np.zeros((4, 4))
        perm[np.arange(4), comb] = 1
        # Compute the translated, scaled matrix padded with zeros
        array_b = np.dot(perm.T, np.dot(3 * array_a + 10, perm))
        # Check
        res = softassign(array_a, array_b, translate=True, scale=True)
        assert_almost_equal(res["t"], perm, decimal=6)
        assert_almost_equal(res["error"], 0, decimal=6)


def test_softassign_4by4_translate_scale_zero_padding():
    r"""Test softassign by zero padded 4by4 matrix."""
    # define a random matrix
    array_a = np.array([[4, 5, -3, 3], [5, 7, 3, -5], [-3, 3, 2, 2], [3, -5, 2, 5]])
    # check with all possible permutation matrices
    perm = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    # Compute the translated, scaled matrix padded with zeros
    array_b = np.dot(perm.T, np.dot(array_a, perm))
    # pad the matrices with zeros
    array_a = np.concatenate((array_a, np.zeros((4, 3))), axis=1)
    array_a = np.concatenate((array_a, np.zeros((10, 7))), axis=0)
    array_b = np.concatenate((array_b, np.zeros((4, 2))), axis=1)
    array_b = np.concatenate((array_b, np.zeros((6, 6))), axis=0)
    # Check
    res = softassign(array_a, array_b, translate=False, scale=False, unpad_col=True, unpad_row=True)
    assert_almost_equal(res["t"], perm, decimal=6)
    assert_almost_equal(res["error"], 0, decimal=6)


def test_softassign_practical_example():
    r"""Test softassign by a practical example."""
    # Example taken from page 64 in parallel solution of
    # svd-related problems, with applications
    # vummath.ma.man.ac.uk/~higham/links/theses/papad93.pdf
    # https://books.google.ca/books/about/Parallel_Solution_of_
    # SVD_related_Problem.html?id=_aVWcgAACAAJ&redir_esc=y
    array_a = np.array(
        [
            [32, 14, 3, 63, 50],
            [24, 22, 1, 56, 4],
            [94, 16, 28, 75, 81],
            [19, 72, 42, 90, 54],
            [71, 85, 10, 96, 58],
        ]
    )
    perm = np.array(
        [[0, 0, 0, 0, 1], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [1, 0, 0, 0, 0]]
    )
    array_b = np.dot(perm.T, np.dot(array_a, perm))
    # Check
    res = softassign(
        array_a, array_b, translate=False, scale=False, unpad_col=False, unpad_row=False
    )
    assert_almost_equal(res["t"], perm, decimal=6)
    assert_almost_equal(res["error"], 0, decimal=6)


def test_softassign_random_noise():
    r"""Test softassign by a practical example with random noise."""
    # Example based on page 64 in parallel solution of
    # svd-related problems, with applications
    # vummath.ma.man.ac.uk/~higham/links/theses/papad93.pdf
    # https://books.google.ca/books/about/Parallel_Solution_of_SVD_related_
    # Problem.html?id=_aVWcgAACAAJ&redir_esc=y
    array_a = np.array(
        [
            [32, 14, 3, 63, 50],
            [24, 22, 1, 56, 4],
            [94, 16, 28, 75, 81],
            [19, 72, 42, 90, 54],
            [71, 85, 10, 96, 58],
        ]
    )
    perm = np.array(
        [[0, 0, 0, 0, 1], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [1, 0, 0, 0, 0]]
    )
    array_b = np.dot(perm.T, np.dot(array_a, perm)) + np.random.randn(5, 5)
    # Check
    res = softassign(
        array_a, array_b, translate=False, scale=False, unpad_col=False, unpad_row=False
    )
    assert_almost_equal(res["t"], perm, decimal=6)


def test_softassign_invalid_beta_r():
    r"""Test softassign by invalid beta_r value."""
    # define a random matrix and symmetric matrix
    array_a = np.array([[4, 5, 3, 3], [5, 7, 3, 5], [3, 3, 2, 2], [3, 5, 2, 5]])
    # define array_b by permuting array_a
    perm = np.array(
        [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]]
    )
    array_b = np.dot(perm.T, np.dot(array_a, perm))
    # Check
    assert_raises(ValueError, softassign, array_a, array_b, beta_r=0.5)


def test_softassign_wrong_shapes():
    r"""Test softassign with wrong shapes for the a, b input matrices."""
    array_a = np.ones((10, 5))
    array_b = np.ones((10, 10))
    # Test A and B are not square matrices.
    assert_raises(ValueError, softassign, array_a, array_b, pad=False)
    # Test B is not square matrices.
    array_a = np.ones((10, 10))
    array_b = np.ones((10, 5))
    assert_raises(ValueError, softassign, array_a, array_b, pad=False)
    # Test A, B are square but with different shape.
    array_a = np.ones((10, 10))
    array_b = np.ones((20, 20))
    assert_raises(ValueError, softassign, array_a, array_b, pad=False)


def test_softassign_4by4_beta_0():
    r"""Test softassign by 4by4 matrix specified beta_0.."""
    # define a random matrix
    array_a = np.array([[4, 5, -3, 3], [5, 7, 3, -5], [-3, 3, 2, 2], [3, -5, 2, 5]])
    # random permutation matrix_rank
    perm = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    array_b = np.dot(perm.T, np.dot(array_a, perm))
    # Check
    res = softassign(
        array_a,
        array_b,
        translate=False,
        scale=False,
        beta_0=1.0e-6,
        adapted=False,
        epsilon_soft=1e-8,
    )
    assert_almost_equal(res["t"], perm, decimal=6)
    assert_almost_equal(res["error"], 0, decimal=6)


def test_softassign_4by4_anneal_steps():
    r"""Test softassign by 4by4 matrix specified annealing steps."""
    # define a random matrix
    array_a = np.array([[4, 5, -3, 3], [5, 7, 3, -5], [-3, 3, 2, 2], [3, -5, 2, 5]])
    # random permutation matrix_rank
    perm = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    array_b = np.dot(perm.T, np.dot(array_a, perm))
    # Check
    res = softassign(array_a, array_b, translate=False, scale=False, iteration_anneal=165)
    assert_almost_equal(res["t"], perm, decimal=6)
    assert_almost_equal(res["error"], 0, decimal=6)


def test_softassign_missing_iteration_anneal_beta_f():
    r"""Test softassign by missing iteration_anneal and beta_f."""
    # define a random matrix
    array_a = np.array([[4, 5, -3, 3], [5, 7, 3, -5], [-3, 3, 2, 2], [3, -5, 2, 5]])
    # random permutation matrix_rank
    perm = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    array_b = np.dot(perm.T, np.dot(array_a, perm))
    assert_raises(ValueError, softassign, array_a, array_b, iteration_anneal=None, beta_f=None)


def test_softassign_m_guess():
    r"""Test softassign by given initial permutation guess."""
    # define a random matrix
    array_a = np.array([[4, 5, -3, 3], [5, 7, 3, -5], [-3, 3, 2, 2], [3, -5, 2, 5]])
    # random permutation matrix_rank
    perm = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    array_b = np.dot(perm.T, np.dot(array_a, perm))
    m_guess1 = np.array([[0, -0.98, 0, 0], [0.8, 0, 0, 0], [0, 0, 1.01, 0], [0, 0, 0, 1]])
    m_guess2 = np.array([[0, 0.98, 0, 0], [0.8, 0, 0, 0], [0, 0, 1.01, 0], [0, 0, 0, 1]])
    m_guess3 = np.array([[0, 0.98, 0, 0], [0, 0, 1.01, 0], [0, 0, 0, 1]])
    # check assert raises
    assert_raises(ValueError, softassign, array_a, array_b, m_guess=m_guess1)
    # check if initial guess works
    res = softassign(array_a, array_b, translate=False, scale=False, m_guess=m_guess2)
    assert_almost_equal(res["t"], perm, decimal=6)
    assert_almost_equal(res["error"], 0, decimal=6)
    # check if initial guess given shape not matching
    with warnings.catch_warnings(record=True) as warn_info:
        res = softassign(array_a, array_b, translate=False, scale=False, m_guess=m_guess3)
        # catch the error information
        assert len(warn_info) == 1
        assert not str(warn_info[0].message).startswith("We must specify")
        # check the results
        assert_almost_equal(res["t"], perm, decimal=6)
        assert_almost_equal(res["error"], 0, decimal=6)


def test_softassign_kopt():
    """Test softassign with k-opt heuristic local search."""
    rng = np.random.default_rng(seed=3456)
    # define the input matrices
    array_a = rng.integers(low=-5, high=10, size=(10, 10))
    # generate random permutation matrix
    size = array_a.shape[0]
    perm = np.zeros((size, size))
    idx = np.arange(size)
    rng.shuffle(idx)
    perm[np.arange(size), idx] = 1
    # build matrix array_b
    array_b = np.dot(perm.T, np.dot(array_a, perm))
    # softassign without kopt
    # default parameters will lead to error=0
    # changes the parameters to force it fail
    res_no_kopt = softassign(
        array_a,
        array_b,
        translate=False,
        scale=False,
        unpad_col=False,
        unpad_row=False,
        iteration_soft=1,
        iteration_sink=1,
        beta_r=1.05,
        beta_f=1.0e3,
        epsilon=0.05,
        epsilon_soft=1.0e-3,
        epsilon_sink=1.0e-3,
        k=0.15,
        gamma_scaler=1.5,
        n_stop=2,
        kopt=False,
    )
    # softassign with kopt
    res_with_kopt = softassign(
        array_a,
        array_b,
        translate=False,
        scale=False,
        unpad_col=False,
        unpad_row=False,
        iteration_soft=1,
        iteration_sink=1,
        beta_r=1.05,
        beta_f=1.0e3,
        epsilon=0.05,
        epsilon_soft=1.0e-3,
        epsilon_sink=1.0e-3,
        k=0.15,
        gamma_scaler=1.5,
        n_stop=2,
        kopt=True,
    )
    assert res_no_kopt["error"] >= res_with_kopt["error"]
