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

import warnings

import itertools
import numpy as np

from numpy.testing import assert_raises, assert_almost_equal

from procrustes import softassign


def test_softassign_4by4():
    r"""Test softassign by a 4by4 matrix."""
    # define a random matrix
    array_a = np.array([[4, 5, 3, 3], [5, 7, 3, 5], [3, 3, 2, 2], [3, 5, 2, 5]])
    # define array_b by permuting array_a
    perm = np.array([[0., 0., 1., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.],
                     [0., 1., 0., 0.]])
    array_b = np.dot(perm.T, np.dot(array_a, perm))
    # Check
    new_a, new_b, M_ai, e_opt = softassign(array_a, array_b,
                                           remove_zero_col=False,
                                           remove_zero_row=False)
    assert_almost_equal(M_ai, perm, decimal=6)
    assert_almost_equal(e_opt, 0, decimal=6)


def test_softassign_4by4_loop():
    r"""Test softassign by a 4by4 matrix with all possible permutation matrices."""
    # define a random symmetric matrix
    array_a = np.array([[4, 5, 3, 3], [5, 7, 3, 5],
                        [3, 3, 2, 2], [3, 5, 2, 5]])
    # check with all possible permutation matrices
    for comb in itertools.permutations(np.arange(4)):
        perm = np.zeros((4, 4))
        perm[np.arange(4), comb] = 1
        # get array_b by permutation
        array_b = np.dot(perm.T, np.dot(array_a, perm))
        # Check
        new_a, new_b, U, e_opt = softassign(array_a, array_b,
                                            remove_zero_col=False,
                                            remove_zero_row=False)
        assert_almost_equal(U, perm, decimal=6)
        assert_almost_equal(e_opt, 0, decimal=6)


def test_softassign_4by4_loop_negative():
    r"""Test softassign by a 4by4 negative matrix with all possible permutation matrices."""
    # define a random matrix
    array_a = np.array([[4, 5, -3, 3], [5, 7, 3, -5],
                        [-3, 3, 2, 2], [3, -5, 2, 5]])
    # check with all possible permutation matrices
    for comb in itertools.permutations(np.arange(4)):
        # Compute the permutation matrix
        perm = np.zeros((4, 4))
        perm[np.arange(4), comb] = 1
        if not np.allclose(perm, np.eye(4)):
            # Compute the translated, scaled matrix padded with zeros
            array_b = np.dot(perm.T, np.dot(array_a, perm))
            # Check
            new_a, new_b, U, e_opt = softassign(array_a, array_b,
                                                remove_zero_row=False,
                                                remove_zero_col=False)
            assert_almost_equal(U, perm, decimal=6)
            assert_almost_equal(e_opt, 0, decimal=6)


def test_softassign_4by4_translate_scale():
    r"""Test softassign by 4by4 matrix with translation and scaling."""
    # define a random matrix
    array_a = np.array([[5., 2., 1.], [4., 6., 1.], [1., 6., 3.]])
    array_a = np.dot(array_a, array_a.T)
    # define array_b by scale-translate array_a and permuting
    perm = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
    array_b = np.dot(perm.T, np.dot((14.7 * array_a + 3.14), perm))
    # Check
    new_a, new_b, U, e_opt = softassign(array_a, array_b,
                                        translate=True, scale=True,
                                        remove_zero_row=False,
                                        remove_zero_col=False)
    assert_almost_equal(U, perm, decimal=6)
    assert_almost_equal(e_opt, 0, decimal=6)


def test_softassign_4by4_translate_scale_loop():
    r"""Test softassign by 4by4 matrix with all permutations with translation and scaling."""
    # define a random matrix
    array_a = np.array([[4, 5, -3, 3], [5, 7, 3, -5],
                        [-3, 3, 2, 2], [3, -5, 2, 5]])
    # check with all possible permutation matrices
    for comb in itertools.permutations(np.arange(4)):
        # Compute the permutation matrix
        perm = np.zeros((4, 4))
        perm[np.arange(4), comb] = 1
        # Compute the translated, scaled matrix padded with zeros
        array_b = np.dot(perm.T, np.dot(3 * array_a + 10, perm))
        # Check
        new_a, new_b, U, e_opt = softassign(array_a,
                                            array_b,
                                            translate=True,
                                            scale=True)
        assert_almost_equal(U, perm, decimal=6)
        assert_almost_equal(e_opt, 0, decimal=6)


def test_softassign_4by4_translate_scale_zero_padding():
    r"""Test softassign by zero padded 4by4 matrix."""
    # define a random matrix
    array_a = np.array(
        [[4, 5, -3, 3], [5, 7, 3, -5], [-3, 3, 2, 2], [3, -5, 2, 5]])
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
    new_a, new_b, U, e_opt = softassign(array_a, array_b,
                                        translate=False,
                                        scale=False,
                                        remove_zero_row=True,
                                        remove_zero_col=True)
    assert_almost_equal(U, perm, decimal=6)
    assert_almost_equal(e_opt, 0, decimal=6)


def test_softassign_practical_example():
    r"""Test softassign by a practical example."""
    # Example taken from page 64 in parallel solution of
    # svd-related problems, with applications
    # vummath.ma.man.ac.uk/~higham/links/theses/papad93.pdf
    # https://books.google.ca/books/about/Parallel_Solution_of_
    # SVD_related_Problem.html?id=_aVWcgAACAAJ&redir_esc=y
    array_a = np.array([[32, 14, 3, 63, 50],
                        [24, 22, 1, 56, 4],
                        [94, 16, 28, 75, 81],
                        [19, 72, 42, 90, 54],
                        [71, 85, 10, 96, 58]])
    perm = np.array([[0, 0, 0, 0, 1],
                     [0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0],
                     [1, 0, 0, 0, 0]])
    array_b = np.dot(perm.T, np.dot(array_a, perm))
    # Check
    new_a, new_b, U, e_opt = softassign(array_a, array_b,
                                        translate=False,
                                        scale=False,
                                        remove_zero_row=False,
                                        remove_zero_col=False)
    assert_almost_equal(U, perm, decimal=6)
    assert_almost_equal(e_opt, 0, decimal=6)


def test_softassign_random_noise():
    r"""Test softassign by a practical example with random noise."""
    # Example based on page 64 in parallel solution of
    # svd-related problems, with applications
    # vummath.ma.man.ac.uk/~higham/links/theses/papad93.pdf
    # https://books.google.ca/books/about/Parallel_Solution_of_SVD_related_
    # Problem.html?id=_aVWcgAACAAJ&redir_esc=y
    array_a = np.array([[32, 14, 3, 63, 50],
                        [24, 22, 1, 56, 4],
                        [94, 16, 28, 75, 81],
                        [19, 72, 42, 90, 54],
                        [71, 85, 10, 96, 58]])
    perm = np.array([[0, 0, 0, 0, 1],
                     [0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0],
                     [1, 0, 0, 0, 0]])
    array_b = np.dot(perm.T, np.dot(array_a, perm)) + np.random.randn(5, 5)
    # Check
    new_a, new_b, U, e_opt = softassign(array_a, array_b,
                                        translate=False,
                                        scale=False,
                                        remove_zero_row=False,
                                        remove_zero_col=False)
    assert_almost_equal(U, perm, decimal=6)


def test_softassign_invalid_beta_r():
    r"""Test softassign by invalid beta_r value."""
    # define a random matrix and symmetric matrix
    array_a = np.array([[4, 5, 3, 3], [5, 7, 3, 5], [3, 3, 2, 2], [3, 5, 2, 5]])
    # define array_b by permuting array_a
    perm = np.array([[0., 0., 1., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.],
                     [0., 1., 0., 0.]])
    array_b = np.dot(perm.T, np.dot(array_a, perm))
    # Check
    assert_raises(ValueError, softassign, array_a, array_b, beta_r=0.5)


def test_softassign_4by4_beta_0():
    r"""Test softassign by 4by4 matrix specified beta_0.."""
    # define a random matrix
    array_a = np.array([[4, 5, -3, 3], [5, 7, 3, -5],
                        [-3, 3, 2, 2], [3, -5, 2, 5]])
    # random permutation matrix_rank
    perm = np.array([[0, 1, 0, 0], [1, 0, 0, 0],
                     [0, 0, 1, 0], [0, 0, 0, 1]])
    array_b = np.dot(perm.T, np.dot(array_a, perm))
    # Check
    new_a, new_b, U, e_opt = softassign(array_a,
                                        array_b,
                                        beta_0=1.e-6,
                                        translate=False,
                                        scale=False)
    assert_almost_equal(U, perm, decimal=6)
    assert_almost_equal(e_opt, 0, decimal=6)


def test_softassign_4by4_anneal_steps():
    r"""Test softassign by 4by4 matrix specified annealing steps."""
    # define a random matrix
    array_a = np.array([[4, 5, -3, 3], [5, 7, 3, -5],
                        [-3, 3, 2, 2], [3, -5, 2, 5]])
    # random permutation matrix_rank
    perm = np.array([[0, 1, 0, 0], [1, 0, 0, 0],
                     [0, 0, 1, 0], [0, 0, 0, 1]])
    array_b = np.dot(perm.T, np.dot(array_a, perm))
    # Check
    new_a, new_b, U, e_opt = softassign(array_a,
                                        array_b,
                                        iteration_anneal=165,
                                        translate=False,
                                        scale=False)
    assert_almost_equal(U, perm, decimal=6)
    assert_almost_equal(e_opt, 0, decimal=6)


def test_softassign_missing_iteration_anneal_beta_f():
    r"""Test softassign by missing iteration_anneal and beta_f."""
    # define a random matrix
    array_a = np.array([[4, 5, -3, 3], [5, 7, 3, -5],
                        [-3, 3, 2, 2], [3, -5, 2, 5]])
    # random permutation matrix_rank
    perm = np.array([[0, 1, 0, 0], [1, 0, 0, 0],
                     [0, 0, 1, 0], [0, 0, 0, 1]])
    array_b = np.dot(perm.T, np.dot(array_a, perm))
    assert_raises(ValueError, softassign, array_a, array_b, iteration_anneal=None, beta_f=None)


def test_softassign_M_guess():
    r"""Test softassign by given initial permutation guess."""
    # define a random matrix
    array_a = np.array([[4, 5, -3, 3], [5, 7, 3, -5],
                        [-3, 3, 2, 2], [3, -5, 2, 5]])
    # random permutation matrix_rank
    perm = np.array([[0, 1, 0, 0], [1, 0, 0, 0],
                     [0, 0, 1, 0], [0, 0, 0, 1]])
    array_b = np.dot(perm.T, np.dot(array_a, perm))
    M_guess1 = np.array([[0, -0.98, 0, 0], [0.8, 0, 0, 0],
                         [0, 0, 1.01, 0], [0, 0, 0, 1]])
    M_guess2 = np.array([[0, 0.98, 0, 0], [0.8, 0, 0, 0],
                         [0, 0, 1.01, 0], [0, 0, 0, 1]])
    M_guess3 = np.array([[0, 0.98, 0, 0], [0, 0, 1.01, 0], [0, 0, 0, 1]])
    # check assert raises
    assert_raises(ValueError, softassign, array_a, array_b, M_guess=M_guess1)
    # check if initial guess works
    new_a, new_b, U, e_opt = softassign(array_a,
                                        array_b,
                                        M_guess=M_guess2,
                                        translate=False,
                                        scale=False)
    assert_almost_equal(U, perm, decimal=6)
    assert_almost_equal(e_opt, 0, decimal=6)
    # check if initial guess given shape not matching
    with warnings.catch_warnings(record=True) as w:
        new_a, new_b, U, e_opt = softassign(array_a,
                                            array_b,
                                            M_guess=M_guess3,
                                            translate=False,
                                            scale=False)
        # catch the error information
        assert len(w) == 1
        assert not str(w[0].message).startswith("We must specify")
        # chekc the results
        assert_almost_equal(U, perm, decimal=6)
        assert_almost_equal(e_opt, 0, decimal=6)
