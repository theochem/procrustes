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
"""Test Module for Kopt."""

import numpy as np
from numpy.testing import assert_equal, assert_raises
from procrustes.kopt import kopt_heuristic_double, kopt_heuristic_single
from procrustes.utils import compute_error


def test_kopt_heuristic_single_raises():
    r"""Test k-opt heuristic search algorithm raises."""
    # check raises for k
    assert_raises(ValueError, kopt_heuristic_single, lambda p: np.sum(p), np.eye(2), 1)
    assert_raises(ValueError, kopt_heuristic_single, lambda p: np.sum(p), np.eye(3), -2)
    assert_raises(ValueError, kopt_heuristic_single, lambda p: np.sum(p), np.eye(5), 6)
    # check raises for p0
    assert_raises(ValueError, kopt_heuristic_single, lambda p: np.sum(p), np.ones(3), 2)
    assert_raises(ValueError, kopt_heuristic_single, lambda p: np.sum(p), np.ones((2, 3)), 2)
    assert_raises(ValueError, kopt_heuristic_single, lambda p: np.sum(p), np.ones((4, 4)), 2)
    assert_raises(ValueError, kopt_heuristic_single, lambda p: np.sum(p), np.zeros((5, 5)), 2)
    assert_raises(ValueError, kopt_heuristic_single, lambda p: np.sum(p), np.eye(6) + 0.1, 2)


def test_kopt_heuristic_single():
    r"""Test k-opt heuristic search algorithm."""
    arr_a = np.array([[3, 6, 1, 0, 7],
                      [4, 5, 2, 7, 6],
                      [8, 6, 6, 1, 7],
                      [4, 4, 7, 9, 4],
                      [4, 8, 0, 3, 1]])
    arr_b = np.array([[1, 8, 0, 4, 3],
                      [6, 5, 2, 4, 7],
                      [7, 6, 6, 8, 1],
                      [7, 6, 1, 3, 0],
                      [4, 4, 7, 4, 9]])
    perm_guess = np.array([[0, 0, 1, 0, 0],
                           [1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1],
                           [0, 1, 0, 0, 0]])
    perm_exact = np.array([[0, 0, 0, 1, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0]])

    perm, error = kopt_heuristic_single(lambda p: compute_error(arr_a, arr_b, p, p.T), perm_guess)
    assert_equal(perm, perm_exact)
    assert error == 0


def test_kopt_heuristic_double():
    r"""Test double sided k-opt heuristic search algorithm."""
    np.random.seed(998)
    arr_b = np.random.randint(low=-10, high=10, size=(4, 3)).astype(np.float)
    perm1 = np.array([[0., 0., 0., 1.],
                      [0., 1., 0., 0.],
                      [1., 0., 0., 0.],
                      [0., 0., 1., 0.]])
    perm2 = np.array([[0., 0., 1.],
                      [1., 0., 0.],
                      [0., 1., 0.]])
    arr_a = np.linalg.multi_dot([perm1.T, arr_b, perm2])
    # shuffle the permutation matrices
    perm1_shuff = np.array([[0., 0., 0., 1.],
                            [1., 0., 0., 0.],
                            [0., 1., 0., 0.],
                            [0., 0., 1., 0.]])
    perm2_shuff = np.array([[1., 0., 0.],
                            [0., 0., 1.],
                            [0., 1., 0.]])
    error = compute_error(arr_b, arr_a, perm2_shuff, perm1_shuff.T)
    fun_error = lambda p1, p2: compute_error(arr_b, arr_a, p2, p1.T)
    perm_left, perm_right, kopt_error = kopt_heuristic_double(fun_error, p1=perm1_shuff,
                                                              p2=perm2_shuff, k=3)
    _, _, kopt_error = kopt_heuristic_double(fun_error, p1=perm_left, p2=perm_right, k=3)
    assert kopt_error <= error
    assert kopt_error == 0
