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
    perm, kopt_error = kopt_heuristic_single(arr_a, arr_b, perm_guess, 3)
    assert_equal(perm, perm_exact)
    assert kopt_error == 0
    # test the error exceptions
    assert_raises(ValueError, kopt_heuristic_single, arr_a, arr_b, perm_guess, 1)


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
    error = compute_error(arr_b, arr_a, perm1_shuff.T, perm2_shuff)
    perm_left, perm_right, kopt_error = kopt_heuristic_double(a=arr_a, b=arr_b, p1=perm1_shuff,
                                                              p2=perm2_shuff, k=4)
    _, _, kopt_error = kopt_heuristic_double(a=arr_a, b=arr_b, p1=perm_left, p2=perm_right, k=3)
    assert kopt_error <= error
    assert kopt_error == 0
