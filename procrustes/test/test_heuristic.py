# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2020 The Procrustes Development Team
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
"""Testings for heuristic module."""

import numpy as np
from numpy.testing import assert_equal, assert_raises

from procrustes.heuristic import optimal_heuristic
from procrustes.utils import error


def test_optimal_heuristic():
    r"""Test optimal_heuristic with manually set up example."""
    # test whether it works correctly
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
    error_old = error(arr_a, arr_b, perm_guess, perm_guess)
    perm, kopt_error = optimal_heuristic(perm_guess, arr_a, arr_b, error_old, 3)
    assert_equal(perm, perm_exact)
    assert kopt_error == 0
    # test the error exceptions
    assert_raises(ValueError, optimal_heuristic, perm_guess, arr_a, arr_b, error_old, 1)
