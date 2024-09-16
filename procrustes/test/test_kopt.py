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
"""Test Module for Kopt."""

import numpy as np
import pytest
from numpy.testing import assert_equal, assert_raises

from procrustes.kopt import kopt_heuristic_double, kopt_heuristic_single
from procrustes.utils import compute_error


def test_kopt_heuristic_single_raises():
    r"""Test k-opt heuristic single search algorithm raises."""
    # check raises for k
    assert_raises(ValueError, kopt_heuristic_single, np.sum, np.eye(2), 1)
    assert_raises(ValueError, kopt_heuristic_single, np.sum, np.eye(3), -2)
    assert_raises(ValueError, kopt_heuristic_single, np.sum, np.eye(5), 6)
    # check raises for p0
    assert_raises(ValueError, kopt_heuristic_single, np.sum, np.ones(3), 2)
    assert_raises(ValueError, kopt_heuristic_single, np.sum, np.ones((2, 3)), 2)
    assert_raises(ValueError, kopt_heuristic_single, np.sum, np.ones((4, 4)), 2)
    assert_raises(ValueError, kopt_heuristic_single, np.sum, np.zeros((5, 5)), 2)
    assert_raises(ValueError, kopt_heuristic_single, np.sum, np.eye(6) + 0.1, 2)


@pytest.mark.parametrize("m", np.random.randint(5, 15, 2))
def test_kopt_heuristic_single_identity(m):
    r"""Test k-opt heuristic single search algorithm with identity permutation."""
    # create a random matrix A and random permutation of identity matrix
    a = np.random.uniform(-2.0, 2.0, (m, m))
    p0 = np.eye(m)
    # find and check permutation for when B=A with guess p0=I
    perm, error = kopt_heuristic_single(lambda x: compute_error(a, a, x, x.T), p0, k=2)
    assert_equal(perm, np.eye(m))
    assert_equal(error, 0.0)
    # find and check permutation for when B=A with guess p0 being swapped I
    p0[[m - 2, -1]] = p0[[-1, m - 2]]
    perm, error = kopt_heuristic_single(lambda x: compute_error(a, a, x, x.T), p0, k=2)
    assert_equal(perm, np.eye(m))
    assert_equal(error, 0.0)


@pytest.mark.parametrize("m", np.random.randint(5, 15, 2))
def test_kopt_heuristic_single_k_permutations(m):
    r"""Test k-opt heuristic single search algorithm going upto k permutations."""
    # create a random matrix A
    a = np.random.uniform(-10.0, 10.0, (m, m))
    # create permutation matrix by swapping rows m-3 & -1 of identity matrix (this makes sures that
    # heuristic algorithm only finds the solution towards the end of its search)
    p = np.eye(m)
    p[[m - 3, -1]] = p[[-1, m - 3]]
    # compute B = P^T A P
    b = np.linalg.multi_dot([p.T, a, p])
    # find and check permutation
    perm, error = kopt_heuristic_single(lambda x: compute_error(a, b, x, x.T), np.eye(m), k=2)
    assert_equal(perm, p)
    assert_equal(error, 0.0)


@pytest.mark.parametrize("m", np.random.randint(2, 10, 3))
def test_kopt_heuristic_single_all_permutations(m):
    r"""Test k-opt heuristic single search algorithm going through all permutations."""
    # create a random matrix A and random permutation of identity matrix
    a = np.random.uniform(-10.0, 10.0, (m, m))
    p = np.random.permutation(np.eye(m))
    # compute B = P^T A P
    b = np.linalg.multi_dot([p.T, a, p])
    # find and check permutation
    perm, error = kopt_heuristic_single(lambda x: compute_error(a, b, x, x.T), np.eye(m), m)
    assert_equal(perm, p)
    assert_equal(error, 0.0)


def test_kopt_heuristic_double_raises():
    r"""Test k-opt heuristic double search algorithm raises."""
    # check raises for k
    assert_raises(ValueError, kopt_heuristic_double, np.sum, np.eye(2), np.eye(2), 1)
    assert_raises(ValueError, kopt_heuristic_double, np.sum, np.eye(3), np.eye(2), -2.5)
    assert_raises(ValueError, kopt_heuristic_double, np.sum, np.eye(5), np.eye(2), 6)
    # check raises for p0
    assert_raises(ValueError, kopt_heuristic_double, np.sum, np.ones(4), np.eye(3), 2)
    assert_raises(ValueError, kopt_heuristic_double, np.sum, np.eye(3), np.ones(4), 2)
    assert_raises(ValueError, kopt_heuristic_double, np.sum, np.ones((2, 3)), np.eye(2), 2)
    assert_raises(ValueError, kopt_heuristic_double, np.sum, np.eye(2), np.ones((2, 3)), 2)
    assert_raises(ValueError, kopt_heuristic_double, np.sum, np.ones((4, 4)), np.eye(4), 3)
    assert_raises(ValueError, kopt_heuristic_double, np.sum, np.eye(4), np.ones((4, 4)), 3)
    assert_raises(ValueError, kopt_heuristic_double, np.sum, np.zeros((5, 5)), np.eye(6), 4)
    assert_raises(ValueError, kopt_heuristic_double, np.sum, np.eye(6), np.zeros((5, 5)), 4)
    assert_raises(ValueError, kopt_heuristic_double, np.sum, np.eye(6) + 0.1, np.eye(5), 3)
    assert_raises(ValueError, kopt_heuristic_double, np.sum, np.eye(5), np.eye(6) + 0.1, 3)


@pytest.mark.parametrize("m, n", np.random.randint(5, 7, (2, 2)))
def test_kopt_heuristic_double_identity(m, n):
    r"""Test k-opt heuristic double search algorithm with identity permutation."""
    # create a random matrix A and random permutation of identity matrix
    a = np.random.uniform(-6.0, 6.0, (m, n))
    p1, p2 = np.eye(m), np.eye(n)
    # find and check permutation for when B=A with guesses p1=I & p2=I
    perm1, perm2, error = kopt_heuristic_double(lambda x, y: compute_error(a, a, y, x.T), p1, p2, 2)
    assert_equal(perm1, p1)
    assert_equal(perm2, p2)
    assert_equal(error, 0.0)
    # find and check permutation for when B=A with guesses p1 & p2 being swapped I
    p1[[m - 4, -1]] = p1[[-1, m - 4]]
    p2[[0, -1]] = p2[[-1, 0]]
    perm1, perm2, error = kopt_heuristic_double(lambda x, y: compute_error(a, a, y, x.T), p1, p2, 2)
    assert_equal(perm1, np.eye(m))
    assert_equal(perm2, np.eye(n))
    assert_equal(error, 0.0)


@pytest.mark.parametrize("m, n", np.random.randint(5, 10, (2, 2)))
def test_kopt_heuristic_double_k_permutations(m, n):
    r"""Test k-opt heuristic double search algorithm going upto k permutations."""
    # create a random matrix A
    a = np.random.uniform(-7.0, 7.0, (m, n))
    # create permutation matrix by swapping rows m-3 & -1 of identity matrix (this makes sures that
    # heuristic algorithm only finds the solution towards the end of its search)
    p1 = np.eye(m)
    p1[[m - 2, -1]] = p1[[-1, m - 2]]
    p2 = np.eye(n)
    p2[[n - 1, -1]] = p2[[-1, n - 1]]
    # compute B = P^T A P
    b = np.linalg.multi_dot([p1.T, a, p2])
    # find and check permutation
    perm1, perm2, error = kopt_heuristic_double(
        lambda x, y: compute_error(a, b, y, x.T), np.eye(m), np.eye(n), k=2
    )
    assert_equal(perm1, p1)
    assert_equal(perm2, p2)
    assert_equal(error, 0.0)


@pytest.mark.parametrize("m, n", np.random.randint(2, 7, (3, 2)))
def test_kopt_heuristic_double_all_permutations(m, n):
    r"""Test k-opt heuristic double search algorithm going through all permutations."""
    # create a random matrix A and random permutation of identity matrix
    a = np.random.uniform(-5.0, 5.0, (m, n))
    p1 = np.random.permutation(np.eye(m))
    p2 = np.random.permutation(np.eye(n))
    # compute B = P1^T A P2
    b = np.linalg.multi_dot([p1.T, a, p2])
    # find and check permutations
    perm1, perm2, error = kopt_heuristic_double(
        lambda x, y: compute_error(a, b, y, x.T), np.eye(m), np.eye(n), max(n, m)
    )
    assert_equal(perm1, p1)
    assert_equal(perm2, p2)
    assert_equal(error, 0.0)
