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
"""Kopt Module."""

from copy import deepcopy
import itertools as it

import numpy as np
from procrustes.utils import compute_error

__all__ = [
    "kopt_heuristic_single",
    "kopt_heuristic_double",
]


def kopt_heuristic_single(a, b, p=None, k=3):
    r"""Compute locally optimal permutation matrix and its error using k-opt heuristic.

    Perform k-opt local search with every possible valid combination of the swapping mechanism.

    Parameters
    ----------
    a : ndarray
        The 2D-array :math:`\mathbf{A}` which is going to be transformed.
    b : ndarray
        The 2D-array :math:`\mathbf{B}` representing the reference matrix.
    p : ndarray, optional
        The 2D-array :math:`\mathbf{P}` representing the initial permutation matrix. If `None`, the
        identity matrix is used.
    k : int, optional
        The order of the permutation. For example, `k=3` swaps all possible 3-permutations of the
        given p matrix.

    Returns
    -------
    perm : ndarray
        The permutation array after optimal heuristic search.
    error : float
        The error distance of two arrays with the updated permutation array.
    """
    if k < 2 or not isinstance(k, int):
        raise ValueError(f"Argument k must be a integer greater than 2. Given k={k}")
    # assign p to be an identity array, if not specified
    if p is None:
        p = np.identity(np.shape(a)[0])
    # compute 2-sided permutation error of the initial p matrix
    error = compute_error(a, b, p, p)
    # pylint: disable=too-many-nested-blocks
    # swap rows and columns until the permutation matrix is not improved
    search = True
    while search:
        search = False
        for comb in it.combinations(np.arange(p.shape[0]), r=k):
            for comb_perm in it.permutations(comb, r=k):
                if comb_perm != comb:
                    p_new = deepcopy(p)
                    p_new[comb, :] = p_new[comb_perm, :]
                    error_new = compute_error(a, b, p_new, p_new)
                    if error_new < error:
                        search = True
                        p, error = p_new, error_new
                        # check whether perfect permutation matrix is found
                        # TODO: smarter threshold based on norm of matrix
                        if error <= 1.0e-8:
                            return p, error
    return p, error


def kopt_heuristic_double(a, b, p=None, q=None, k=3):
    r"""
    K-opt kopt for regular two-sided permutation Procrustes to improve the accuracy.

    Perform k-opt local search with every possible valid combination of the swapping mechanism for
    regular 2-sided permutation Procrustes.

    Parameters
    ----------
    a : ndarray
        The 2D-array :math:`\mathbf{A}` which is going to be transformed.
    b : ndarray
        The 2D-array :math:`\mathbf{B}` representing the reference matrix.
    p : ndarray, optional
        The 2D-array :math:`\mathbf{P}` representing the initial right-hand-side permutation matrix.
        If `None`, the identity matrix is used.
    q : ndarray, optional
        The 2D-array :math:`\mathbf{Q}` representing the initial left-hand-side permutation matrix.
        If `None`, the identity matrix is used.
    k : int, optional
        The order of the permutation. For example, `k=3` swaps all possible 3-permutations of the
        given p matrix.

    Returns
    -------
    perm_p : ndarray
        The right-hand-side permutation matrix after optimal heuristic search.
    perm_q : ndarray
        The left-hand-side permutation matrix after optimal heuristic search.
    error : float
        The error distance of two arrays with the updated permutation array.
    """
    if k < 2 or not isinstance(k, int):
        raise ValueError(f"Argument k must be a integer greater than 2. Given k={k}")
    # assign p & q to be an identity arrays, if not specified
    if p is None:
        p = np.identity(np.shape(a)[0])
    if q is None:
        q = np.identity(np.shape(a)[0])

    # compute 2-sided permutation error of the initial P & Q matrices
    error = compute_error(a, b, p, q)
    # pylint: disable=too-many-nested-blocks
    num_row_left = p.shape[0]
    num_row_right = q.shape[0]

    for comb_p in it.combinations(np.arange(num_row_left), r=k):
        for perm_p in it.permutations(comb_p, r=k):
            for comb_q in it.combinations(np.arange(num_row_right), r=k):
                for perm_q in it.permutations(comb_q, r=k):
                    # permute rows of matrix P
                    p_new = deepcopy(p)
                    p_new[comb_p, :] = p_new[perm_p, :]
                    # permute rows of matrix Q
                    q_new = deepcopy(q)
                    q_new[comb_q, :] = q_new[perm_q, :]
                    # compute error with new matrices & compare
                    error_new = compute_error(b, a, p_new, q_new)
                    if error_new < error:
                        p, q, error = p_new, q_new, error_new
                        # check whether perfect permutation matrix is found
                        # TODO: smarter threshold based on norm of matrix
                        if error <= 1.0e-8:
                            break
    return p, q, error
