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


def kopt_heuristic_single(array_a, array_b, ref_error,
                          perm=None, kopt_k=3, kopt_tol=1.e-8):
    r"""K-opt heuristic to improve the accuracy for two-sided permutation with one transformation.

    Perform k-opt local search with every possible valid combination of the swapping mechanism.

    Parameters
    ----------
    array_a : ndarray
        The array to be permuted.
    array_b : ndarray
        The reference array.
    ref_error : float
        The reference error value.
    perm : ndarray, optional
        The permutation array which remains to be processed with k-opt local search. Default is the
        identity matrix with the same shape of array_a.
    kopt_k : int, optional
        Defines the oder of k-opt heuristic local search. For example, kopt_k=3 leads to a local
        search of 3 items and kopt_k=2 only searches for two items locally. Default=3.
    kopt_tol : float, optional
        Tolerance value to check if k-opt heuristic converges. Default=1.e-8.

    Returns
    -------
    perm : ndarray
        The permutation array after optimal heuristic search.
    kopt_error : float
        The error distance of two arrays with the updated permutation array.
    """
    if kopt_k < 2:
        raise ValueError("Kopt_k value must be a integer greater than 2.")
    # if perm is not specified, use the identity matrix as default
    if perm is None:
        perm = np.identity(np.shape(array_a)[0])
    num_row = perm.shape[0]
    kopt_error = ref_error
    # all the possible row-wise permutations
    for comb in it.combinations(np.arange(num_row), r=kopt_k):
        for comb_perm in it.permutations(comb, r=kopt_k):
            if comb_perm != comb:
                perm_kopt = deepcopy(perm)
                perm_kopt[comb, :] = perm_kopt[comb_perm, :]
                e_kopt_new = compute_error(array_a, array_b, perm_kopt, perm_kopt)
                if e_kopt_new < kopt_error:
                    perm = perm_kopt
                    kopt_error = e_kopt_new
                    if kopt_error <= kopt_tol:
                        break
    return perm, kopt_error


def kopt_heuristic_double(array_m, array_n, ref_error,
                          perm_p=None, perm_q=None,
                          kopt_k=3, kopt_tol=1.e-8):
    r"""
    K-opt kopt for regular two-sided permutation Procrustes to improve the accuracy.

    Perform k-opt local search with every possible valid combination of the swapping mechanism for
    regular 2-sided permutation Procrustes.

    Parameters
    ----------
    array_m : ndarray
        The array to be permuted.
    array_n : ndarray
        The reference array.
    ref_error : float
        The reference error value.
    perm_p : ndarray, optional
        The left permutation array which remains to be processed with k-opt local search. Default
        is the identity matrix with the same shape of array_m.
    perm_q : ndarray, optional
        The right permutation array which remains to be processed with k-opt local search. Default
        is the identity matrix with the same shape of array_m.
    kopt_k : int, optional
        Defines the oder of k-opt heuristic local search. For example, kopt_k=3 leads to a local
        search of 3 items and kopt_k=2 only searches for two items locally. Default=3.
    kopt_tol : float, optional
        Tolerance value to check if k-opt heuristic converges. Default=1.e-8.

    Returns
    -------
    perm_kopt_p : ndarray
        The left permutation array after optimal heuristic search.
    perm_kopt_q : ndarray
        The right permutation array after optimal heuristic search.
    kopt_error : float
        The error distance of two arrays with the updated permutation array.
    """
    if kopt_k < 2:
        raise ValueError("Kopt_k value must be a integer greater than 2.")
    # if perm_p is not specified, use the identity matrix as default
    if perm_p is None:
        perm_p = np.identity(np.shape(array_m)[0])
    # if perm_p is not specified, use the identity matrix as default
    if perm_q is None:
        perm_q = np.identity(np.shape(array_m)[0])

    num_row_left = perm_p.shape[0]
    num_row_right = perm_q.shape[0]
    kopt_error = ref_error
    # the left hand side permutation
    # pylint: disable=too-many-nested-blocks
    for comb_left in it.combinations(np.arange(num_row_left), r=kopt_k):
        for comb_perm_left in it.permutations(comb_left, r=kopt_k):
            if comb_perm_left != comb_left:
                perm_kopt_left = deepcopy(perm_p)
                # the right hand side permutation
                for comb_right in it.combinations(np.arange(num_row_right), r=kopt_k):
                    for comb_perm_right in it.permutations(comb_right, r=kopt_k):
                        if comb_perm_right != comb_right:
                            perm_kopt_right = deepcopy(perm_q)
                            perm_kopt_right[comb_right, :] = perm_kopt_right[comb_perm_right, :]
                            e_kopt_new_right = compute_error(array_n, array_m, perm_p.T,
                                                             perm_kopt_right)
                            if e_kopt_new_right < kopt_error:
                                perm_q = perm_kopt_right
                                kopt_error = e_kopt_new_right
                                if kopt_error <= kopt_tol:
                                    break

                perm_kopt_left[comb_left, :] = perm_kopt_left[comb_perm_left, :]
                e_kopt_new_left = compute_error(array_n, array_m, perm_kopt_left.T, perm_q)
                if e_kopt_new_left < kopt_error:
                    perm_p = perm_kopt_left
                    kopt_error = e_kopt_new_left
                    if kopt_error <= kopt_tol:
                        break

    return perm_p, perm_q, kopt_error
