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
"""Heuristic Module."""
import itertools as it
from copy import deepcopy

import numpy as np

from procrustes.utils import error


def optimal_heuristic(perm, array_a, array_b, ref_error, k_opt=3):
    r"""
    K-opt heuristic to improve the accuracy.

    Perform k-opt local search with every possible valid combination of the swapping mechanism.

    Parameters
    ----------
    perm : np.ndarray
        The permutation array which remains to be processed with k-opt local search.
    array_a : np.ndarray
        The array to be permuted.
    array_b : np.ndarray
        The reference array.
    ref_error : float
        The reference error value.
    k_opt : int, optional
        Order of local search. Default=3.

    Returns
    -------
    perm : ndarray
        The permutation array after optimal heuristic search.
    kopt_error : float
        The error distance of two arrays with the updated permutation array.
    """
    if k_opt < 2:
        raise ValueError("K_opt value must be a integer greater than 2.")
    num_row = perm.shape[0]
    kopt_error = ref_error
    # all the possible row-wise permutations
    for comb in it.combinations(np.arange(num_row), r=k_opt):
        for comb_perm in it.permutations(comb, r=k_opt):
            if comb_perm != comb:
                perm_kopt = deepcopy(perm)
                perm_kopt[comb, :] = perm_kopt[comb_perm, :]
                e_kopt_new = error(array_a, array_b, perm_kopt, perm_kopt)
                if e_kopt_new < kopt_error:
                    perm = perm_kopt
                    kopt_error = e_kopt_new
                    if kopt_error == 0:
                        break
    return perm, kopt_error
