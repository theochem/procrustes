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
"""K-opt (Greedy) Heuristic Module."""


import itertools as it

import numpy as np
from procrustes.utils import compute_error


__all__ = [
    "kopt_heuristic_single",
    "kopt_heuristic_double",
]


def kopt_heuristic_single(fun, p0, k=3):
    r"""Find a locally-optimal permutation matrix using the k-opt (greedy) heuristic.

    .. math::
       \underbrace{\text{min}}_{\left\{\mathbf{P} \left| {[\mathbf{P}]_{ij} \in \{0, 1\}
       \atop \sum_{i=1}^n [\mathbf{P}]_{ij} = \sum_{j=1}^n [\mathbf{P}]_{ij} = 1} \right. \right\}}
       f(\mathbf{P})

    All possible 2-, ..., k-fold column-permutations of the initial permutation matrix are tried to
    identify one which gives a lower value of objective function :math:`f`.
    Starting from this updated permutation matrix, the process is repeated until no further k-fold
    column-reordering of a given permutation matrix lower the objective function.

    Parameters
    ----------
    fun : callable
        The objective function :math:`f` to be minimized.
    p0 : ndarray
        The 2D-array permutation matrix representing the initial guess for :math:`\mathbf{P}`.
    k : int, optional
        The order of the permutation. For example, `k=3` swaps all possible 3-permutations.

    Returns
    -------
    p : ndarray
        The locally-optimal permutation matrix (i.e., solution).
    fun : float
        The locally-optimal value of the objective function.

    """
    # pylint: disable=too-many-nested-blocks
    if k < 2 or not isinstance(k, int):
        raise ValueError(f"Argument k must be a integer greater than 2. Given k={k}")
    if p0.ndim != 2 or p0.shape[0] != p0.shape[1]:
        raise ValueError(f"Argument p0 should be a square array. Given p0 shape={p0.shape}")
    if k > p0.shape[0]:
        raise ValueError(f"Argument k={k} is not smaller than {p0.shape[0]} (number of p0 rows).")

    # check whether p0 is a valid permutation matrix
    if not np.all(np.logical_or(p0 == 0, p0 == 1)):
        raise ValueError("Elements of permutation matrix p0 can only be 0 or 1.")
    if np.all(np.sum(p0, axis=0) != 1) or np.all(np.sum(p0, axis=1) != 1):
        raise ValueError("Sum over rows or columns of p0 matrix isn't equal 1.")

    # compute initial value of the objective function
    error = fun(p0)
    # swap rows and columns until the permutation matrix is not improved
    search = True
    while search:
        search = False
        for perm in it.permutations(np.arange(p0.shape[0]), r=k):
            comb = tuple(sorted(perm))
            if perm != comb:
                # row-swap P matrix & compute objective function
                perm_p = np.copy(p0)
                perm_p[:, comb] = perm_p[:, perm]
                perm_error = fun(perm_p)
                if perm_error < error:
                    search = True
                    p0, error = perm_p, perm_error
                    # check whether perfect permutation matrix is found
                    # TODO: smarter threshold based on norm of matrix
                    if error <= 1.0e-8:
                        return p0, error
    return p0, error


def kopt_heuristic_double(fun, p1=None, p2=None, k=3):
    r"""Find a locally-optimal two-sided permutation matrices using the k-opt (greedy) heuristic.

    .. math::
        \underbrace{\text{arg min}}_{\left\{ {\mathbf{P}_1, \mathbf{P}_2} \left|
        {[\mathbf{P}_1]_{ij} \in \{0, 1\} \text{and} [\mathbf{P}_2]_{ij} \in \{0, 1\} \atop
         \sum_{i=1}^m [\mathbf{P}_1]_{ij} = \sum_{j=1}^m [\mathbf{P}_1]_{ij} = 1 \atop
         \sum_{i=1}^n [\mathbf{P}_2]_{ij} = \sum_{j=1}^n [\mathbf{P}_2]_{ij} = 1} \right. \right\}}
         f(\mathbf{P}_1, \mathbf{P}_2)

    All possible 2-, ..., k-fold permutations of the initial permutation matrices are tried to
    identify ones which give a lower value of objective function :math:`f`.
    This corresponds to row-swaps for :math:`\mathbf{ P}_1` and column-swaps for :math:`\mathbf{
    P}_2`. Starting from these updated permutation matrices, the process is repeated until no
    further k-fold reordering of either permutation matrix lower the objective function.

    Parameters
    ----------
    fun : callable
        The objective function :math:`f` to be minimized.
    p1 : ndarray
        The 2D-array permutation matrix representing the initial guess for :math:`\mathbf{P}_1`.
    p2 : ndarray
        The 2D-array permutation matrix representing the initial guess for :math:`\mathbf{P}_2`.
    k : int, optional
        The order of the permutation. For example, ``k=3`` swaps all possible 3-permutations.

    Returns
    -------
    perm_p1 : ndarray
        The locally-optimal left-hand-side permutation matrix.
    perm_p2 : ndarray
        The locally-optimal right-hand-side permutation matrix.
    fun : float
        The locally-optimal value of the objective function.

    """
    # check whether p1 & p2 are square arrays
    if p1.ndim != 2 or p1.shape[0] != p1.shape[1]:
        raise ValueError(f"Argument p1 should be a square array. Given p1 shape={p1.shape}")
    if p2.ndim != 2 or p2.shape[0] != p2.shape[1]:
        raise ValueError(f"Argument p2 should be a square array. Given p2 shape={p2.shape}")

    # check whether p1 & p2 are valid permutation matrices
    if not np.all(np.logical_or(p1 == 0, p1 == 1)):
        raise ValueError("Elements of permutation matrix p1 can only be 0 or 1.")
    if not np.all(np.logical_or(p2 == 0, p2 == 1)):
        raise ValueError("Elements of permutation matrix p2 can only be 0 or 1.")

    if np.all(np.sum(p1, axis=0) != 1) or np.all(np.sum(p1, axis=1) != 1):
        raise ValueError("Sum over rows or columns of p1 matrix isn't equal 1.")
    if np.all(np.sum(p2, axis=0) != 1) or np.all(np.sum(p2, axis=1) != 1):
        raise ValueError("Sum over rows or columns of p2 matrix isn't equal 1.")

    # check k
    if k < 2 or not isinstance(k, int):
        raise ValueError(f"Argument k must be a integer greater than 2. Given k={k}")
    if k > min(p1.shape[0], p2.shape[0]):
        raise ValueError(f"Argument k={k} is not smaller than {min(p1.shape[0], p2.shape[0])}.")

    # compute 2-sided permutation error of the initial P & Q matrices
    error = fun(p1, p2)
    # pylint: disable=too-many-nested-blocks

    for perm1 in it.permutations(np.arange(p1.shape[0]), r=k):
        comb1 = tuple(sorted(perm1))
        for perm2 in it.permutations(np.arange(p2.shape[0]), r=k):
            comb2 = tuple(sorted(perm2))
            if not (perm1 == comb1 and perm2 == comb2):
                # permute rows of matrix P1
                perm_p1 = np.copy(p1)
                perm_p1[comb1, :] = perm_p1[perm1, :]
                # permute rows of matrix P2
                perm_p2 = np.copy(p2)
                perm_p2[comb2, :] = perm_p2[perm2, :]
                # compute error with new matrices & compare
                perm_error = fun(perm_p1, perm_p2)
                if perm_error < error:
                    p1, p2, error = perm_p1, perm_p2, perm_error
                    # check whether perfect permutation matrix is found
                    # TODO: smarter threshold based on norm of matrix
                    if error <= 1.0e-8:
                        break
    return p1, p2, error
