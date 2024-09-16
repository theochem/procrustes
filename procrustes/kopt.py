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
"""K-opt (Greedy) Heuristic Module."""

import itertools as it
from typing import Callable, Tuple

import numpy as np

__all__ = [
    "kopt_heuristic_single",
    "kopt_heuristic_double",
]


def kopt_heuristic_single(
    fun: Callable, p0: np.ndarray, k: int = 3, tol: float = 1.0e-8
) -> Tuple[np.ndarray, float]:
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
    tol : float, optional
        When value of the objective function is less than given tolerance, the algorithm stops.

    Returns
    -------
    p_opt : ndarray
        The locally-optimal permutation matrix :math:`\mathbf{P}` (i.e., solution).
    f_opt : float
        The locally-optimal value of objective function given by :math:`\text{fun(p_opt)}`.

    """
    # pylint: disable=too-many-nested-blocks

    # check whether p0 is a valid permutation matrix
    if p0.ndim != 2 or p0.shape[0] != p0.shape[1]:
        raise ValueError(f"Argument p0 should be a square array. Given p0 shape={p0.shape}")
    if not np.all(np.logical_or(p0 == 0, p0 == 1)):
        raise ValueError("Elements of permutation matrix p0 can only be 0 or 1.")
    if np.all(np.sum(p0, axis=0) != 1) or np.all(np.sum(p0, axis=1) != 1):
        raise ValueError("Sum over rows or columns of p0 matrix isn't equal 1.")

    # check k
    if k < 2 or not isinstance(k, (int, np.integer)):
        raise ValueError(f"Argument k={k} must be a integer greater than 1. Given type {type(k)}")
    if k > p0.shape[0]:
        raise ValueError(f"Argument k={k} is not smaller than {p0.shape[0]} (number of p0 rows).")

    # compute initial value of the objective function & assign best P matrix
    f_opt = fun(p0)
    p_opt = np.copy(p0)
    # swap rows and columns until the permutation matrix is not improved
    search = True
    while search:
        search = False
        # make sure p0 guess is the best permutation matrix found thus far
        p0 = np.copy(p_opt)
        for perm in it.permutations(np.arange(p0.shape[0]), r=int(k)):
            comb = tuple(sorted(perm))
            if perm != comb:
                # row-swap P matrix & compute objective function
                perm_p = np.copy(p0)
                perm_p[:, comb] = perm_p[:, perm]
                # compute objective function for permuted P matrix & compare
                perm_f = fun(perm_p)
                if perm_f < f_opt:
                    p_opt, f_opt = perm_p, perm_f
                    # set search=True to keep permuting the new p_opt unless this is already an
                    # exhaustive search (i.e., k equals number of rows of p matrix)
                    search = bool(k < p0.shape[0])
                    # check whether perfect permutation matrix is found
                    # TODO: smarter threshold based on norm of matrix
                    if f_opt <= tol:
                        return p_opt, f_opt
    return p_opt, f_opt


def kopt_heuristic_double(
    fun: Callable, p1: np.ndarray, p2: np.ndarray, k: int = 3, tol: float = 1.0e-8
) -> Tuple[np.ndarray, np.ndarray, float]:
    r"""Find locally-optimal permutation matrices using the k-opt (greedy) heuristic.

    .. math::
        \underbrace{\text{arg min}}_{
        \left\{ {\mathbf{P}_1, \mathbf{P}_2} \left|
        {{[\mathbf{P}_1]_{ij} \in \{0, 1\} \atop [\mathbf{P}_2]_{ij} \in \{0, 1\}} \atop
        {\sum_{i=1}^m [\mathbf{P}_1]_{ij} = \sum_{j=1}^m [\mathbf{P}_1]_{ij} = 1 \atop
        \sum_{i=1}^n [\mathbf{P}_2]_{ij} = \sum_{j=1}^n [\mathbf{P}_2]_{ij} = 1}} \right. \right\}}
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
    tol : float, optional
        When value of the objective function is less than given tolerance, the algorithm stops.

    Returns
    -------
    p1_opt : ndarray
        The locally-optimal permutation matrix :math:`\mathbf{P}_1`.
    p2_opt : ndarray
        The locally-optimal permutation matrix :math:`\mathbf{P}_2`.
    f_opt : float
        The locally-optimal value of objective function given by :math:`\text{fun(p1_opt, p2_opt)}`.

    """
    # pylint: disable=too-many-nested-blocks

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
    if k < 2 or not isinstance(k, (int, np.integer)):
        raise ValueError(f"Argument k={k} must be a integer greater than 1. Give type {type(k)}")
    if k > max(p1.shape[0], p2.shape[0]):
        raise ValueError(f"Argument k={k} is not smaller than {max(p1.shape[0], p2.shape[0])}.")

    # compute initial value of the objective function & assign best P1 & P2 matrices
    f_opt = fun(p1, p2)
    p1_opt, p2_opt = np.copy(p1), np.copy(p2)

    # swap rows and columns until the permutation matrix is not improved
    search = True
    while search:
        search = False
        # make sure p1 & p2 guesses are the best permutation matrices found thus far
        p1, p2 = np.copy(p1_opt), np.copy(p2_opt)
        for perm1 in it.permutations(np.arange(p1.shape[0]), r=int(min(k, p1.shape[0]))):
            comb1 = tuple(sorted(perm1))
            for perm2 in it.permutations(np.arange(p2.shape[0]), r=int(min(k, p2.shape[0]))):
                comb2 = tuple(sorted(perm2))
                if not (perm1 == comb1 and perm2 == comb2):
                    # permute rows of matrix P1
                    perm_p1 = np.copy(p1)
                    perm_p1[comb1, :] = perm_p1[perm1, :]
                    # permute rows of matrix P2
                    perm_p2 = np.copy(p2)
                    perm_p2[comb2, :] = perm_p2[perm2, :]
                    # compute objective function for permuted P matrix & compare
                    perm_f = fun(perm_p1, perm_p2)
                    if perm_f < f_opt:
                        p1_opt, p2_opt, f_opt = perm_p1, perm_p2, perm_f
                        # set search=True to keep permuting the new p1_opt & p2_opt unless this
                        # is already an exhaustive search
                        search = bool(k < max(p1.shape[0], p2.shape[0]))
                        # check whether perfect permutation matrix is found
                        # TODO: smarter threshold based on norm of matrix
                        if f_opt <= tol:
                            return p1_opt, p2_opt, f_opt
    return p1_opt, p2_opt, f_opt
