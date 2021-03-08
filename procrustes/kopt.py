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


def kopt_heuristic_single(a, b, p=None, k=3):
    r"""Find a locally-optimal one-sided permutation matrix using the k-opt (greedy) heuristic.

    .. math::
       \underbrace{\text{min}}_{\left\{\mathbf{P} \left| {[\mathbf{P}]_{ij} \in \{0, 1\}
       \atop \sum_{i=1}^n [\mathbf{P}]_{ij} = \sum_{j=1}^n [\mathbf{P}]_{ij} = 1} \right. \right\}}
       \|\mathbf{P}^\dagger \mathbf{A} \mathbf{P} - \mathbf{B}\|_{F}^2\\

    All possible 2-, ..., k-fold column-permutations of the initial permutation matrix are tried to
    identify one which gives a lower error. Starting from this updated permutation matrix, the
    process is repeated until no further k-fold column-reordering of a given permutation matrix
    lower the error.

    Parameters
    ----------
    a : ndarray
        The 2D-array :math:`\mathbf{A}` which is going to be transformed.
    b : ndarray
        The 2D-array :math:`\mathbf{B}` representing the reference matrix.
    p : ndarray, optional
        The 2D-array :math:`\mathbf{P}` representing the initial permutation matrix. If ``None``,
        the identity matrix is used.
    k : int, optional
        The order of the permutation. For example, `k=3` swaps all possible 3-permutations.

    Returns
    -------
    perm_p : ndarray
        The locally-optimal permutation matrix.
    error : float
        The locally-optimal error.

    """
    if k < 2 or not isinstance(k, int):
        raise ValueError(f"Argument k must be a integer greater than 2. Given k={k}")
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"Argument a should be a square array. Given shape={a.shape}")
    if a.shape != b.shape:
        raise ValueError(f"Argument b should have same shape as a. Given {b.shape} != {a.shape}")

    # assign p to be an identity array, if not specified
    n = a.shape[0]
    if p is None:
        p = np.identity(n)
    # compute 2-sided permutation error of the initial p matrix
    error = compute_error(a, b, p, p)
    # pylint: disable=too-many-nested-blocks
    # swap rows and columns until the permutation matrix is not improved
    search = True
    while search:
        search = False
        for perm in it.permutations(np.arange(n), r=k):
            comb = tuple(sorted(perm))
            if perm != comb:
                # row-swap P matrix & compute error
                perm_p = np.copy(p)
                perm_p[:, comb] = perm_p[:, perm]
                perm_error = compute_error(a, b, perm_p, perm_p)
                if perm_error < error:
                    search = True
                    p, error = perm_p, perm_error
                    # check whether perfect permutation matrix is found
                    # TODO: smarter threshold based on norm of matrix
                    if error <= 1.0e-8:
                        return p, error
    return p, error


def kopt_heuristic_double(a, b, p1=None, p2=None, k=3):
    r"""Find a locally-optimal two-sided permutation matrices using the k-opt (greedy) heuristic.

    .. math::
        &\underbrace{\text{min}}_{\left\{\mathbf{P}_1,\mathbf{P}_2 \left|
        {[\mathbf{P}_1]_{ij} \in \{0, 1\} \atop \sum_{i=1}^n [\mathbf{P}_1]_{ij} = \sum_{j=1}^n [
        \mathbf{P}_1]_{ij} = 1} \atop {[\mathbf{P}_2]_{ij} \in \{0, 1\} \atop \sum_{i=1}^n [
        \mathbf{P}_2]_{ij} = \sum_{j=1}^n [\mathbf{P}_2]_{ij} = 1} \right. \right\}}
            \|\mathbf{P}_1 \mathbf{A} \mathbf{P}_2 - \mathbf{B}\|_{F}^2\\

    All possible 2-, ..., k-fold permutations of the initial permutation matrices are tried to
    identify ones which give a lower error. This corresponds to row-swaps for :math:`\mathbf{
    P}_1` and column-swaps for :math:`\mathbf{P}_2`. Starting from these updated permutation
    matrices, the process is repeated until no further k-fold reordering of either permutation
    matrix lower the error.

    Parameters
    ----------
    a : ndarray
        The 2D-array :math:`\mathbf{A}` which is going to be transformed.
    b : ndarray
        The 2D-array :math:`\mathbf{B}` representing the reference matrix.
    p1 : ndarray, optional
        The 2D-array :math:`\mathbf{P}_1` representing the initial left-hand-side permutation
        matrix. If ``None``, the identity matrix is used.
    p2 : ndarray, optional
        The 2D-array :math:`\mathbf{P}_2` representing the initial right-hand-side permutation
        matrix. If ``None``, the identity matrix is used.
    k : int, optional
        The order of the permutation. For example, ``k=3`` swaps all possible 3-permutations.

    Returns
    -------
    perm_p1 : ndarray
        The locally-optimal left-hand-side permutation matrix.
    perm_p2 : ndarray
        The locally-optimal right-hand-side permutation matrix.
    error : float
        The locally-optimal error.

    """
    if k < 2 or not isinstance(k, int):
        raise ValueError(f"Argument k must be a integer greater than 2. Given k={k}")
    if a.shape != b.shape:
        raise ValueError(f"Argument b should have same shape as a. Given {b.shape} != {a.shape}")

    # assign p1 & p2 to be an identity arrays, if not specified
    n, m = a.shape
    if p1 is None:
        p1 = np.identity(n)
    if p2 is None:
        p2 = np.identity(m)

    # compute 2-sided permutation error of the initial P & Q matrices
    error = compute_error(a, b, p1, p2)
    # pylint: disable=too-many-nested-blocks

    for perm1 in it.permutations(np.arange(n), r=k):
        comb1 = tuple(sorted(perm1))
        for perm2 in it.permutations(np.arange(m), r=k):
            comb2 = tuple(sorted(perm2))
            if not (perm1 == comb1 and perm2 == comb2):
                # permute rows of matrix P1
                perm_p1 = np.copy(p1)
                perm_p1[comb1, :] = perm_p1[perm1, :]
                # permute rows of matrix P2
                perm_p2 = np.copy(p2)
                perm_p2[comb2, :] = perm_p2[perm2, :]
                # compute error with new matrices & compare
                perm_error = compute_error(b, a, perm_p1, perm_p2)
                if perm_error < error:
                    p1, p2, error = perm_p1, perm_p2, perm_error
                    # check whether perfect permutation matrix is found
                    # TODO: smarter threshold based on norm of matrix
                    if error <= 1.0e-8:
                        break
    return p1, p2, error
