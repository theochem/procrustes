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
"""Positive semidefinite Procrustes Module."""

import numpy as np
import scipy.linalg as lin
from scipy.optimize import minimize
from procrustes.utils import ProcrustesResult

__all__ = ["woodgate"]


def permutation_matrix(arr: np.ndarray) -> np.ndarray:
    """
    Find required permutation matrix.

    Parameters
    ----------
    arr : np.ndarray
        The array :math:`A` such that :math:`v(A') = Pv(A)`.

    Returns
    -------
    np.ndarray
        The permutation matrix.
    """
    k = 0
    x, y = arr.shape
    P = np.zeros((x**2, x**2))

    for i in range(x**2):
        if i % x == 0:
            j = k
            k += 1
            P[i, j] = 1
        else:
            j += x
            P[i, j] = 1
    return P


def is_pos_semi_def(x: np.ndarray) -> bool:
    """
    Check if a matrix is positive semidefinite.

    Parameters
    ----------
    x : np.ndarray
        The matrix to be checked.

    Returns
    -------
    bool
        True if the matrix is positive semidefinite.
    """
    return np.all(np.linalg.eigvals(x) >= 0)


def make_positive(arr: np.ndarray) -> np.ndarray:
    """
    Re-construct a matrix by making all its negative eigenvalues zero.

    Parameters
    ----------
    arr : np.ndarray
        The matrix to be re-constructed.

    Returns
    -------
    np.ndarray
        The re-constructed matrix.
    """
    eigenvalues, U = np.linalg.eig(arr)
    eigenvalues_pos = [max(0, i) for i in eigenvalues]
    U_inv = np.linalg.inv(U)
    return U @ np.diag(eigenvalues_pos) @ U_inv


def find_gradient(E: np.ndarray, LE: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    Find the gradient of the function f(E).

    Parameters
    ----------
    E : np.ndarray
        The input to the function f. This is E_i in the paper.

    LE : np.ndarray
        A matrix to be used in the gradient computation.
        This is L(E_i) in the paper.

    Returns
    -------
    np.ndarray
        The required gradient. This is D_i in the paper.

    Notes
    -----
    The gradient is defined as :math:`D_i = \nabla_{E} f(E_i)` and it
    is constructed using two parts, namely, D1 and D2, which denote the
    top and bottom parts of the gradient matrix.

    Specifically, D1 denoyes the top :math:`s` rows of the gradient matrix,
    where, :math:`s` is the rank of the matrix :math:`E_i`. We, furthermore,
    define E1 as the first :math:`s` rows of E_i.

    .. math::
        D2 L(E_i) = 0
        (X + (I \otimes L(E_i))) v(D1) = (I \otimes L(E_i)) v(E1)

    References
    ----------
    Refer to equations (26) and (27) in [1] for exact deinitions of the
    several terms mentioned in this function.
    .. [1] Woodgate, K. G. (1996). "A new algorithm for positive
        semidefinite procrustes". Journal of the American Statistical
        Association, 93(453), 584-588.
    """
    n = E.shape[0]
    s = np.linalg.matrix_rank(E)
    v = lin.null_space(LE.T).flatten()
    D2 = np.outer(v, v)

    P = permutation_matrix(E)
    identity_Z = np.eye(
        (np.kron(E @ E.T, G @ G.T)).shape[0] // (E @ G @ G.T @ E.T).shape[0]
    )
    Z = (
        np.kron(E @ G @ G.T, E.T) @ P
        + np.kron(E, G @ G.T @ E.T) @ P
        + np.kron(E @ G @ G.T @ E.T, identity_Z)
        + np.kron(E @ E.T, G @ G.T)
    )

    X = Z if s == n else Z[: n * (n - s), : n * (n - s)]
    identity_X = np.eye(X.shape[0] // LE.shape[0])
    flattened_D1 = (
        np.linalg.pinv(X + np.kron(identity_X, LE))
        @ np.kron(identity_X, LE)
        @ E[:s, :].flatten()
    )

    if s == n:
        D = flattened_D1.reshape(s, n)
    else:
        D = np.concatenate((flattened_D1, D2), axis=0)
    return D


def woodgate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Woodgate's algorithm for positive semidefinite Procrustes.

    Parameters
    ----------
    a : np.ndarray
        The matrix to be transformed.
        This is relabelled to G as in the paper.
    b : np.ndarray
        The target matrix.
        This is relabellled to F as in the paper.

    Returns
    -------
    ProcrustesResult
        The result of the Procrustes transformation.

    Notes
    -----
    Given :math:`F, G \in R^{\ n\times m}`, the woodgate algorithm finds
    :math:`P \in S^{\ n\times n}_{\geq}` such that the following is true:

    .. math::
        \text{PSDP: } min_{P} \|F - PG\|

    Woodgate's algorithm takes a non-convex approach to the above problem.
    It finds solution to the following which serves as a subroutine to our
    original problem.

    .. math::
        \text{PSDP*: } min_{E \in R^{\ n\times n}} \|F - E'EG\|

    Now, since all local minimizers of PSDP* are also global minimizers, we
    have :math:`\hat{P} = \hat{E}'E` where :math:`\hat{E}` is any local
    minimizer of PSDP* and :math:`\hat{P}` is the required minimizer for
    our originak PSDP problem.

    The main algorithm is as follows:

    - :math:`E_0` is chosen randomly, :math:`i = 0`.
    - Compute :math:`L(E_i)`.
    - If :math:`L(E_i) \geq 0` then we stop and :math:`E_i` is the answer.
    - Compute :math:`D_i`.
    - Minimize :math:`f(E_i - w_i D_i)`.
    - :math:`E_{i + 1} = E_i - w_i_min D_i`
    - :math:`i = i + 1`, start from 2 again.


    References
    ----------
    .. [1] Woodgate, K. G. (1996). "A new algorithm for positive semidefinite
        procrustes". Journal of the American Statistical Association, 93(453),
        584-588.
    """

    # We define the matrices F, G and Q as in the paper.
    F = a
    G = b
    Q = F @ G.T + G @ F.T

    # We define the functions f and L as in the paper.
    L = lambda arr: (arr.T @ arr @ G @ G.T) + (G @ G.T @ arr.T @ arr) - Q
    f = lambda arr: (1 / 2) * (
        np.trace(F.T @ F)
        + np.trace(arr.T @ arr @ arr.T @ arr @ G @ G.T)
        - np.trace(arr.T @ arr @ Q)
    )

    # Main part of the algorithm.
    i = 0
    n = F.shape[0]
    E = np.random.rand(n, n)

    while True:
        LE = L(E)
        if is_pos_semi_def(LE):
            break

        LE_pos = make_positive(LE)
        D = find_gradient(E=E, LE=LE_pos, G=G)

        func = lambda w: f(E - w * D)
        w_min = minimize(func, 1, bounds=((1e-9, None),)).x[0]
        E = E - w_min * D
        i += 1

    print(f"Woodgate's algorithm took {i} iterations.")
    print(f"Error = {np.linalg.norm(F - E.T @ E @ G)}.")
    print(f"Required P = {E.T @ E}")
    return E.T @ E
