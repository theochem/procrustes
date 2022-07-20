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

from math import inf, sqrt

import numpy as np
from procrustes.utils import ProcrustesResult
import scipy
from scipy.optimize import minimize


__all__ = ["psdp_woodgate"]


def psdp_woodgate(a: np.ndarray, b: np.ndarray) -> ProcrustesResult:
    r"""
    Woodgate's algorithm for positive semidefinite Procrustes.

    Parameters
    ----------
    a : np.ndarray
        The matrix to be transformed.
        This is relabelled to :math:`\mathbf{G}` as in the paper.

    b : np.ndarray
        The target matrix.
        This is relabellled to :math:`\mathbf{F}` as in the paper.

    Returns
    -------
    ProcrustesResult
        The result of the Procrustes transformation.

    Notes
    -----
    Given :math:`\mathbf{F}, \mathbf{G} \in R^{\ n\times m}`, the woodgate
    algorithm finds :math:`\mathbf{P} \in S^{\ n\times n}_{\geq}` such
    that the following is true:

    .. math::
        \text{PSDP: } min_{\mathbf{P}} \|\mathbf{F}-\mathbf{P}\mathbf{G}\|

    Woodgate's algorithm takes a non-convex approach to the above problem.
    It finds solution to the following which serves as a subroutine to our
    original problem.

    .. math::
        \text{PSDP*: } min_{\mathbf{E} \in R^{\ n\times n}} \|\mathbf{F}
            - \mathbf{E}^T\mathbf{E}\mathbf{G}\|

    Now, since all local minimizers of PSDP* are also global minimizers, we
    have :math:`\hat{\mathbf{P}} = \hat{\mathbf{E}}^T\mathbf{E}` where
    :math:`\hat{\mathbf{E}}` is any local minimizer of PSDP* and
    :math:`\hat{\mathbf{P}}` is the required minimizer for our original PSDP
    problem.

    The main algorithm is as follows:

    - :math:`\mathbf{E}_0` is chosen randomly, :math:`i = 0`.
    - Compute :math:`L(\mathbf{E}_i)`.
    - If :math:`L(\mathbf{E}_i) \geq 0` then we stop and
        :math:`\mathbf{E}_i` is the answer.
    - Compute :math:`\mathbf{D}_i`.
    - Minimize :math:`f(\mathbf{E}_i - w_i \mathbf{D}_i)`.
    - :math:`\mathbf{E}_{i + 1} = \mathbf{E}_i - w_i_min \mathbf{D}_i`
    - :math:`i = i + 1`, start from 2 again.


    References
    ----------
    .. [1] Woodgate, K. G. (1996). "A new algorithm for positive semidefinite
        procrustes". Journal of the American Statistical Association, 93(453),
        584-588.
    """

    # We define the matrices F, G and Q as in the paper.
    # Our plan is to find matrix P such that, |F - PG| is minimized.
    # Now, |F - PG| = |PG - F| = |PGI - F| = |PAI - F|.
    f = b
    g = a
    q = f @ g.T + g @ f.T

    # We define the functions L and f as in the paper.
    func_l = lambda arr: (arr.T @ arr @ g @ g.T) + (g @ g.T @ arr.T @ arr) - q
    func_f = lambda arr: (1 / 2) * (
        np.trace(f.T @ f)
        + np.trace(arr.T @ arr @ arr.T @ arr @ g @ g.T)
        - np.trace(arr.T @ arr @ q)
    )

    # Main part of the algorithm.
    i = 0
    n = f.shape[0]
    e = _scale(e=np.eye(n), g=g, q=q)
    error = [inf]

    while True:
        le = func_l(e)
        error.append(np.linalg.norm(f - e.T @ e @ g))

        # Check if positive semidefinite or if the algorithm has converged.
        if (np.all(np.linalg.eigvals(le)) >= 0) or (abs(error[-1] - error[-2]) < 1e-5):
            break

        # Make all the eigenvalues of le positive and use it to compute d.
        le_pos = _make_positive(le)
        d = _find_gradient(e=e, le=le_pos, g=g)

        # Objective function which we want to minimize.
        func_obj = lambda w: func_f(e - w * d)
        w_min = minimize(func_obj, 1, bounds=((1e-9, None),)).x[0]
        e = _scale(e=(e - w_min * d), g=g, q=q)
        i += 1

    # Returning the result as a ProcrastesResult object.
    return ProcrustesResult(
        new_a=a, new_b=b, error=error[-1], s=(e.T @ e), t=np.eye(a.shape[1])
    )


def _permutation_matrix(arr: np.ndarray) -> np.ndarray:
    r"""
    Find required permutation matrix.

    Parameters
    ----------
    arr : np.ndarray
        The array :math:`\mathbf{A}` such that
        :math:`v(\mathbf{A}') = \mathbf{P}v(\mathbf{A})`.

    Returns
    -------
    np.ndarray
        The permutation matrix.
    """
    k = 0
    n = arr.shape[0]
    p = np.zeros((n**2, n**2))

    for i in range(n**2):
        if i % n == 0:
            j = k
            k += 1
            p[i, j] = 1
        else:
            j += n
            p[i, j] = 1
    return p


def _make_positive(arr: np.ndarray) -> np.ndarray:
    r"""
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
    eigenvalues, unitary = np.linalg.eig(arr)
    eigenvalues_pos = [max(0, i) for i in eigenvalues]
    unitary_inv = np.linalg.inv(unitary)
    return np.dot(unitary, np.dot(np.diag(eigenvalues_pos), unitary_inv))


def _find_gradient(e: np.ndarray, le: np.ndarray, g: np.ndarray) -> np.ndarray:
    r"""
    Find the gradient of the function f(E).

    Parameters
    ----------
    e : np.ndarray
        The input to the function f. This is :math:`\mathbf{E_i}` in the paper.

    l : np.ndarray
        A matrix to be used in the gradient computation.
        This is :math:`L(\mathbf{E}_i)` in the paper.

    g : np.ndarray
        This is the original :math:`\mathbf{G}` matrix obtained as input.

    Returns
    -------
    np.ndarray
        The required gradient. This is :math:`\mathbf{D_i}` in the paper.

    Notes
    -----
    The gradient is :math:`\mathbf{D}_i = \nabla_{\mathbf{E}} f(\mathbf{E}_i)`
    and it is constructed using two parts, namely, :math:`\mathbf{D}_1` and
    :math:`\mathbf{D}_2`, which denote the top and bottom parts of the gradient
    matrix.

    Specifically, :math:`\mathbf{D}_1` denoyes the top :math:`s` rows of the
    gradient matrix, where, :math:`s` is the rank of the matrix :math:`\mathbf{E}_i`.
    We, furthermore, define E1 as the first :math:`s` rows of :math:`\mathbf{E_i}`.

    .. math::
        \mathbf{D}_2 L(\mathbf{E}_i) = 0\\
        (X + (I\otimes L(\mathbf{E}_i))) v(\mathbf{D}_1)
            = (I\otimes L(\mathbf{E}_i)) v(\mathbf{E}_1)

    In the following implementation, the variables d1 and d2 represent
    :math:`\mathbf{D}_1` and :math:`\mathbf{D}_2`, respectively.

    References
    ----------
    Refer to equations (26) and (27) in [1] for exact deinitions of the
    several terms mentioned in this function.
    .. [1] Woodgate, K. G. (1996). "A new algorithm for positive
        semidefinite procrustes". Journal of the American Statistical
        Association, 93(453), 584-588.
    """
    n = e.shape[0]
    s = np.linalg.matrix_rank(e)
    v = scipy.linalg.null_space(le.T).flatten()
    d2 = np.outer(v, v)

    p = _permutation_matrix(e)
    identity_z = np.eye(
        (np.kron(e @ e.T, g @ g.T)).shape[0] // (e @ g @ g.T @ e.T).shape[0]
    )
    z = (
        np.kron(e @ g @ g.T, e.T) @ p
        + np.kron(e, g @ g.T @ e.T) @ p
        + np.kron(e @ g @ g.T @ e.T, identity_z)
        + np.kron(e @ e.T, g @ g.T)
    )

    x = z if s == n else z[: n * (n - s), : n * (n - s)]
    identity_x = np.eye(x.shape[0] // le.shape[0])
    flattened_d1 = (
        np.linalg.pinv(x + np.kron(identity_x, le))
        @ np.kron(identity_x, le)
        @ e[:s, :].flatten()
    )

    if s == n:
        d = flattened_d1.reshape(s, n)
    else:
        d = np.concatenate((flattened_d1, d2), axis=0)
    return d


def _scale(e: np.ndarray, g: np.ndarray, q: np.ndarray) -> np.ndarray:
    r"""
    Find the scaling factor and scale the matrix e.

    Parameters
    ----------
    e : np.ndarray
        This is the matrix to be scaled.

    g : np.ndarray
        This is the original :math:`\mathbf{G}` matrix obtained as input.

    q : np.ndarray
        This is the matrix :math:`\mathbf{Q}` in the paper.

    Returns
    -------
    np.ndarray
        The scaled matrix.
    """
    alpha = sqrt(
        max(1e-9, np.trace(e.T @ e @ q) / (2 * np.trace(e.T @ e @ e.T @ e @ g @ g.T)))
    )
    return alpha * e
