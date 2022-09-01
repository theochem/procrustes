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
from typing import Optional

import numpy as np
from numpy.linalg import multi_dot
from procrustes.utils import ProcrustesResult, setup_input_arrays
import scipy
from scipy.optimize import minimize

__all__ = ["psdp_woodgate", "psdp_peng"]


def psdp_peng(
    a: np.ndarray,
    b: np.ndarray,
    pad: bool = True,
    translate: bool = False,
    scale: bool = False,
    unpad_col: bool = False,
    unpad_row: bool = False,
    check_finite: bool = True,
    weight: Optional[np.ndarray] = None,
) -> ProcrustesResult:
    r"""
    Peng's algorithm for the symmetric positive semi-definite Procrustes problem.

    Given a matrix :math:`\mathbf{A}_{n \times m}` and a reference matrix :math:`\mathbf{B}_{n
    \times m}`, find the positive semidefinite transformation matrix :math:`\mathbf{P}_{n
    \times n}` that makes :math:`\mathbf{PA}` as close as possible to :math:`\mathbf{B}`.
    In other words,

    .. math::
        \text{PSDP: } min_{\mathbf{P}} \|\mathbf{B}-\mathbf{P}\mathbf{A}\|_{F}^2

    Throughout all methods used for implementing the Peng et al algorithm, the matrices
    :math:`\mathbf{A}` and :math:`\mathbf{B}` are referred to as :math:`\mathbf{G}` and
    :math:`\mathbf{F}` respectively, following the nomenclature used in [1]_.

    Parameters
    ----------
    a : np.ndarray
        The matrix :math:`\mathbf{A}` which is to be transformed.
        This is relabelled to variable g representing the matrix :math:`\mathbf{G}` as
        in the paper.

    b : np.ndarray
        The target matrix :math:`\mathbf{B}`.
        This is relabelled to variable f representing the matrix :math:`\mathbf{F}` as
        in the paper.

    pad : bool, optional
        Add zero rows (at the bottom) and/or columns (to the right-hand side) of matrices
        :math:`\mathbf{A}` and :math:`\mathbf{B}` so that they have the same shape.

    translate : bool, optional
        If True, both arrays are centered at origin (columns of the arrays will have mean zero).

    scale : bool, optional
        If True, both arrays are normalized with respect to the Frobenius norm, i.e.,
        :math:`\text{Tr}\left[\mathbf{A}^\dagger\mathbf{A}\right] = 1` and
        :math:`\text{Tr}\left[\mathbf{B}^\dagger\mathbf{B}\right] = 1`.

    unpad_col : bool, optional
        If True, zero columns (with values less than 1.0e-8) on the right-hand side of the intial
        :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices are removed.

    unpad_row : bool, optional
        If True, zero rows (with values less than 1.0e-8) at the bottom of the intial
        :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices are removed.

    check_finite : bool, optional
        If True, convert the input to an array, checking for NaNs or Infs.

    weight : ndarray, optional
        The 1D-array representing the weights of each row of :math:`\mathbf{A}`. This defines the
        elements of the diagonal matrix :math:`\mathbf{W}` that is multiplied by :math:`\mathbf{A}`
        matrix, i.e., :math:`\mathbf{A} \rightarrow \mathbf{WA}`.

    Returns
    -------
    ProcrustesResult
        The result of the Procrustes transformation.

    Notes
    -----
    The algorithm is constructive in nature and can be described as follows:

    - Decompose the matrix :math:`\mathbf{G}` into its singular value decomposition.
    - Construct :math:`\hat{\mathbf{S}} = \mathbf{\Phi} \ast (
        \mathbf{U}^T_1 \mathbf{F} \mathbf{V}_1 \mathbf{\Sigma}
        + \mathbf{\Sigma} \mathbf{V}^T_1 \mathbf{F}^T \mathbf{U}_1)`.
    - Perform spectral decomposition of :math:`\hat{\mathbf{S}}`.
    - Computing intermediate matrices $\mathbf{P}_{11}$ and $\mathbf{P}_{12}$.
    - Check if solution exists.
    - Compute $\hat{\mathbf{P}}$ (required minimizer) using $\mathbf{P}_{11}$ and
        $\mathbf{P}_{12}$.

    References
    ----------
    .. [1] Jingjing Peng et al (2019). "Solution of symmetric positive semidefinite Procrustes
        problem". Electronic Journal of Linear Algebra, ISSN 1081-3810. Volume 35, pp. 543-554.
    """
    # Check inputs and define the matrices F (matrix to be transformed) and
    # G (the target matrix).
    g, f = setup_input_arrays(
        a,
        b,
        unpad_col,
        unpad_row,
        pad,
        translate,
        scale,
        check_finite,
        weight,
    )
    if g.shape != f.shape:
        raise ValueError(
            f"Shape of A and B does not match: {g.shape} != {f.shape} "
            "Check pad, unpad_col, and unpad_row arguments."
        )

    # Perform Singular Value Decomposition (SVD) of G (here g).
    u, singular_values, v_transpose = np.linalg.svd(g, full_matrices=True)
    r = len(singular_values)
    u1, u2 = u[:, :r], u[:, r:]
    v1 = v_transpose.T[:, :r]
    sigma = np.diag(singular_values)

    # Representing the intermediate matrix S.
    phi = np.array(
        [[1 / (i**2 + j**2) for i in singular_values] for j in singular_values]
    )
    s = np.multiply(
        phi, multi_dot([u1.T, f, v1, sigma]) + multi_dot([sigma, v1.T, f.T, u1])
    )

    # Perform Spectral Decomposition on S (here named s).
    eigenvalues, unitary = np.linalg.eig(s)
    eigenvalues_pos = [max(0, i) for i in eigenvalues]

    # Computing intermediate matrices required to construct the required
    # optimal transformation.
    p11 = multi_dot([unitary, np.diag(eigenvalues_pos), np.linalg.inv(unitary)])
    p12 = multi_dot([np.linalg.inv(sigma), v1.T, f.T, u2])

    # Checking if solution is possible.
    if np.linalg.matrix_rank(p11) != np.linalg.matrix_rank(
        np.concatenate((p11, p12), axis=1)
    ):
        raise ValueError(
            "Rank mismatch. Symmetric positive semidefinite Procrustes problem has no solution."
        )

    # Finding the required minimizer (optimal transformation).
    mid1 = np.concatenate((p11, p12), axis=1)
    mid2 = np.concatenate((p12.T, multi_dot([p12.T, np.linalg.pinv(p11), p12])), axis=1)
    mid = np.concatenate((mid1, mid2), axis=0)
    p = multi_dot([u, mid, u.T])

    # Returning the result as a ProcrastesResult object.
    return ProcrustesResult(
        new_a=a,
        new_b=b,
        error=np.linalg.norm(f - np.dot(p, g)),
        s=p,
        t=np.eye(a.shape[1]),
    )


def psdp_woodgate(
    a: np.ndarray,
    b: np.ndarray,
    pad: bool = True,
    translate: bool = False,
    scale: bool = False,
    unpad_col: bool = False,
    unpad_row: bool = False,
    check_finite: bool = True,
    weight: Optional[np.ndarray] = None,
) -> ProcrustesResult:
    r"""
    Woodgate's algorithm for positive semidefinite Procrustes.

    Given a matrix :math:`\mathbf{A}_{n \times m}` and a reference matrix :math:`\mathbf{B}_{n
    \times m}`, find the positive semidefinite transformation matrix :math:`\mathbf{P}_{n
    \times n}` that makes :math:`\mathbf{PA}` as close as possible to :math:`\mathbf{B}`.
    In other words,

    .. math::
        \text{PSDP: } min_{\mathbf{P}} \|\mathbf{B}-\mathbf{P}\mathbf{A}\|_{F}^2

    This Procrustes method requires the :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices to
    have the same shape, which is gauranteed with the default ``pad`` argument for any given
    :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices. In preparing the :math:`\mathbf{A}` and
    :math:`\mathbf{B}` matrices, the (optional) order of operations is: **1)** unpad zero
    rows/columns, **2)** translate the matrices to the origin, **3)** weight entries of
    :math:`\mathbf{A}`, **4)** scale the matrices to have unit norm, **5)** pad matrices with zero
    rows/columns so they have the same shape.

    Throughout all methods used for implementing the Woodgate algorithm, the matrices
    :math:`\mathbf{A}` and :math:`\mathbf{B}` are referred to as :math:`\mathbf{G}` and
    :math:`\mathbf{F}` respectively, following the nomenclature used in [1]_.

    Parameters
    ----------
    a : np.ndarray
        The matrix :math:`\mathbf{A}` which is to be transformed.
        This is relabelled to variable g representing the matrix :math:`\mathbf{G}` as
        in the paper.

    b : np.ndarray
        The target matrix :math:`\mathbf{B}`.
        This is relabelled to variable f representing the matrix :math:`\mathbf{F}` as
        in the paper.

    pad : bool, optional
        Add zero rows (at the bottom) and/or columns (to the right-hand side) of matrices
        :math:`\mathbf{A}` and :math:`\mathbf{B}` so that they have the same shape.

    translate : bool, optional
        If True, both arrays are centered at origin (columns of the arrays will have mean zero).

    scale : bool, optional
        If True, both arrays are normalized with respect to the Frobenius norm, i.e.,
        :math:`\text{Tr}\left[\mathbf{A}^\dagger\mathbf{A}\right] = 1` and
        :math:`\text{Tr}\left[\mathbf{B}^\dagger\mathbf{B}\right] = 1`.

    unpad_col : bool, optional
        If True, zero columns (with values less than 1.0e-8) on the right-hand side of the intial
        :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices are removed.

    unpad_row : bool, optional
        If True, zero rows (with values less than 1.0e-8) at the bottom of the intial
        :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices are removed.

    check_finite : bool, optional
        If True, convert the input to an array, checking for NaNs or Infs.

    weight : ndarray, optional
        The 1D-array representing the weights of each row of :math:`\mathbf{A}`. This defines the
        elements of the diagonal matrix :math:`\mathbf{W}` that is multiplied by :math:`\mathbf{A}`
        matrix, i.e., :math:`\mathbf{A} \rightarrow \mathbf{WA}`.

    Returns
    -------
    ProcrustesResult
        The result of the Procrustes transformation.

    Notes
    -----
    Given :math:`\mathbf{F}, \mathbf{G} \in \mathbb{R}^{\ n\times m}`, the woodgate
    algorithm finds the positive semidefinite matrix :math:`\mathbf{P}_{n \times n}`
    such that the following is true:

    .. math::
        \text{PSDP: } min_{\mathbf{P}} \|\mathbf{F}-\mathbf{P}\mathbf{G}\|

    Woodgate's algorithm takes a non-convex approach to the above problem.
    It finds solution to the following which serves as a subroutine to our
    original problem.

    .. math::
        \text{PSDP*: } min_{\mathbf{E} \in \mathbb{R}^{\ n\times n}}
            \|\mathbf{F} - \mathbf{E}^T\mathbf{E}\mathbf{G}\|

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
    # Check inputs and define the matrices F (matrix to be transformed) and
    # G (the target matrix).
    g, f = setup_input_arrays(
        a,
        b,
        unpad_col,
        unpad_row,
        pad,
        translate,
        scale,
        check_finite,
        weight,
    )
    if g.shape != f.shape:
        raise ValueError(
            f"Shape of A and B does not match: {g.shape} != {f.shape} "
            "Check pad, unpad_col, and unpad_row arguments."
        )

    # We define the matrix Q as in the paper.
    # Our plan is to find matrix P such that, |F - PG| is minimized.
    # Now, |F - PG| = |PG - F| = |PGI - F| = |PAI - F|.
    q = f.dot(g.T) + g.dot(f.T)

    # We define the functions L and f as in the paper.
    func_l = (
        lambda arr: (multi_dot([arr.T, arr, g, g.T]))
        + (multi_dot([g, g.T, arr.T, arr]))
        - q
    )

    def func_f(arr):
        return (1 / 2) * (
            np.trace(np.dot(f.T, f))
            + np.trace(multi_dot([arr.T, arr, arr.T, arr, g, g.T]))
            - np.trace(multi_dot([arr.T, arr, q]))
        )

    # Main part of the algorithm.
    i = 0
    n = f.shape[0]
    e = _scale(e=np.eye(n), g=g, q=q)
    error = [inf]

    while True:
        le = func_l(e)
        error.append(np.linalg.norm(f - multi_dot([e.T, e, g])))

        # Check if positive semidefinite or if the algorithm has converged.
        if np.all(np.linalg.eigvals(le) >= 0) or abs(error[-1] - error[-2]) < 1e-5:
            break

        # Make all the eigenvalues of le positive and use it to compute d.
        le_pos = _make_positive(le)
        d = _find_gradient(e=e, le=le_pos, g=g)

        # Objective function which we want to minimize.
        def func_obj(w):
            return func_f(e - w * d)

        w_min = minimize(func_obj, 1, bounds=((1e-9, None),)).x[0]
        e = _scale(e=(e - w_min * d), g=g, q=q)
        i += 1

    # Returning the result as a ProcrastesResult object.
    return ProcrustesResult(
        new_a=a, new_b=b, error=error[-1], s=np.dot(e.T, e), t=np.eye(a.shape[1])
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
    :math:`\mathbf{D}_1` and :math:`\mathbf{D}_2`, respectively. Refer to
    equations (26) and (27) in [1]_ for exact deinitions of the several terms mentioned
    in this function.

    References
    ----------
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
        (np.kron(e.dot(e.T), g.dot(g.T))).shape[0]
        // (multi_dot([e, g, g.T, e.T])).shape[0]
    )
    z = (
        np.kron(multi_dot([e, g, g.T]), e.T).dot(p)
        + np.kron(e, multi_dot([g, g.T, e.T])).dot(p)
        + np.kron(multi_dot([e, g, g.T, e.T]), identity_z)
        + np.kron(e.dot(e.T), g.dot(g.T))
    )

    x = z if s == n else z[: n * (n - s), : n * (n - s)]
    identity_x = np.eye(x.shape[0] // le.shape[0])
    flattened_d1 = multi_dot(
        [
            np.linalg.pinv(x + np.kron(identity_x, le)),
            np.kron(identity_x, le),
            e[:s, :].flatten(),
        ]
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
        max(
            1e-9,
            np.trace(multi_dot([e.T, e, q]))
            / (2 * np.trace(multi_dot([e.T, e, e.T, e, g, g.T]))),
        )
    )
    return alpha * e
