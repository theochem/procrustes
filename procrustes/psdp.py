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
from typing import Dict, Optional

import numpy as np
import scipy
from numpy.linalg import multi_dot
from scipy.optimize import minimize

from procrustes.utils import ProcrustesResult, compute_error, setup_input_arrays

__all__ = [
    "psdp_woodgate",
    "psdp_peng",
    "psdp_opt",
    "psdp_projgrad",
]


def psdp_projgrad(
    a: np.ndarray,
    b: np.ndarray,
    options_dict: Dict = None,
    pad: bool = True,
    translate: bool = False,
    scale: bool = False,
    unpad_col: bool = False,
    unpad_row: bool = False,
    check_finite: bool = True,
    weight: Optional[np.ndarray] = None,
) -> ProcrustesResult:
    r"""
    Projected gradient method for the positive semi-definite Procrustes problem.

    We want to minimize the function F = ||SAT-B||_F where A and B are n*m matrices and S is n*n
    transformation we want to find that minimizes the above function and T is just an identity
    matrix for now. We are only considering left transformation.

    The paper minimizes the following function ||AX-B||_F, so the mapping between the problem
    statements are

    Term used in paper             Term used in our implementation
    --------------------------------------------------------------
    A                               S
    X                               A
    B                               B

    Parameters
    ----------
    a : np.ndarray
        The matrix :math:`\mathbf{A}` which is to be transformed.

    b : np.ndarray
        The target matrix :math:`\mathbf{B}`.

    options : Dict, optional
        Dictionary with fields that serve as parameters for the algorithm.

        max_iter : int
            Maximum number of iterations.
            Default value is 10000.

        s_tol : float
            Stop control for ||S_i - S_{i-1}||_F / ||S_1 - S_0||_F
            Defaut value is 1e-5. Should be kept below 1

        f_tol : float
            Stop control for ||F_i - F_{i-1}||_F/(1+||F_{i-1}||_F).
            Default value is 1e-12. Should be kept way less than 1

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
    The Projected Gradient algorithm (on which this implementation is based) is defined well in
    p. 131 of [1]_.

    References
    ----------
    .. [1] Nicolas Gillis, Punit Sharma, "A semi-analytical approach for the positive semidefinite
        Procrustesproblem, Linear Algebra and its Applications, Volume 540, 2018, Pages 112-137
    """
    a, b = setup_input_arrays(
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

    if a.shape != b.shape:
        raise ValueError(
            f"Shape of A and B does not match: {a.shape} != {b.shape} "
            "Check pad, unpad_col, and unpad_row arguments."
        )

    _, m = a.shape

    # Specifies the default parameters of the algorithm
    options = {
        # The maximum number of iterations the gradient descent algorithm must run for
        "max_iter": 10000,
        # The minimum ratio of ||S_i - S_{i - 1}||_F with ||S_1 - S_0||_F, below which algorithm
        # terminates
        "s_tol": 1e-5,
        # The minimum ratio of ||F_i - F_{i - 1}||_F with 1 + ||F_{i - 1}||_F, below which algorithm
        # terminates
        "f_tol": 1e-12
    }

    if options_dict:
        options.update(options_dict)

    # Performing some precomputations
    aat = a@a.conj().T
    x, _ = np.linalg.eig(aat)
    max_eig = np.max(x) ** 2  # if they are complex, then need to take norm
    bat = b@a.conj().T

    # Initialization of the algorithm
    i = 1
    err = np.zeros((options["max_iter"] + 1, 1))
    # S is the right transformation in our problem statement
    s = _init_procustes_projgrad(a, b)
    # F is the function whose norm we want to minimize, F = S@A - B
    f = s @ a - b
    # eps = ||S_i - S_{i - 1}||_F
    eps = 1
    # eps0 = ||S_1 - S_0||_F
    eps0 = 0

    # Algorithm is run until the max iterations
    while i <= options["max_iter"]:
        s_old = s
        f_old = f
        # Projected gradient step
        s = _psd_proj(s - (s@aat - bat)/max_eig)
        f = s @ a - b
        err[i] = np.linalg.norm(f, 'fro')

        # Stop conditions to check if the algorithm should be terminated

        # If the ||F_i - F_{i - 1}||_F / (1 + ||F_old||_F) is lesser than some tolerance level,
        # then we terminate the algorithm, as there is not much improvement gain in the
        # optimisation problem for the extra iterations we perform
        if np.linalg.norm(f - f_old, 'fro') / (1 + np.linalg.norm(f_old, 'fro')) < options["f_tol"]:
            break

        # If the ||S_i - S_{i - 1}||_F / ||S_1 - S_0||_F is lesser than some tolerance level,
        # then we terminate the algorithm. TODO: to decide if this is useful or not.
        if i == 1:
            eps0 = np.linalg.norm(s - s_old, 'fro')
        eps = np.linalg.norm(s - s_old, 'fro')
        if eps < options["s_tol"] * eps0:
            break

        i += 1

    return ProcrustesResult(
        new_a=a,
        new_b=b,
        error=compute_error(a=a, b=b, t=np.eye(m), s=s),
        t=np.eye(m),
        s=s,
    )


def psdp_opt(
    a: np.ndarray,
    b: np.ndarray,
    options_dict: Dict = None,
    pad: bool = True,
    translate: bool = False,
    scale: bool = False,
    unpad_col: bool = False,
    unpad_row: bool = False,
    check_finite: bool = True,
    weight: Optional[np.ndarray] = None,
) -> ProcrustesResult:
    r"""
    Spectral projected gradient method for the positive semi-definite Procrustes problem.

    Given a matrix :math:`\mathbf{A}_{n \times m}` and a reference matrix :math:`\mathbf{B}_{n
    \times m}`, find the positive semidefinite transformation matrix :math:`\mathbf{X}_{n
    \times n}` that makes :math:`\mathbf{XA}` as close as possible to :math:`\mathbf{B}`.
    In other words,

    .. math::
        \text{PSDP: } min_{\mathbf{X}} \|\mathbf{X}\mathbf{A}-\mathbf{B}\|_{F}^2

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

    options : Dict, optional
        Dictionary with fields that serve as parameters for the algorithm.

        max_iter : int
            Maximum number of iterations.
            Default value is 10000.

        x_tol : float
            Stop control for ||X_k - X_{k-1}||_F.
            Defaut value is 1e-5.

        f_tol : float
            Stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|).
            Default value is 1e-12.

        proj : bool
            If proj is True we perform Cholesky decomposition else we do spectral
            decomposition.
            Default value is True.

        gamma : float
            Parameter of the non-monotone technique proposed by Zhang-Hager.
            Default value is 0.85.

        rho : float
            Parameter for control the linear approximation in line search.
            Default value is 1e-4.

        eta : float
            Factor for decreasing the step size in the backtracking line search.
            Default value is 0.1.

        tau : float
            Initial step size with default value 1e-3.

        nls : int
            Number of internal iterations with default value 5.

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
    The OptPSDP algorithm (on which this implementation is based) is defined well in p. 114
    of [1]_.

    References
    ----------
    .. [1] Harry F. Oviedo (2019). "A Spectral Gradient Projection Method for the Positive
        Semi-definite Procrustes Problem". Revista Colombiana de Matematicas, Volume 55(2021)1,
        pages 109-123.
    """
    a, b = setup_input_arrays(
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
    if a.shape != b.shape:
        raise ValueError(
            f"Shape of A and B does not match: {a.shape} != {b.shape} "
            "Check pad, unpad_col, and unpad_row arguments."
        )

    # Initializing the required minimizer.
    # Here, "a" denotes the matrix to be transformed i.e. A, x is the transformer X
    # and b is the target matrix B. Our goal is to minimize ||XA - B||_F
    # (as mentioned in the function description).
    n, m = a.shape
    x = np.eye(n)

    # Option structure with fields that serve as parameters for the algorithm.
    options = {
        # Maximum number of iterations.
        "max_iter": 10000,
        # Stop control for ||X_k - X_{k-1}||_F.
        "x_tol": 1e-5,
        # Stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|).
        "f_tol": 1e-12,
        # If proj is True we perform Cholesky decomposition else we do spectral
        # decomposition.
        "proj": True,
        # Parameter of the non-monotone technique proposed by Zhang-Hager.
        "gamma": 0.85,
        # Parameter for control the linear approximation in line search.
        "rho": 1e-4,
        # Factor for decreasing the step size in the backtracking line search.
        "eta": 0.1,
        # Initial step size.
        "tau": 1e-3,
        # Number of internal iterations.
        "nls": 5,
    }
    # update the parameter dictionary based on values provided by the user
    if options_dict:
        options.update(options_dict)

    hold = np.dot(x, a) - b
    grad = np.dot(hold, a.T)
    f = np.linalg.norm(hold) ** 2
    norm_grad = np.linalg.norm(grad)
    q = 1
    cval = f

    # Main iteration of the algorithm.
    for i in range(options["max_iter"]):
        x_old = x
        f_old = f
        grad_old = grad
        derivative = options["rho"] * (norm_grad**2)
        nls = 1

        while True:
            x = x_old - options["tau"] * grad_old
            x = _psd_proj(x, options["proj"])
            hold = np.dot(x, a) - b
            f = np.linalg.norm(hold) ** 2

            if f <= cval - options["tau"] * derivative or nls >= options["nls"]:
                break
            options["tau"] = options["eta"] * options["tau"]
            nls += 1

        grad = np.dot(hold, a.T)
        norm_grad = np.linalg.norm(grad)

        # Calculate the Barzilai-Borwein step-size.
        s = x - x_old
        norm_s = np.linalg.norm(s)
        x_diff = norm_s / np.linalg.norm(x)
        f_diff = abs((f - f_old) / (f_old + 1))

        y = grad - grad_old
        sy = abs(np.sum(np.multiply(s, y)))
        if i % 2 == 0:
            tau = (norm_s**2) / sy
        else:
            tau = sy / (np.linalg.norm(y) ** 2)
        # Bounding the step size between 1e-20 and 1e20
        # so that it is neither too low or too high.
        tau = max(min(tau, 1e20), 1e-20)

        # Stopping conditions.
        if norm_s < options["x_tol"] or (
            x_diff < 100 * options["f_tol"] and f_diff < options["f_tol"]
        ):
            if i <= 2:
                options["f_tol"] /= 10
                options["x_tol"] /= 10
            else:
                break

        q_old = q
        q = options["gamma"] * q_old + 1
        cval = (options["gamma"] * q_old * cval + f) / q

    # error = ||SAT - B||_F
    return ProcrustesResult(
        new_a=a,
        new_b=b,
        error=compute_error(a=a, b=b, t=np.eye(m), s=x),
        t=np.eye(m),
        s=x,
    )


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
        error=compute_error(a=a, b=b, s=p, t=np.eye(a.shape[1])),
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
        error.append(compute_error(a=a, b=b, s=np.dot(e.T, e), t=np.eye(a.shape[1])))

        # Check if positive semidefinite or if the algorithm has converged.
        if (
            np.all(np.linalg.eigvals(le) >= 0)
            or abs(sqrt(error[-1]) - sqrt(error[-2])) < 1e-5
        ):
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


def _psd_proj(arr: np.ndarray, do_cholesky: bool = True) -> np.ndarray:
    r"""
    Return the symmetric positive semi-definite matrix nearest to a given matrix.

    Parameters
    ----------
    arr : np.ndarray
        The input matrix.

    do_cholesky : bool
        Parameter to decide whether or not to perform
        Cholesky decomposition.

    Returns
    -------
    np.ndarray
        The nearest symmetric positive semi-definite matrix.
    """
    arr = (arr + arr.T) / 2
    if np.isnan(arr).any() or np.isposinf(abs(arr)).any():
        raise ValueError("Array has atleast one entry which is NaN or infinite.")

    try:
        assert do_cholesky
        _ = np.linalg.cholesky(arr)
        return arr
    except np.linalg.LinAlgError:
        return _make_positive(arr)


def _init_procustes_projgrad(
    a: np.ndarray,
    b: np.ndarray,
    choice: int = 0
) -> np.ndarray:
    r"""Return the starting point of the transformation S of the projection gradient algorithm."""

    n, _ = a.shape

    # We will find two choices S1 and S2 and return the one that gives a lower error in the
    # minimization function

    # Finding S1
    s1t = np.linalg.lstsq(a.conj().T, b.conj().T, rcond=1)[0]
    s1 = s1t.conj().T
    s1 = _psd_proj(s1)
    e1 = np.linalg.norm(s1@a-b, 'fro')

    # Finding S2
    eps = 1e-6
    s2 = np.zeros((n, n))
    for i in range(n):
        s2[i, i] = max(0, (a[i, :]@b[i, :].conj().T) / (np.linalg.norm(a[i, :])**2 + eps))
        # Adding eps to avoid 0 division
    e2 = np.linalg.norm(s2@a-b, "fro")

    s_init = None

    if e1 < e2 or choice == 1:
        s_init = s1
    elif e2 < e1 or choice == 2:
        s_init = s2

    return s_init
