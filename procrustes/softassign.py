# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2024 The QC-Devs Community
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
"""The Softassign Procrustes Module."""

import warnings
from copy import deepcopy
from typing import Optional

import numpy as np

from procrustes.kopt import kopt_heuristic_single
from procrustes.permutation import permutation
from procrustes.utils import ProcrustesResult, compute_error, setup_input_arrays

__all__ = [
    "softassign",
]


def softassign(
    a: np.ndarray,
    b: np.ndarray,
    pad: bool = True,
    translate: bool = False,
    scale: bool = False,
    unpad_col: bool = False,
    unpad_row: bool = False,
    check_finite: bool = True,
    weight: Optional[np.ndarray] = None,
    iteration_soft: int = 50,
    iteration_sink: int = 200,
    beta_r: float = 1.10,
    beta_f: float = 1.0e5,
    epsilon: float = 0.05,
    epsilon_soft: float = 1.0e-3,
    epsilon_sink: float = 1.0e-3,
    k: float = 0.15,
    gamma_scaler: float = 1.01,
    n_stop: int = 3,
    adapted: bool = True,
    beta_0: Optional[float] = None,
    m_guess: Optional[float] = None,
    iteration_anneal: Optional[int] = None,
    kopt: bool = False,
    kopt_k: int = 3,
) -> ProcrustesResult:
    r"""
    Find the transformation matrix for 2-sided permutation Procrustes with softassign algorithm.

    Parameters
    ----------
    a : ndarray
        The 2D-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
    b : ndarray
        The 2D-array :math:`\mathbf{B}_{m \times n}` representing the reference.
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
        If true, convert the input to an array, checking for NaNs or Infs. Default=True.
    weight : ndarray, optional
        The 1D-array representing the weights of each row of :math:`\mathbf{A}`. This defines the
        elements of the diagonal matrix :math:`\mathbf{W}` that is multiplied by :math:`\mathbf{A}`
        matrix, i.e., :math:`\mathbf{A} \rightarrow \mathbf{WA}`.
    iteration_soft : int, optional
        Number of iterations for softassign loop.
    iteration_sink : int, optional
        Number of iterations for Sinkhorn loop.
    beta_r : float, optional
        Annealing rate which should greater than 1.
    beta_f : float, optional
        The final inverse temperature.
    epsilon : float, optional
        The tolerance value for annealing loop.
    epsilon_soft : float, optional
        The tolerance value used for softassign.
    epsilon_sink : float, optional
        The tolerance value used for Sinkhorn loop. If adapted version is used, it will use the
        adapted tolerance value for Sinkhorn instead.
    k : float, optional
        This parameter controls how much tighter the coverage threshold for the interior loop should
        be than the coverage threshold for the loops outside. It has be be within the integral
        :math:`(0,1)`.
    gamma_scaler : float, optional
        This parameter ensures the quadratic cost function including  self-amplification positive
        define.
    n_stop : int, optional
        Number of running steps after the calculation converges in the relaxation procedure.
    adapted : bool, optional
        If adapted, this function will use the tighter covergence threshold for the interior loops.
    beta_0 : float, optional
        Initial inverse temperature.
    beta_f : float, optional
        Final inverse temperature.
    m_guess : ndarray, optional
        The initial guess of the doubly-stochastic matrix.
    iteration_anneal : int, optional
        Number of iterations for annealing loop.
    kopt : bool, optional
        If True, the k_opt heuristic search will be performed.
    kopt_k : int, optional
        Defines the oder of k-opt heuristic local search. For example, kopt_k=3 leads to a local
        search of 3 items and kopt_k=2 only searches for two items locally.
    weight : ndarray, optional
        The weighting matrix.

    Returns
    -------
    res : ProcrustesResult
        The Procrustes result represented as a class:`utils.ProcrustesResult` object.

    Notes
    -----
    Quadratic assignment problem (QAP) has played a very special but fundamental role in
    combinatorial optimization problems. The problem can be defined as a optimization problem to
    minimize the cost to assign a set of facilities to a set of locations. The cost is a function
    of the flow between the facilities and the geographical distances among various facilities.

    The objective function (also named loss function in machine learning) is
    defined as

    .. math::
        E_{qap}(M, \mu, \nu) =
            - \frac{1}{2}\Sigma_{aibj}C_{ai;bj}M_{ai}M_{bj}
            + \Sigma_{a}{\mu}_a (\Sigma_i M_{ai} -1) \\
            + \Sigma_i {\nu}_i (\Sigma_i M_{ai} -1)
            - \frac{\gamma}{2}\Sigma_{ai} {M_{ai}}^2
            + \frac{1}{\beta} \Sigma_{ai} M_{ai}\log{M_{ai}}

    where :math:`C_{ai,bj}` is the benefit matrix, :math:`M` is the
    desired :math:`N \times N` permutation matrix. :math:`E` is the
    energy function which comes along with a self-amplification term with
    `\gamma`, two Lagrange parameters :math:`\mu` and :math:`\nu` for
    constrained optimization and :math:`M_{ai} \log{M_{ai}}` servers as a
    barrier function which ensures positivity of :math:`M_{ai}`. The
    inverse temperature :math:`\beta` is a deterministic annealing
    control parameter.

    Examples
    --------
    >>> import numpy as np
    >>> array_a = np.array([[4, 5, 3, 3], [5, 7, 3, 5],
    ...                     [3, 3, 2, 2], [3, 5, 2, 5]])
        # define a random matrix
    >>> perm = np.array([[0., 0., 1., 0.], [1., 0., 0., 0.],
    ...                  [0., 0., 0., 1.], [0., 1., 0., 0.]])
        # define b by permuting array_a
    >>> b = np.dot(perm.T, np.dot(a, perm))
    >>> new_a, new_b, M_ai, error = softassign(a,b,unpad_col=False,unpad_row=False)
    >>> M_ai # the permutation matrix
    array([[0., 0., 1., 0.],
           [1., 0., 0., 0.],
           [0., 0., 0., 1.],
           [0., 1., 0., 0.]])
    >>> error # the error
    0.0

    """
    # pylint: disable-msg=too-many-arguments
    # pylint: disable-msg=too-many-branches
    # todo: add linear_cost_func with default value 0
    # Check beta_r
    if beta_r <= 1:
        raise ValueError("Argument beta_r cannot be less than 1.")
    new_a, new_b = setup_input_arrays(
        a, b, unpad_col, unpad_row, pad, translate, scale, check_finite, weight
    )

    # Check that A & B are square and that they match each other.
    if new_a.shape[0] != new_a.shape[1]:
        raise ValueError(
            f"Matrix A should be square but A.shape={new_a.shape}"
            "Check pad, unpad_col, and unpad_row arguments."
        )
    if new_b.shape[0] != new_b.shape[1]:
        raise ValueError(
            f"Matrix B should be square but B.shape={new_b.shape}"
            "Check pad, unpad_col, and unpad_row arguments."
        )
    if new_a.shape != new_b.shape:
        raise ValueError(
            f"New matrix A {new_a.shape} should match the new" f" matrix B shape {new_b.shape}."
        )

    # Initialization
    # Compute the benefit matrix
    array_c = np.kron(new_a, new_b)
    # Get the shape of A (B and the permutation matrix as well)
    row_num = new_a.shape[0]
    c_tensor = array_c.reshape((row_num, row_num, row_num, row_num))
    # Compute the beta_0
    gamma = _compute_gamma(array_c, row_num, gamma_scaler)
    if beta_0 is None:
        c_gamma = array_c + gamma * (np.identity(row_num * row_num))
        eival_gamma = np.amax(np.abs(np.linalg.eigvalsh(c_gamma)))
        beta_0 = gamma_scaler * max(1.0e-10, eival_gamma / row_num)
        beta_0 = 1 / beta_0
    else:
        beta_0 *= row_num
    beta = beta_0

    # We will use iteration_anneal if provided even if the final inverse temperature is specified
    # iteration_anneal is not None, beta_f can be None or not
    if iteration_anneal is not None:
        beta_f = beta_0 * np.power(beta_r, iteration_anneal) * row_num
    # iteration_anneal is None and beta_f is not None
    elif iteration_anneal is None and beta_f is not None:
        beta_f *= row_num
    # Both iteration_anneal and beta_f are None
    else:
        raise ValueError(
            "We must specify at least one of iteration_anneal and beta_f and "
            "specify only one is recommended."
        )
    # Initialization of m_ai
    # check shape of m_guess
    if m_guess is not None:
        if np.any(m_guess < 0):
            raise ValueError(
                "The initial guess of permutation matrix cannot contain any negative values."
            )
        if m_guess.shape[0] == row_num and m_guess.shape[1] == row_num:
            array_m = m_guess
        else:
            warnings.warn(
                f"The shape of m_guess does not match ({row_num}, {row_num})."
                "Use random initial guess instead."
            )
            array_m = np.abs(np.random.normal(loc=1.0, scale=0.1, size=(row_num, row_num)))
    else:
        # m_relax_old = 1 / N + np.random.rand(N, N)
        array_m = np.abs(np.random.normal(loc=1.0, scale=0.1, size=(row_num, row_num)))
    array_m[array_m < 0] = 0
    array_m = array_m / row_num

    nochange = 0
    if adapted:
        epsilon_sink = epsilon_soft * k
    while beta < beta_f:
        # relaxation
        m_old_beta = deepcopy(array_m)
        # softassign loop
        for _ in np.arange(iteration_soft):
            m_old_soft = deepcopy(array_m)
            # Compute Z in relaxation step
            # C_gamma_tensor = C_gamma.reshape(N, N, N, N)
            # Z = -np.einsum('ijkl,jl->ik', C_gamma_tensor, M)
            # Z -= linear_cost_func
            array_z = np.einsum("aibj,bj->ai", c_tensor, array_m)
            array_z += gamma * array_m
            # soft_assign
            array_m = np.exp(beta * array_z)
            # Sinkhorn loop
            for _ in np.arange(iteration_sink):
                # Row normalization
                array_m = array_m / array_m.sum(axis=1, keepdims=1)
                # Column normalization
                array_m = array_m / array_m.sum(axis=0, keepdims=1)
                # Compute the delata_M_sink
                if np.amax(np.abs(array_m.sum(axis=1, keepdims=1) - 1)) < epsilon_sink:
                    array_m = array_m / array_m.sum(axis=1, keepdims=1)
                    break

            change_soft = np.amax(np.abs(array_m - m_old_soft))
            # pylint: disable-msg=no-else-break
            if change_soft < epsilon_soft:
                break
            else:
                if adapted:
                    epsilon_sink = change_soft * k
                else:
                    continue

        change_annealing = np.amax(np.abs(array_m - m_old_beta))
        if change_annealing < epsilon:
            nochange += 1
            if nochange > n_stop:
                break
        else:
            nochange = 0

        beta *= beta_r
        if adapted:
            epsilon_soft = change_soft * k
            epsilon_sink = epsilon_soft * k

    # Compute the error
    array_m = permutation(np.eye(array_m.shape[0]), array_m)["t"]
    # k-opt heuristic
    if kopt:
        fun_error = lambda p: compute_error(new_a, new_b, p, p.T)
        array_m, error = kopt_heuristic_single(fun_error, p0=array_m, k=kopt_k)
    else:
        error = compute_error(new_a, new_b, array_m, array_m.T)
    return ProcrustesResult(error=error, new_a=new_a, new_b=new_b, t=array_m, s=None)


def _compute_gamma(array_c: np.ndarray, row_num: int, gamma_scaler: float) -> float:
    r"""Compute gamma for relaxation."""
    array_r = np.eye(row_num) - np.ones(row_num) / row_num
    big_r = np.kron(array_r, array_r)
    rcr = np.dot(big_r, np.dot(array_c, big_r))
    gamma = np.max(np.abs(np.linalg.eigvalsh(rcr))) * gamma_scaler
    return gamma
