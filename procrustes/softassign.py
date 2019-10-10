# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2018 The Procrustes Development Team
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

import numpy as np

import warnings

from copy import deepcopy

from procrustes.permutation import permutation
from procrustes.utils import _get_input_arrays
from procrustes.utils import error

__all__ = [
    "softassign",
]


def softassign(A, B, iteration_soft=50, iteration_sink=200, linear_cost_func=0, beta_r=1.10,
               beta_f=1.e5, epsilon=0.05, epsilon_soft=1.e-3, epsilon_sink=1.e-3, k=0.15,
               gamma_scaler=1.01, n_stop=3, pad_mode='row-col', remove_zero_col=True,
               remove_zero_row=True, translate=False, scale=False, check_finite=True,
               adapted=True, beta_0=None, M_guess=None, iteration_anneal=None):
    r"""
    Find the transformation matrix for 2-sided permutation Procrustes with softassign algorithm.

    Parameters
    ----------
    A : numpy.ndarray
        The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
    B : numpy.ndarray
        The 2d-array :math:`\mathbf{B}_{m \times n}` representing the reference.
    iteration_soft ： int, optional
        Number of iterations for softassign loop. Default=50.
    iteration_sink ： int, optional
        Number of iterations for Sinkhorn loop. Default=50.
    linear_cost_func :  numpy.ndarray
        Linear cost function. Default=0.
    beta_r : float, optional
        Annealing rate which should greater than 1. Default=1.10.
    beta_f : float, optional
        The final inverse temperature. Default=1.e5.
    epsilon : float, optional
        The tolerance value for annealing loop. Default=0.05.
    epsilon_soft : float, optional
        The tolerance value used for softassign. Default=1.e-3.
    epsilon_sink : float, optional
        The tolerance value used for Sinkhorn loop. If adapted version is used, it will use the
        adapted tolerance value for Sinkhorn instead. Default=1.e-3.
    k : float, optional
        This parameter controls how much tighter the coverage threshold for the interior loop should
        be than the coverage threshold for the loops outside. It has be be within the integral
        :math:`\(0,1\)`. Default=0.15.
    gamma_scaler : float
        This parameter ensures the quadratic cost function including  self-amplification positive
        define. Default=1.01.
    n_stop : int, optional
        Number of running steps after the calculation converges in the relaxation procedure.
        Default=10.
    pad_mode : str, optional
      Zero padding mode when the sizes of two arrays differ. Default='row-col'.
      'row': The array with fewer rows is padded with zero rows so that both have the same
           number of rows.
      'col': The array with fewer columns is padded with zero columns so that both have the
           same number of columns.
      'row-col': The array with fewer rows is padded with zero rows, and the array with fewer
           columns is padded with zero columns, so that both have the same dimensions.
           This does not necessarily result in square arrays.
      'square': The arrays are padded with zero rows and zero columns so that they are both
           squared arrays. The dimension of square array is specified based on the highest
           dimension, i.e. :math:`\text{max}(n_a, m_a, n_b, m_b)`.'
    remove_zero_col : bool, optional
        If True, the zero columns on the right side will be removed.
        Default=True.
    remove_zero_row : bool, optional
        If True, the zero rows on the top will be removed.
        Default=True.
    translate : bool, optional
        If True, both arrays are translated to be centered at origin.
        Default=False.
    scale : bool, optional
        If True, both arrays are column normalized to unity. Default=False.
    check_finite : bool, optional
        If true, convert the input to an array, checking for NaNs or Infs. Default=True.
    adapted : bool, optional
        If adapted, this function will use the tighter covergence threshold for the interior loops.
        Default=True.
    beta_0 : float, optional
        Initial inverse temperature. Default=None.
    beta_f : float, optional
        Final inverse temperature. Default=None.
    M_guess : numpy.ndarray, optional
        The initial guess of the doubly-stochastic matrix. Default=None.
    iteration_anneal : int, optional
        Number of iterations for annealing loop. Default=None.

    Returns
    -------
    A : numpy.ndarray
        The transformed numpy.ndarray A.
    B : numpy.ndarray
        The transformed numpy.ndarray B.
    M_ai : numpy.ndarray
        The optimum permutation transformation matrix.
    e_opt : float
        Two-sided Procrustes error.

    Notes
    -----
    Quadratic assignment problem (QAP) has played a very special but
    fundamental role in combinatorial optimization problems. The problem can
    be defined as a optimization problem to minimize the cost to assign a set
    of facilities to a set of locations. The cost is a function of the flow
    between the facilities and the geographical distances among various
    facilities.

    The objective function (also named loss function in machine learning) is
    defined as [1]_

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
    control parameter. More detailed information about the algorithm can be
    referred to Rangarajan's paper.

    References
    ----------
    .. [1] Rangarajan, Anand and Yuille, Alan L and Gold, Steven and
       Mjolsness, Eric, "A convergence proof for the softassign quadratic
       assignment algorithm", Advances in Neural Information Processing
       Systems, page 620-626, 1997.
    .. [2] Stefan Roth, "Analysis of a Deterministic Annealing Method for Graph Matching and
       Quadratic Assignment", Ph.D. thesis, University of Mannheim, 2001


    Examples
    --------
    >>> import numpy as np
    >>> array_a = np.array([[4, 5, 3, 3], [5, 7, 3, 5],\
                            [3, 3, 2, 2], [3, 5, 2, 5]])
        # define a random matrix
    >>> perm = np.array([[0., 0., 1., 0.], [1., 0., 0., 0.],\
                         [0., 0., 0., 1.], [0., 1., 0., 0.]])
        # define array_b by permuting array_a
    >>> array_b = np.dot(perm.T, np.dot(array_a, perm))
    >>> new_a, new_b, M_ai, e_opt = softassign(array_a, array_b,\
                                               remove_zero_col=False,\
                                               remove_zero_row=False)
    >>> M_ai # the permutation matrix
    array([[0., 0., 1., 0.],
           [1., 0., 0., 0.],
           [0., 0., 0., 1.],
           [0., 1., 0., 0.]])
    >>> e_opt # the error
    0.0

    """
    # Check beta_r
    if beta_r <= 1:
        raise ValueError("Argument beta_r cannot be less than 1.")

    A, B = _get_input_arrays(A, B, remove_zero_col, remove_zero_row,
                             pad_mode, translate, scale, check_finite)
    # Initialization
    # Compute the benefit matrix
    C = np.kron(A, B)
    # Get the shape of A (B and the permutation matrix as well)
    N = A.shape[0]
    C_tensor = C.reshape(N, N, N, N)
    # Compute the beta_0
    gamma = _compute_gamma(C, N, gamma_scaler)
    if beta_0 is None:
        C_gamma = C + gamma * (np.identity(N * N))
        eival_gamma = np.amax(np.abs(np.linalg.eigvalsh(C_gamma)))
        beta_0 = gamma_scaler * max(1.e-10, eival_gamma / N)
        beta_0 = 1 / beta_0
    else:
        beta_0 *= N
    beta = beta_0

    # We will use iteration_anneal if provided even if the final inverse temperature is specified
    # iteration_anneal is not None, beta_f can be None or not
    if iteration_anneal is not None:
        beta_f = beta_0 * np.power(beta_r, iteration_anneal) * N
    # iteration_anneal is None and beta_f is not None
    elif iteration_anneal is None and beta_f is not None:
        beta_f *= N
    # Both iteration_anneal and beta_f are None
    else:
        raise ValueError("We must specify at least one of iteration_anneal and beta_f and "
                         "specify only one is recommended.")

    # Initialization of M_ai
    # check shape of M_guess
    if M_guess is not None:
        if np.any(M_guess < 0):
            raise ValueError(
                "The initial guess of permutation matrix cannot contain any negative values.")
        if M_guess.shape[0] == N and M_guess.shape[1] == N:
            M = M_guess
        else:
            warnings.warn("The shape of M_guess does not match ({}, {})."
                          "Use random initial guess instead.".format(N, N))
            M = np.abs(np.random.normal(loc=1.0, scale=0.1, size=(N, N)))
    else:
        # M_relax_old = 1 / N + np.random.rand(N, N)
        M = np.abs(np.random.normal(loc=1.0, scale=0.1, size=(N, N)))
    M[M < 0] = 0
    M = M / N

    nochange = 0
    if adapted:
        epsilon_sink = epsilon_soft * k
    while beta < beta_f:
        # relaxation
        M_old_beta = deepcopy(M)
        # softassign loop
        for _ in np.arange(iteration_soft):
            M_old_soft = deepcopy(M)
            # Compute Z in relaxation step
            # C_gamma_tensor = C_gamma.reshape(N, N, N, N)
            # Z = -np.einsum('ijkl,jl->ik', C_gamma_tensor, M)
            # Z -= linear_cost_func
            Z = np.einsum('aibj,bj->ai', C_tensor, M)
            Z += gamma * M
            # soft_assign
            M = np.exp(beta * Z)
            # Sinkhorn loop
            for _ in np.arange(iteration_sink):
                # Row normalization
                M = M / M.sum(axis=1, keepdims=1)
                # Column normalization
                M = M / M.sum(axis=0, keepdims=1)
                # Compute the delata_M_sink
                if np.amax(np.abs(M.sum(axis=1, keepdims=1) - 1)) < epsilon_sink:
                    M = M / M.sum(axis=1, keepdims=1)
                    break

            change_soft = np.amax(np.abs(M - M_old_soft))
            if change_soft < epsilon_soft:
                break
            else:
                if adapted:
                    epsilon_sink = change_soft * k
                else:
                    continue

        change_annealing = np.amax(np.abs(M - M_old_beta))
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
    _, _, M, _ = permutation(np.eye(M.shape[0]), M)
    e_opt = error(A, B, M, M)
    return A, B, M, e_opt


def _compute_gamma(C, N, gamma_scaler):
    r"""Compute gamma for relaxation."""
    r = np.eye(N) - np.ones(N) / N
    R = np.kron(r, r)
    RCR = np.dot(R, np.dot(C, R))
    gamma = np.max(np.abs(np.linalg.eigvalsh(RCR))) * gamma_scaler
    return gamma
