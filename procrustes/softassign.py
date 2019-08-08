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

import warnings

import numpy as np

from procrustes.permutation import permutation
from procrustes.utils import _get_input_arrays, eigendecomposition
from procrustes.utils import error

__all__ = [
    "softassign",
]


def softassign(A, B, remove_zero_col=True, remove_zero_row=True,
               pad_mode='row-col', translate=False, scale=False,
               check_finite=True, iteration_r=500, iteration_s=500,
               beta_r=1.075, tol_r=1.0e-8, tol_s=1.0e-8,
               epsilon_gamma=0.01, idx_stop=10, beta_0=1.e-5,
               beta_f=1.e4, Mai_guess=None, return_guess=False):
    r"""
    Parameters
    ----------
    A : ndarray
        The 2d-array :math:`\mathbf{A}_{m \times n}` which
        is going to be transformed.
    B : ndarray
        The 2d-array :math:`\mathbf{B}_{m \times n}` representing
        the reference.
    remove_zero_col : bool, optional
        If True, the zero columns on the right side will be removed
        Default=True.
    remove_zero_row : bool, optional
        If True, the zero rows on the top will be removed.
        Default=True.
    pad_mode : str, optional
      Zero padding mode when the sizes of two arrays differ. Defaul
      'row': The array with fewer rows is padded with zero rows so
           number of rows.
      'col': The array with fewer columns is padded with zero colum
           same number of columns.
      'row-col': The array with fewer rows is padded with zero rows
           columns is padded with zero columns, so that both have t
           This does not necessarily result in square arrays.
      'square': The arrays are padded with zero rows and zero colum
           squared arrays. The dimension of square array is specifi
           dimension, i.e. :math:`\text{max}(n_a, m_a, n_b, m_b)`.'
    translate : bool, optional
        If True, both arrays are translated to be centered at origi
        Default=False.
    scale : bool, optional
        If True, both arrays are column normalized to unity. Defaul
    check_finite : bool, optional
        If true, convert the input to an array, checking for NaNs o
        Default=True.
    iteration_r : int, optional
        Number of iterations in relaxation step. Default=500.
    iteration_s : int, optional
        Number of iterations in Sinkhorn step. Default=500.
    beta_r : float, optional
        Annealing rate which should greater than 1. Default=1.075.
    tol_r : float, optional
        The tolerance value used for relaxation. Default=1.e-8.
    tol_s : float, optional
        The tolerance value used for sinkhorn. Default=1.e-8.
    epsilon_gamma : float, optional
        Small quantity which is required to compute gamma. Default=
    idx_stop : int, optional
        Number of running steps after the calculation converges in
        relaxation step. Default=10.
    beta_0 : float, optional
        Initial inverse temperature. Default=1.e-5.
    beta_f : float, optional
        Final inverse temperatue. Default=1.e4.
    Mai_guess : numpy.array, optional
        Initial guess for determinstic annealing. If None, the function
        will use a random guess. Default=None.
    return_guess : bool, optional
        Whether to return the :math:`\mathbf{M}_{ai}` from the last step
        if the calculations failed. Default=False.

    Returns
    -------
    A : ndarray
        The transformed ndarray A.
    B : ndarray
        The transformed ndarray B.
    M_ai : ndarray
        The optimum permutation transformation matrix.
    e_opt : float
        Two-sided Procrustes error.

    Notes
    -----
    Quadratic assignment problem (QAP) has played a very special bu
    fundamental role in combinatorial optimization problems. The pr
    be defined as a optimization problem to minimize the cost to as
    of facilities to a set of locations. The cost is a function of
    between the facilities and the geographical distances among var
    facilities.
    The objective function (also named loss function in machine lea
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
    energy function which comes along with a self-amplification ter
    `\gamma`, two Lagrange parameters :math:`\mu` and :math:`\nu` f
    constrained optimization and :math:`M_{ai} \log{M_{ai}}` server
    barrier function which ensures positivity of :math:`M_{ai}`. Th
    inverse temperature :math:`\beta` is a deterministic annealing
    control parameter. More detailed information about the algorith
    referred to Rangarajan's paper.

    References
    ----------
    .. [1] Rangarajan, Anand and Yuille, Alan L and Gold, Steven an
       Mjolsness, Eric, "A convergence proof for the softassign qua
       assignment algorithm" Advances in Neural Information Process
       Systems, page 620-626, 1997.

    Examples
    --------
    >>> import numpy as np
    >>> array_a = np.array([[4, 5, 3, 3], [5, 7, 3, 5],
                            [3, 3, 2, 2], [3, 5, 2, 5]])
        # define a random matrix
    >>> perm = np.array([[0., 0., 1., 0.], [1., 0., 0., 0.],
                         [0., 0., 0., 1.], [0., 1., 0., 0.]])
        # define array_b by permuting array_a
    >>> array_b = np.dot(perm.T, np.dot(array_a, perm))
    >>> new_a, new_b, M_ai, e_opt = softassign(array_a, array_b,
                                               remove_zero_col=Fals
                                               remove_zero_row=Fals
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
    gamma = _compute_gamma(C, N, epsilon_gamma)
    # Initialization of M_ai
    # check shape of Mai_guess
    if Mai_guess is not None:
        if Mai_guess.shape[0] == N and Mai_guess.shape[1] == N:
            M_relax_old = Mai_guess
        else:
            warning_info = "The shape of Mai_guess does not match (" \
                           + str(N) + "," + str(N) + "). " \
                           + "Use random initial guess instead."
            warnings.warn(warning_info)
            M_relax_old = 1 / N + np.random.rand(N, N)
    else:
        M_relax_old = 1 / N + np.random.rand(N, N)
    beta = beta_0
    step_r = 0
    # step to control when to stop the calculation
    idx = 0
    delta_M_relax = np.inf
    # Deterministic annealing
    while beta < beta_f:
        # relaxation
        while np.amax(np.abs(delta_M_relax)) > tol_r and step_r < iteration_r:
            step_r += 1
            if step_r == iteration_r:
                print('Maximum iteration in relaxation stage reached!')
            # Compute Q in relaxation step
            Q = np.einsum('aibj,bj->ai', C_tensor, M_relax_old)
            Q += gamma * M_relax_old
            # soft_assign
            M_soft = np.exp(beta * Q)
            # Sinkhorn initial value
            M_sink_old = M_soft
            # step_s for Shinkhorn balancing
            step_s = 0
            # delta_M_sink = (M_relax_new - M_relax_old)/N
            delta_M_sink = np.inf
            # while np.amax(np.abs(delta_M_sink)) > tol and step_s < iteration_s:
            while np.amax(np.abs(delta_M_sink)) > tol_s and step_s < iteration_s:
                step_s += 1
                # Row normalization
                M_sink_new = M_sink_old / M_sink_old.sum(axis=1, keepdims=1)
                # Column normalization
                M_sink_new = M_sink_new / M_sink_new.sum(axis=0, keepdims=1)
                # Compute the delata_M_sink
                delta_M_sink = M_sink_new - M_sink_old
                # Update M_sink_old
                M_sink_old = M_sink_new
                # tol_s = max(np.amax(np.abs(delta_M_sink))/20, tol_s)
            # use the result of Sinkhorn to update M_ai for relaxation
            M_relax_new = M_sink_new
            # Compute the delta_M_relax
            delta_M_relax = M_relax_new - M_relax_old
            # tol_r = max(np.amax(np.abs(delta_M_relax))/20, tol_r)
            # Update the M_relax
            M_relax_old = M_relax_new

        beta = beta_r * beta

        if np.isnan(M_relax_new).any() or np.isinf(M_relax_new).any():
            break
        else:
            # keep a temporary value for the case of nan
            M_tmp = M_relax_new

        # if np.amax(np.abs(delta_M_relax)) < tol_r:
        #     idx += 1
        #     if idx == idx_stop:
        #         break
    #
    if return_guess:
        try:
            _, _, M_ai, _ = permutation(np.eye(M_relax_new.shape[0]), M_relax_new)
            final_guess = M_relax_new
        except ValueError:
            print("M_ai cannot contain infs. Please try to decrease annealing rate (beta_r) or "
                  "increase the temperature (beta_f).")
            _, _, M_ai, _ = permutation(np.eye(M_tmp.shape[0]), M_tmp)
            final_guess = M_tmp
        # Compute error
        e_opt = error(A, B, M_ai, M_ai)
        return A, B, M_ai, e_opt, final_guess
    else:
        try:
            _, _, M_ai, _ = permutation(np.eye(M_relax_new.shape[0]), M_relax_new)
        except ValueError:
            print("M_ai cannot contain infs. Please try to decrease annealing rate (beta_r) or "
                  "increase the temperature (beta_f).")
            _, _, M_ai, _ = permutation(np.eye(M_tmp.shape[0]), M_tmp)

        # Compute error
        e_opt = error(A, B, M_ai, M_ai)
        print(e_opt)
        return A, B, M_ai, e_opt


def _compute_gamma(C, N, epsilon_gamma):
    r"""Compute gamma for relaxation."""
    r = np.eye(N) - np.ones(N) / N
    R = np.kron(r, r)
    RCR = np.dot(R, np.dot(C, R))
    # gamma = -s[-1] + epsilon_gamma
    s = np.linalg.eigvalsh(RCR)
    # get index of sorted eigenvalues from largest to smallest
    idx = s.argsort()[::-1]
    gamma = -s[idx][-1] + epsilon_gamma
    return gamma
