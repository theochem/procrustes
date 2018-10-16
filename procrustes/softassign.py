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

from procrustes.permutation import permutation
from procrustes.utils import _get_input_arrays, eigendecomposition
from procrustes.utils import error

__all__ = [
    "softassign",
]


def softassign(A, B, remove_zero_col=True, remove_zero_row=True,
               pad_mode='row-col', translate=False, scale=False,
               check_finite=True, iteration_r=500, iteration_s=500,
               beta_r=1.075, tol=1.0e-8, epsilon_gamma=0.01,
               beta_0=None, beta_f=None):
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
        If True, the zero columns on the right side will be removed.
        Default=True.
    remove_zero_row : bool, optional
        If True, the zero rows on the top will be removed.
        Default=True.
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
    translate : bool, optional
        If True, both arrays are translated to be centered at origin.
        Default=False.
    scale : bool, optional
        If True, both arrays are column normalized to unity. Default=False.
    check_finite : bool, optional
        If true, convert the input to an array, checking for NaNs or Infs.
        Default=True.
    iteration_r : int, optional
        Number of iterations in relaxation step. Default=500.
    iteration_s : int, optional
        Number of iterations in Sinkhorn step. Default=500.
    beta_r : float, optional
        Annealing rate which should greater than 1. Default=1.075.
    tol : float, optional
        The tolerance value used for relaxation and softassign. Default=1.e-8.
    epsilon_gamma : float, optional
        Small quantity which is required to compute gamma. Default=0.01.
    beta_0 : float, optional
        Initial inverse temperature. Default=None.
    beta_f : float, optional
        Final inverse temperatue. Default=None.

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
       assignment algorithm" Advances in Neural Information Processing
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
                                               remove_zero_col=False,
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
        raise ValueError("Argument beta_r cannot be greater than 1.")

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
    if beta_0 == None:
        beta_0 = 1.e-5 * np.sqrt(N * N)
    if beta_f == None:
        beta_f = 1.e4 * np.sqrt(N * N)
    # Initialization of M_ai
    M_ai = 1 / N + np.abs(np.random.rand(N, N))
    beta = beta_0
    M_relax_old = M_ai
    step_r = 0
    # step to control when to stop the calculation
    idx_stop = 0
    # Deterministic annealing
    while beta < beta_f:
        delta_relax = (M_relax_old - 0) / N
        while np.amax(np.abs(delta_relax)) > tol and step_r < iteration_r:
            step_r += 1
            if step_r == iteration_r:
                print('Maximum iteration in relaxation stage reached!')
            # Compute Q in relaxation step
            Q = np.einsum('aibj,bj->ai', C_tensor, M_relax_old)
            Q += gamma * M_relax_old
            # soft_assign
            M_relax_new = np.exp(beta * Q)
            # Sinkhorn
            M_sink_old = M_relax_new
            # step_s for Shinkhorn balancing
            step_s = 0
            # delta_M_sink = (M_relax_new - M_relax_old)/N
            delta_M_sink = (M_sink_old - 0) / N
            while np.amax(np.abs(delta_M_sink)) > tol and step_s < iteration_s:
                step_s += 1
                # Row normalization
                M_sink_new = M_sink_old / M_sink_old.sum(axis=1, keepdims=1)
                # Column normalization
                M_sink_new = M_sink_new / M_sink_new.sum(axis=0, keepdims=1)
                # Compute the delata_M_sink
                delta_M_sink = (M_sink_new - M_sink_old) / N
                # Update M_sink_old
                M_sink_old = M_sink_new
            # use the result of Sinkhorn to update M_ai for
            M_relax_new = M_sink_new
            # Compute the delta_relax
            delta_relax = (M_relax_new - M_relax_old) / N
            # Update the M_relax
            M_relax_old = M_relax_new

        beta = beta_r * beta
        if np.amax(np.abs(delta_relax)) < tol:
            idx_stop += 1
            if idx_stop == 10:
                break

    # Perform clean-up heuristic
    _, _, M_ai, _ = permutation(np.eye(M_relax_new.shape[0]), M_relax_new)
    # Compute error
    e_opt = error(A, B, M_ai, M_ai)
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
