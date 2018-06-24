# -*- coding: utf-8 -*-
# Procrustes is a collection of interpretive chemical tools for
# analyzing outputs of the quantum chemistry calculations.
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
"""
Two-Sided Orthogonal Procrustes Module.
"""


import numpy as np

from itertools import product

from procrustes.base import Procrustes
from procrustes.orthogonal import orthogonal
from procrustes.utils import _get_input_arrays, _check_rank, error
from procrustes.utils import singular_value_decomposition, eigendecomposition


def orthogonal_2sided(A, B, remove_zero_col=True, remove_zero_row=True,
                      translate=False, scale=False, single_transform=True,
                      mode="approx", check_finite=True, tol=1.0e-8):
    r"""
    Two-Sided Orthogonal Procrustes.

    Parameters
    ----------
    A : ndarray
        The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
    B : ndarray
        The 2d-array :math:`\mathbf{B}_{m \times n}` representing the reference array.
    remove_zero_col : bool, optional
        If True, the zero columns on the right side will be removed.
        Default= True.
    remove_zero_row : bool, optional
        If True, the zero rows on the top will be removed.
        Default= True.
    translate : bool, optional
        If True, both arrays are translated to be centered at origin.
        Default=False.
    scale : bool, optional
        If True, both arrays are column normalized to unity. Default=False.
    single_transform : bool
        If True, two-sided orthogonal Procrustes with one transformation
        will be performed. Default=False.
    mode : string, optional
        The scheme to solve for unitary transformation.
        Options: 'exact' and 'approx'. Default="approx".
    check_finite : bool, optional
        If true, convert the input to an array, checking for NaNs or Infs.
        Default=True.
    tol : float, optional
        The tolerance value used for 'approx' mode. Default=1.e-8.

    Returns
    -------
    A : ndarray
        The transformed ndarray A.
    B : ndarray
        The transformed ndarray B.
    U_opt1 : ndarray
        The optimal orthogonal left-multiplying transformation ndarray if "single_transform=True".
    U_opt2 : ndarray
        The second transformation ndarray if "single_transform=True".
    U_opt : ndarray
        The transformation ndarray if "single_transform=False".
    e_opt : float
        The single- or double- sided orthogonal Procrustes error.

    Raises
    ------
    ValueError
        When input array `A` or `B` is not symmetric.
    numpy.linalg.LinAlgError
        If array `A` or `B` is not diagonalizable when `mode='umeyama'` or
        `mode='umeyama_approx'`.
    ValueError
        If the mode is not 'exact' or 'approx' when `single_transform=True`.

    Notes
    -----
    **Two-Sided Orthogonal Procrustes:**

    Given matrix :math:`\mathbf{A}_{m \times n}` and a reference :math:`\mathbf{B}_{m \times n}`,
    find two unitary/orthogonal transformation of :math:`\mathbf{A}_{m \times n}` that makes it as
    as close as possible to :math:`\mathbf{B}_{m \times n}`. I.e.,

    .. math::
          \underbrace{\text{min}}_{\left\{ {\mathbf{U}_1 \atop \mathbf{U}_2} \left|
            {\mathbf{U}_1^{-1} = \mathbf{U}_1^\dagger \atop \mathbf{U}_2^{-1} =
            \mathbf{U}_2^\dagger} \right. \right\}}
            \|\mathbf{U}_1^\dagger \mathbf{A} \mathbf{U}_2 - \mathbf{B}\|_{F}^2
       &= \underbrace{\text{min}}_{\left\{ {\mathbf{U}_1 \atop \mathbf{U}_2} \left|
             {\mathbf{U}_1^{-1} = \mathbf{U}_1^\dagger \atop \mathbf{U}_2^{-1} =
             \mathbf{U}_2^\dagger} \right. \right\}}
        \text{Tr}\left[\left(\mathbf{U}_1^\dagger\mathbf{A}\mathbf{U}_2 - \mathbf{B} \right)^\dagger
                   \left(\mathbf{U}_1^\dagger\mathbf{A}\mathbf{U}_2 - \mathbf{B} \right)\right] \\
       &= \underbrace{\text{min}}_{\left\{ {\mathbf{U}_1 \atop \mathbf{U}_2} \left|
             {\mathbf{U}_1^{-1} = \mathbf{U}_1^\dagger \atop \mathbf{U}_2^{-1} =
             \mathbf{U}_2^\dagger} \right. \right\}}
          \text{Tr}\left[\mathbf{U}_2^\dagger\mathbf{A}^\dagger\mathbf{U}_1\mathbf{B} \right]

    We can get the solution by taking singular value decomposition of the matrices. Having,

    .. math::
       \mathbf{A} = \mathbf{U}_A \mathbf{\Sigma}_A \mathbf{V}_A^\dagger \\
       \mathbf{B} = \mathbf{U}_B \mathbf{\Sigma}_B \mathbf{V}_B^\dagger

    The transformation is foubd by,

    .. math::
       \mathbf{U}_1 = \mathbf{U}_A \mathbf{U}_B^\dagger \\
       \mathbf{U}_2 = \mathbf{V}_B \mathbf{V}_B^\dagger

    **Two-Sided Orthogonal Procrustes with Single-Transformation:**

    Given matrix :math:`\mathbf{A}_{n \times n}` and a reference :math:`\mathbf{B}_{n \times n}`,
    find one unitary/orthogonal transformation matrix :math:`\mathbf{U}_{n \times n}` that makes
    :math:`\mathbf{A}_{n \times n}` as close as possible to :math:`\mathbf{B}_{n \times n}`. I.e.,

    .. math::
       \underbrace{\min}_{\left\{\mathbf{U} | \mathbf{U}^{-1} = {\mathbf{U}}^\dagger \right\}}
                          \|\mathbf{U}^\dagger\mathbf{A}\mathbf{U} - \mathbf{B}\|_{F}^2
       &= \underbrace{\text{min}}_{\left\{\mathbf{U} | \mathbf{U}^{-1} = {\mathbf{U}}^\dagger
                                   \right\}}
          \text{Tr}\left[\left(\mathbf{U}^\dagger\mathbf{A}\mathbf{U} - \mathbf{B} \right)^\dagger
                         \left(\mathbf{U}^\dagger\mathbf{A}\mathbf{U} - \mathbf{B} \right)\right] \\
       &= \underbrace{\text{max}}_{\left\{\mathbf{U} | \mathbf{U}^{-1} = {\mathbf{U}}^\dagger
                                   \right\}}
          \text{Tr}\left[\mathbf{U}^\dagger\mathbf{A}^\dagger\mathbf{U}\mathbf{B} \right]

    Taking the eigenvalue decomposition of the matrices:

    .. math::
       \mathbf{A} = \mathbf{U}_A \mathbf{\Lambda}_A \mathbf{U}_A^\dagger \\
       \mathbf{B} = \mathbf{U}_B \mathbf{\Lambda}_B \mathbf{U}_B^\dagger

    the solution is obtained by,

    .. math::
       \mathbf{U} = \mathbf{U}_A \mathbf{S} \mathbf{U}_A^\dagger

    where :math:`\mathbf{S}` is a diagonal matrix for which every diagonal element is
    :math:`\pm{1}`,

    .. math::
       \mathbf{S} =
       \begin{bmatrix}
        { \pm 1} & 0       &\cdots &0 \\
        0        &{ \pm 1} &\ddots &\vdots \\
        \vdots   &\ddots   &\ddots &0\\
        0        &\cdots   &0      &{ \pm 1}
       \end{bmatrix}

    Finding the best choice of :math:`\mathbf{S}` requires :math:`2^n` trial-and-error tests.
    This is called the ``exact`` scheme for solving the probelm.

    A heuristic, due to Umeyama, is to take the element-wise absolute value of the elements
    of the unitary transformations,

    .. math::
       \mathbf{U}_\text{Umeyama} = \text{abs}(\mathbf{U}_A) \cdot \text{abs}(\mathbf{U}_B^\dagger)

    This is not actually a unitary matrix. But we can use the orthogonal procrustes problem
    to find the closest unitray matrix (i.e., the closest matrix that is unitarily equivalent
    to the identity matrix),

    .. math::
       \underbrace{\min}_{\left\{\mathbf{U} | \mathbf{U}^{-1} = {\mathbf{U}}^\dagger \right\}}
                          \|\mathbf{I}\mathbf{U} -  \mathbf{U}_\text{Umeyama}\|_{F}^2
       &= \underbrace{\text{min}}_{\left\{\mathbf{U} | \mathbf{U}^{-1} = {\mathbf{U}}^\dagger
                                   \right\}}
          \text{Tr}\left[\left(\mathbf{U} - \mathbf{U}_\text{Umeyama} \right)^\dagger
                         \left(\mathbf{U} - \mathbf{U}_\text{Umeyama} \right)\right] \\
       &= \underbrace{\text{max}}_{\left\{\mathbf{U} | \mathbf{U}^{-1} = {\mathbf{U}}^\dagger
                                   \right\}}
          \text{Tr}\left[\mathbf{U}^\dagger \mathbf{U}_\text{Umeyama} \right]

    considering the singular value decomposition of :math:`\mathbf{U}_\text{Umeyama}`,

    .. math::
       \mathbf{U}_\text{Umeyama} =
            \tilde{\mathbf{U}} \tilde{\mathbf{\Sigma}} \tilde{\mathbf{V}}^\dagger

    the solution is give by,

    .. math::
       \mathbf{U}_\text{Umeyama}^\text{approx} = \tilde{\mathbf{U}} \tilde{\mathbf{V}}^\dagger

    This is called the ``approx`` scheme for solving the probelm.

    Examples
    --------
    >>> import numpy as np
    >>> array_a = np.array([[30, 33, 20], [33, 53, 43], [20, 43, 46]])
    >>> array_b = np.array([
        [ 22.78131838, -0.58896768,-43.00635291, 0., 0.],
        [ -0.58896768, 16.77132475,  0.24289990, 0., 0.],
        [-43.00635291,  0.2428999 , 89.44735687, 0., 0.],
        [  0.        ,  0.        ,  0.        , 0., 0.]])
    >>> new_a, new_b, array_u, error_opt = orthogonal_2sided(
            array_a, array_b, single_transform=True,
            remove_zero_col=True, remove_zero_rwo=True, mode='exact')
    >>> array_u
    array([[ 0.25116633,  0.76371527,  0.59468855],
        [-0.95144277,  0.08183302,  0.29674906],
        [ 0.17796663, -0.64034549,  0.74718507]])
    >>> error_opt
    1.9646186414076689e-26

    """
    # Check symmetry if single_transform=True
    if single_transform:
        if (not np.allclose(A.T, A)):
            raise ValueError('Array A should be symmetric.')
        if (not np.allclose(B.T, B)):
            raise ValueError('Array B should be symmetric.')
        # Check if matrix A and B are diagonalizable
        try:
            _check_rank(A)
            _check_rank(B)
        except:
            raise np.linalg.LinAlgError("Matrix cannot be diagonalized.")
    # Check inputs
    A, B = _get_input_arrays(A, B, remove_zero_col, remove_zero_row,
                             translate, scale, check_finite)
    # Convert mode strings into lowercase
    mode = mode.lower()
    # Do single-transformation computation if requested
    if single_transform:
        # check A and B are symmetric
        if mode == "approx":
            U = _2sided_1trans_approx(A, B, tol)
            e_opt = error(A, B, U, U)
        elif mode == "exact":
            U = _2sided_1trans_exact(A, B, tol)
            e_opt = error(A, B, U, U)
        else:
            raise ValueError("Invalid mode argument (use 'exact' or 'approx')")
        return A, B, U, e_opt
    # Do regular two-sided orthogonal Procrustes calculations
    else:
        U_opt1, U_opt2 = _2sided(A, B)
        e_opt = error(A, B, U_opt1, U_opt2)
        return A, B, U_opt1, U_opt2, e_opt


def _2sided(A, B):
    r"""
    """
    UA, _, VTA = np.linalg.svd(A)
    UB, _, VTB = np.linalg.svd(B)
    U_opt1 = np.dot(UA, UB.T)
    U_opt2 = np.dot(VTA.T, VTB)
    return U_opt1, U_opt2


def _2sided_1trans_approx(A, B, tol):
    r"""
    """
    # Calculate the eigenvalue decomposition of A and B
    _, UA = eigendecomposition(A, permute_rows=True)
    _, UB = eigendecomposition(B, permute_rows=True)
    # compute U_umeyama
    U_umeyama = np.dot(np.abs(UA), np.abs(UB.T))
    # compute the closet unitary transformation to u_umeyama
    I = np.eye(U_umeyama.shape[0], dtype=U_umeyama.dtype)
    _, _, U_ortho, _ = orthogonal(I, U_umeyama)
    U_ortho[np.abs(U_ortho) < tol] = 0
    return U_ortho


def _2sided_1trans_exact(A, B, tol):
    r"""
    """
    _, UA = eigendecomposition(A)
    _, UB = eigendecomposition(B)
    # 2^n trial-and-error test to find optimum S array
    diags = product((-1, 1.), repeat=A.shape[0])
    for index, diag in enumerate(diags):
        if index == 0:
            U_opt = np.dot(np.dot(UA, np.diag(diag)), UB.T)
            e_opt = error(A, B, U_opt, U_opt)
        else:
            # compute trial transformation and error
            U_trial = np.dot(np.dot(UA, np.diag(diag)), UB.T)
            e_trial = error(A, B, U_trial, U_trial)
            if e_trial < e_opt:
                U_opt = U_trial
                e_opt = e_trial
            else:
                pass
        # stop trial-and-error if error is below threshold
        if e_opt < tol:
            break
    return U_opt
