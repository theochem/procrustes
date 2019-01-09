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
"""Permutation Procrustes Module."""

import numpy as np

import itertools as it

from scipy.optimize import linear_sum_assignment

from procrustes.utils import _get_input_arrays, eigendecomposition, error

__all__ = [
    "permutation",
    "permutation_2sided",
    "permutation_2sided_explicit"
]


def permutation(A, B, remove_zero_col=True, remove_zero_row=True,
                pad_mode='row-col', translate=False, scale=False,
                check_finite=True):
    r"""
    Single sided permutation Procrustes.

    Parameters
    ----------
    A : ndarray
        The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
    B : ndarray
        The 2d-array :math:`\mathbf{B}_{m \times n}` representing the reference.
    remove_zero_col : bool, optional
        If True, the zero columns on the right side will be removed. Default= True.
    remove_zero_row : bool, optional
        If True, the zero rows on the top will be removed. Default= True.
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
        If True, both arrays are translated to be centered at origin. Default=False.
    scale : bool, optional
        If True, both arrays are column normalized to unity. Default=False.
    check_finite : bool, optional
        If true, convert the input to an array, checking for NaNs or Infs. Default=True.

    Returns
    -------
    A : ndarray
        The transformed ndarray A.
    B : ndarray
        The transformed ndarray B.
    U_opt : ndarray
        The optimum permutation transformation matrix.
    e_opt : float
        One-sided permutation Procrustes error.

    Notes
    -----
    Given matrix :math:`\mathbf{A}_{n \times n}` and reference :math:`\mathbf{B}_{n \times n}`
    find a permutation of the rows and/or columns of :math:`\mathbf{A}_{n \times n}` that makes
    it as close as possible to :math:`\mathbf{B}_{n \times n}`. I.e.,

    .. math::
       \underbrace{\text{min}}_{\left\{\mathbf{P} \left| {p_{ij} \in \{0, 1\}
                            \atop \sum_{i=1}^n p_{ij} = \sum_{j=1}^n p_{ij} = 1} \right. \right\}}
                            \|\mathbf{A} \mathbf{P} - \mathbf{B}\|_{F}^2
       &= \underbrace{\text{min}}_{\left\{\mathbf{P} \left| {p_{ij} \in \{0, 1\}
                            \atop \sum_{i=1}^n p_{ij} = \sum_{j=1}^n p_{ij} = 1} \right. \right\}}
          \text{Tr}\left[\left(\mathbf{A}\mathbf{P} - \mathbf{B} \right)^\dagger
                   \left(\mathbf{P}^\dagger\mathbf{A}\mathbf{P} - \mathbf{B} \right)\right] \\
       &= \underbrace{\text{max}}_{\left\{\mathbf{P} \left| {p_{ij} \in \{0, 1\}
                            \atop \sum_{i=1}^n p_{ij} = \sum_{j=1}^n p_{ij} = 1} \right. \right\}}
          \text{Tr}\left[\mathbf{P}^\dagger\mathbf{A}^\dagger\mathbf{B} \right]

    Here, :math:`\mathbf{P}_{n \times n}` is the permutation matrix. The solution is to relax the
    problem into a linear programming problem and note that the solution to a linear programming
    problem is always at the boundary of the allowed region, which means that the solution can
    always be written as a permutation matrix,

    .. math::
       \underbrace{\text{max}}_{\left\{\mathbf{P} \left| {p_{ij} \in \{0, 1\}
                   \atop \sum_{i=1}^n p_{ij} = \sum_{j=1}^n p_{ij} = 1} \right. \right\}}
          \text{Tr}\left[\mathbf{P}^\dagger\mathbf{A}^\dagger\mathbf{B} \right] =
       \underbrace{\text{max}}_{\left\{\mathbf{P} \left| {p_{ij} \geq 0
                   \atop \sum_{i=1}^n p_{ij} = \sum_{j=1}^n p_{ij} = 1} \right. \right\}}
          \text{Tr}\left[\mathbf{P}^\dagger\left(\mathbf{A}^\dagger\mathbf{B}\right) \right]

    This is a matching problem and can be solved by the Hungarian method. Note that if
    :math:`\mathbf{A}` and :math:`\mathbf{B}` have different numbers of items, you choose
    the larger matrix as :math:`\mathbf{B}` and then pad :math:`\mathbf{A}` with rows/columns
    of zeros.

    """
    # check inputs
    A, B = _get_input_arrays(A, B, remove_zero_col, remove_zero_row,
                             pad_mode, translate, scale, check_finite)
    # compute permutation Procrustes matrix
    P = np.dot(A.T, B)
    C = np.full(P.shape, np.max(P))
    C -= P
    U = np.zeros(P.shape)
    # set elements to 1 according to Hungarian algorithm (linear_sum_assignment)
    U[linear_sum_assignment(C)] = 1
    e_opt = error(A, B, U)
    return A, B, U, e_opt


def permutation_2sided(A, B, transform_mode='single_undirected',
                       remove_zero_col=True, remove_zero_row=True,
                       pad_mode='row-col', translate=False, scale=False,
                       mode="normal1", check_finite=True, iteration=500,
                       add_noise=False, tol=1.0e-8):
    r"""
    Single sided permutation Procrustes.

    Parameters
    ----------
    A : ndarray
        The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
    B : ndarray
        The 2d-array :math:`\mathbf{B}_{m \times n}` representing the reference.
    transform_mode : str
        If transform_mode='single_undirected', two-sided permutation Procrustes with one transformation will be performed. If
        transform_mode='single_directed', two-sided permutation for directed
        graph matching will be used. Otherwise, transform_mode='double', the
        two-sided permutation Procrustes with two transformations will be
        performed. Default='single_undirected'.
    remove_zero_col : bool, optional
        If True, the zero columns on the right side will be removed. Default= True.
    remove_zero_row : bool, optional
        If True, the zero rows on the top will be removed. Default= True.
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
        If True, both arrays are translated to be centered at origin. Default=False.
    scale : bool, optional
        If True, both arrays are column normalized to unity. Default=False.
    mode : string, optional
        Option for choosing the initial guess methods, including 'normal1',
        'normal2', 'umeyama' and 'umeyama_approx'. 'umeyama_approx' is the
        approximated umeyama method.
    check_finite : bool, optional
        If true, convert the input to an array, checking for NaNs or Infs.
        Default=True.
    iteration : int, optional
        Maximum number for iterations. Default=500.
    add_noise : bool, optional
        Add small noise if the arrays are non-diagonalizable. Default=False.
    tol : float, optional
        The tolerance value used for updating the initial guess. Default=1.e-8

    Returns
    -------
    A : ndarray
        The transformed ndarray A.
    B : ndarray
        The transformed ndarray B.
    U : ndarray
        The optimum permutation transformation matrix.
    V : ndarray
        The optimum permutation transformation matrix.
    e_opt : float
        Two-sided permutation Procrustes error.

    Notes
    -----
    Given matrix :math:`\mathbf{A}_{n \times n}` and a reference :math:`\mathbf{B}_{n \times n}`,
    find a permutation of rows/columns of :math:`\mathbf{A}_{n \times n}` that makes it as close as
    possible to :math:`\mathbf{B}_{n \times n}`. I.e.,

    .. math::
       \underbrace{\text{min}}_{\left\{\mathbf{P} \left| {p_{ij} \in \{0, 1\}
                              \atop \sum_{i=1}^n p_{ij} = \sum_{j=1}^n p_{ij} = 1} \right. \right\}}
                              \|\mathbf{P}^\dagger \mathbf{A} \mathbf{P} - \mathbf{B}\|_{F}^2
       &= \underbrace{\text{min}}_{\left\{\mathbf{P} \left| {p_{ij} \in \{0, 1\}
                              \atop \sum_{i=1}^n p_{ij} = \sum_{j=1}^n p_{ij} = 1} \right. \right\}}
          \text{Tr}\left[\left(\mathbf{P}^\dagger\mathbf{A}\mathbf{P} - \mathbf{B} \right)^\dagger
                   \left(\mathbf{P}^\dagger\mathbf{A}\mathbf{P} - \mathbf{B} \right)\right] \\
       &= \underbrace{\text{max}}_{\left\{\mathbf{P} \left| {p_{ij} \in \{0, 1\}
                              \atop \sum_{i=1}^n p_{ij} = \sum_{j=1}^n p_{ij} = 1} \right
                              \right\}}
          \text{Tr}\left[\mathbf{P}^\dagger\mathbf{A}^\dagger\mathbf{P}\mathbf{B} \right]

    Here, :math:`\mathbf{P}_{n \times n}` is the permutation matrix. Given an intial guess, the
    best local minimum can be obtained by the iterative procedure,

    .. math::
       p_{ij}^{(n + 1)} = p_{ij}^{(n)} \sqrt{ \frac{2\left[\mathbf{T}^{(n)}\right]_{ij}}{\left[
                          \mathbf{P}^{(n)} \left( \left(\mathbf{P}^{(n)}\right)^T \mathbf{T} +
                          \left( \left(\mathbf{P}^{(n)}\right)^T \mathbf{T} \right)^T  \right)
                          \right]_{ij}} }
    where,

    .. math::
       \mathbf{T}^{(n)} = \mathbf{A} \mathbf{P}^{(n)} \mathbf{B}

    Using an initial guess, the iteration can stops when the change in :math:`d` is below the
    specified threshold,

    .. math::
       d = \text{Tr} \left[\left(\mathbf{P}^{(n+1)} -\mathbf{P}^{(n)} \right)^T
                           \left(\mathbf{P}^{(n+1)} -\mathbf{P}^{(n)} \right)\right]

    The outcome of the iterative procedure :math:`\mathbf{P}^{(\infty)}` is not a permutation
    matrix. So, the closest permutation can be found by setting ``refinement=True``. This uses
    :class:`procrustes.permutation.PermutationProcrustes` to find the closest permutation; that is,

    .. math::
       \underbrace{\text{min}}_{\left\{\mathbf{P} \left| {p_{ij} \in \{0, 1\}
                            \atop \sum_{i=1}^n p_{ij} = \sum_{j=1}^n p_{ij} = 1} \right. \right\}}
                            \|\mathbf{P} - \mathbf{P}^{(\infty)}\|_{F}^2
       = \underbrace{\text{max}}_{\left\{\mathbf{P} \left| {p_{ij} \in \{0, 1\}
                            \atop \sum_{i=1}^n p_{ij} = \sum_{j=1}^n p_{ij} = 1} \right. \right\}}
         \text{Tr}\left[\mathbf{P}^\dagger\mathbf{P}^{(\infty)} \right]

    The answer to this problem is a heuristic solution for the matrix-matching problem that seems
    to be relatively accurate.

    **Initial Guess:**

    Two possible initial guesses are inferred from the Umeyama procedure. One can find either the
    closest permutation matrix to :math:`\mathbf{U}_\text{Umeyama}` or to
    :math:`\mathbf{U}_\text{Umeyama}^\text{approx.}`.

    Considering the :class:`procrustes.permutation.PermutationProcrustes`, the resulting permutation
    matrix can be specified as initial guess through ``guess=umeyama`` and ``guess=umeyama_approx``,
    which solves:

    .. math::
        \underbrace{\text{max}}_{\left\{\mathbf{P} \left| {p_{ij} \in \{0, 1\}
                         \atop \sum_{i=1}^n p_{ij} = \sum_{j=1}^n p_{ij} = 1} \right. \right\}}
          \text{Tr}\left[\mathbf{P}^\dagger\mathbf{U}_\text{Umeyama} \right] \\
        \underbrace{\text{max}}_{\left\{\mathbf{P} \left| {p_{ij} \in \{0, 1\}
                         \atop \sum_{i=1}^n p_{ij} = \sum_{j=1}^n p_{ij} = 1} \right. \right\}}
          \text{Tr}\left[\mathbf{P}^\dagger\mathbf{U}_\text{Umeyama}^\text{approx.} \right]

    Another choice is to start by solving a normal permutation Procrustes problem. In other words,
    write new matrices, :math:`\mathbf{A}^0` and :math:`\mathbf{B}^0`, with columns like,

    .. math::
       \begin{bmatrix}
        a_{ii} \\
        p \cdot \text{sgn}\left( a_{ij_\text{max}} \right)
                \underbrace{\text{max}}_{1 \le j \le n} \left(\left|a_{ij}\right|\right)\\
        p^2 \cdot \text{sgn}\left( a_{ij_{\text{max}-1}} \right)
                  \underbrace{\text{max}-1}_{1 \le j \le n} \left(\left|a_{ij}\right|\right)\\
        \vdots
       \end{bmatrix}

    Here, :math:`\text{max}-1` denotes the second-largest absolute value of elements,
    :math:`\text{max}-2` is the third-largest abosule value of elements, etc.

    The matrices :math:`\mathbf{A}^0` and :math:`\mathbf{B}^0` have the diagonal elements of
    :math:`\mathbf{A}` and :math:`\mathbf{B}` in the first row, and below the first row has the
    largest off-diagonal element in row :math:`i`, the second-largest off-diagonal element, etc.
    The elements are weighted by a factor :math:`0 < p < 1`, so that the smaller elements are
    considered less important for matching. The matrices can be truncated after a few terms; for
    example, after the size of elements falls below some threshold. A reasonable choice would be
    to stop after :math:`\lfloor \frac{-2\ln 10}{\ln p} +1\rfloor` rows; this ensures that the
    size of the elements in the last row is less than 1% of those in the first off-diagonal row.

    There are obviously many different ways to construct the matrices :math:`\mathbf{A}^0` and
    :math:`\mathbf{B}^0`. Another, even better, method would be to try to encode not only what the
    off-diagonal elements are, but which element in the matrix they correspond to. One could do that
    by not only listing the diagonal elements, but also listing the associated off-diagonal element.
    I.e., the columns of :math:`\mathbf{A}^0` and :math:`\mathbf{B}^0` would be,

    .. math::
       \begin{bmatrix}
        a_{ii} \\
        p \cdot a_{j_\text{max} j_\text{max}} \\
        p \cdot \text{sgn}\left( a_{ij_\text{max}} \right)
                \underbrace{\text{max}}_{1 \le j \le n} \left(\left|a_{ij}\right|\right)\\
        p^2 \cdot a_{j_{\text{max}-1} j_{\text{max}-1}} \\
        p^2 \cdot \text{sgn}\left( a_{ij_{\text{max}-1}} \right)
                  \underbrace{\text{max}-1}_{1 \le j \le n} \left(\left|a_{ij}\right|\right)\\
        \vdots
       \end{bmatrix}

    In this case, you wuold stop the procedure after
    :math:``m = \left\lfloor {\frac{{ - 4\ln 10}}{{\ln p}} + 1} \right\rfloor`) rows;` rows.

    Then one uses the :class:`procrustes.permutation.PermutationProcrustes` to match the constructed
    matrices :math:`\mathbf{A}^0` and :math:`\mathbf{B}^0` instead of :math:`\mathbf{A}` and
    :math:`\mathbf{B}`. I.e.,

    .. math::
        \underbrace{\text{max}}_{\left\{\mathbf{P} \left| {p_{ij} \in \{0, 1\}
                         \atop \sum_{i=1}^n p_{ij} = \sum_{j=1}^n p_{ij} = 1} \right. \right\}}
          \text{Tr}\left[\mathbf{P}^\dagger \left(\mathbf{A^0}^\dagger\mathbf{B^0}\right)\right]

    Please note that the 'umeyama_approx' might give inaccurate permutation
    matrix. More specificity, this is a approximated Umeyama method. One example
    we can give is that when we compute the permutation matrix that transforms
    :math:`A` to :math:`B`, the "umeyama_approx" method can not give the exact
    permutation transformation matrix while "umeyama", "normal1" and "normal2" do.

    .. math::
        A =
        \begin{bmatrix}
             4 &  5 & -3 &  3 \\
             5 &  7 &  3 & -5 \\
            -3 &  3 &  2 &  2 \\
             3 & -5 &  2 &  5 \\
        \end{bmatrix} \\
        B =
        \begin{bmatrix}
             73 &  100 &   73 &  -62 \\
            100 &  208 & -116 &  154 \\
             73 & -116 &  154 &  100 \\
            -62 &  154 &  100 &  127 \\
        \end{bmatrix} \\
    """
    # check inputs
    A, B = _get_input_arrays(A, B, remove_zero_col, remove_zero_row,
                             pad_mode, translate, scale, check_finite)
    # np.power() can not handle the negatives values
    # Try to convert the matrices to non-negative
    maximum = np.max(np.abs(B)) if np.max(np.abs(B)) > np.max(
        np.abs(A)) else np.max(np.abs(A))
    A += maximum
    B += maximum
    # A += np.min(A, B)
    # B += np.min(A, B)
    # Do single-transformation computation if requested
    transform_mode = transform_mode.lower()
    if transform_mode == 'single_undirected':
        # the initial guess
        guess = _guess_initial_permutation(A, B, mode, add_noise)
        # Compute the permutation matrix by iterations
        U = _compute_transform(A, B, guess, tol, iteration)
        e_opt = error(A, B, U, U)
        return A, B, U, e_opt

    elif transform_mode == 'single_directed':
        # the initial guess
        guess = _2sided_1trans_initial_guess_directed(A, B)
        # Compute the permutation matrix by iterations
        U = _compute_transform_directed(A, B, guess, tol, iteration)
        e_opt = error(A, B, U, U)
        return A, B, U, e_opt

    # Do regular computation
    elif transform_mode == 'double':
        M = A
        N = B
        P, Q, e_opt = _2sided_regular(M, N, tol, iteration)
        return M, N, P, Q, e_opt
    else:
        raise ValueError(
            "Invalid transform_mode argument"
            "(use 'single_undirected', 'single_directed', or 'double')")


def _2sided_regular(M, N, tol, iteration):
    """
    """
    # :math:` {\(\vert M-PNQ \vert\)}^2_F`
    # taken from page 64 in
    # parallel solution of svd-related problems, with applications
    # Pythagoras Papadimitriou, University of Manchester, 1993

    # Fix P = I first
    # Initial guess for P
    P1 = np.eye(M.shape[0], M.shape[0])
    # Initial guess for Q
    Q1 = _2sided_Hungarian(np.dot(N.T, M))
    e_opt1 = error(N, M, P1.T, Q1)
    step1 = 0

    # while loop for the original algorithm
    while (e_opt1 > tol and step1 < iteration):
        step1 += 1
        # Update P
        P1 = _2sided_Hungarian(np.dot(np.dot(N, Q1), M.T))
        P1 = np.transpose(P1)
        # Update the error
        e_opt1 = error(N, M, P1.T, Q1)
        if e_opt1 <= tol:
            break
        else:
            # Update Q
            Q1 = _2sided_Hungarian(np.dot(np.dot(N.T, P1.T), M))
            # Update the error
            e_opt1 = error(N, M, P1.T, Q1)

        if step1 == iteration:
            print('Maximum iteration reached in the first case! \
                Error={0}'.format(e_opt1))

    # Fix Q = I first
    # Initial guess for Q
    Q2 = np.eye(M.shape[1], M.shape[1])
    # Initial guess for P
    P2 = _2sided_Hungarian(np.dot(N, M.T))
    P2 = np.transpose(P2)
    e_opt2 = error(N, M, P2.T, Q2)
    step2 = 0

    # while loop for the original algorithm
    while (e_opt2 > tol and step2 < iteration):
        # Update Q
        Q2 = _2sided_Hungarian(np.dot(np.dot(N.T, P2.T), M))
        # Update the error
        e_opt2 = error(N, M, P2.T, Q1)
        if e_opt2 <= tol:
            break
        else:
            P2 = _2sided_Hungarian(np.dot(np.dot(N, Q2), M.T))
            P2 = np.transpose(P2)
            # Update the error
            e_opt2 = error(N, M, P2.T, Q2)
            step2 += 1
        if step2 == iteration:
            print('Maximum iteration reached in the second case! \
                Error={0}'.format(e_opt2))

    if e_opt1 <= e_opt2:
        P = P1
        Q = Q1
        e_opt = e_opt1
    else:
        P = P2
        Q = Q2
        e_opt = e_opt2

    return P, Q, e_opt


def _2sided_Hungarian(profit_matrix):
    """
    """

    # Define the profit array & applying the hungarian algorithm
    cost_matrix = np.ones(profit_matrix.shape) * np.max(
        profit_matrix) - profit_matrix

    # Obtain the optimum permutation transformation and convert to array
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    perm_optimum = np.zeros(profit_matrix.shape)
    perm_optimum[row_ind, col_ind] = 1

    return perm_optimum


def _2sided_1trans_initial_guess_normal1(A):
    """
    """
    # build the empty target array
    array_c = np.zeros(A.shape)
    # Fill the first row of array_c with diagonal entries
    array_c[0, :] = A.diagonal()
    array_mask = ~np.eye(A.shape[0], dtype=bool)
    # get all the non-diagonal element
    array_c_non_diag = (A[array_mask]).T.reshape(A.shape[0], A.shape[1] - 1)
    array_c_non_diag = array_c_non_diag[
        np.arange(np.shape(array_c_non_diag)[0])[:, np.newaxis],
        np.argsort(abs(array_c_non_diag))]

    # form the right format in order to combine with matrix A
    array_c_sorted = np.fliplr(array_c_non_diag).T
    # fill the array_c with array_c_sorted
    array_c[1:, :] = array_c_sorted
    # the weight matrix
    weight_c = np.zeros(A.shape)
    p = np.power(2, -0.5)

    for weight in range(A.shape[0]):
        weight_c[weight, :] = np.power(p, weight)
    # build the new matrix array_new
    array_new = np.multiply(array_c, weight_c)

    return array_new


def _2sided_1trans_initial_guess_normal2(A):
    """
    """
    array_mask_a = ~np.eye(A.shape[0], dtype=bool)
    # array_off_diag0 is the off diagonal elements of A
    array_off_diag = A[array_mask_a].reshape((A.shape[0], A.shape[1] - 1))
    # array_off_diag1 is sorted off diagonal elements of A
    array_off_diag = array_off_diag[np.arange(np.shape(array_off_diag)[0])[
                                    :, np.newaxis], np.argsort(
        abs(array_off_diag))]
    array_off_diag = np.fliplr(array_off_diag).T

    # array_c is newly built matrix B without weights
    # build array_c with the expected shape
    col_num_new = A.shape[0] * 2 - 1
    array_c = np.zeros((col_num_new, A.shape[1]))
    array_c[0, :] = A.diagonal()

    # use inf to represent the diagonal element
    A_inf = A - np.diag(np.diag(A)) + np.diag([-np.inf] * A.shape[0])
    index_inf = np.argsort(-np.abs((A_inf)), axis=1)

    # the weight matrix
    p = np.power(2, -0.5)
    weight_c = np.zeros((col_num_new, A.shape[1]))
    weight_c[0, :] = np.power(p, 0)

    for index_col in range(1, A.shape[0]):
        # the index_col*2 row of array_c
        array_c[index_col * 2, :] = array_off_diag[index_col - 1, :]
        # the index_col*2-1 row of array_c
        array_c[index_col * 2 - 1, :] = A[index_inf[:, index_col],
                                          index_inf[:, index_col]]

        # the index_col*2 row of weight_c
        weight_c[index_col * 2, :] = np.power(p, index_col)
        # the index_col*2 row of weight_c
        weight_c[index_col * 2 - 1, :] = np.power(p, index_col)

    # the new matrix B
    array_new = np.multiply(array_c, weight_c)
    return array_new


def _2sided_1trans_initial_guess_umeyama(A, B, add_noise):
    """
    """
    # add small random noise matrix when matrices are not diagonalizable
    if add_noise:
        A = np.float_(A)
        shape_A = np.shape(A)
        A += np.random.random(shape_A) * np.trace(np.abs(A)) / shape_A[
            0] * 1.e-8
        B = np.float_(B)
        shape_B = np.shape(B)
        B += np.random.random(shape_B) * np.trace(np.abs(B)) / shape_B[
            0] * 1.e-8
    # calculate the eigenvalue decomposition of A and B
    _, UA = eigendecomposition(A)
    _, UB = eigendecomposition(B)
    # compute U_umeyama
    U = np.dot(np.abs(UA), np.abs(UB.T))
    # compute closest permutation matrix to U
    # In the original paper, it's not like this
    # _, _, U, _ = permutation(np.eye(U.shape[0], dtype=U.dtype), U)
    return U


def _2sided_1trans_initial_guess_umeyama_approx(A, B, add_noise):
    """
    """
    # add small random noise matrix when matrices are not diagonalizable
    if add_noise:
        A = np.float_(A)
        shape_A = np.shape(A)
        A += np.random.random(shape_A) * np.trace(np.abs(A)) / shape_A[
            0] * 1.e-8
        B = np.float_(B)
        shape_B = np.shape(B)
        B += np.random.random(shape_B) * np.trace(np.abs(B)) / shape_B[
            0] * 1.e-8
    # calculate the eigenvalue decomposition of A and B
    _, UA = eigendecomposition(A)
    _, UB = eigendecomposition(B)
    # compute U_umeyama
    U = np.dot(np.abs(UA), np.abs(UB.T))
    # calculate the approximated umeyama matrix
    U_a, _, VTa = np.linalg.svd(U)
    U_approx = np.dot(abs(U_a), np.abs(VTa))
    # compute closest unitary transformation to U
    # _, _, U, _ = permutation(np.eye(U.shape[0], dtype=U.dtype), U)
    return U_approx


def _2sided_1trans_initial_guess_directed(A, B):
    r"""
    """
    # Build two new hermitian matrices
    A_0 = (A + A.T) * 0.5 + (A - A.T) * 0.5 * 1j
    B_0 = (B + B.T) * 0.5 + (B - B.T) * 0.5 * 1j

    _, UA_0 = eigendecomposition(A_0)
    _, UB_0 = eigendecomposition(B_0)
    # Compute the magnitudes of each element
    UA = np.sqrt(np.imag(UA_0) ** 2 + np.real(UA_0) ** 2)
    UB = np.sqrt(np.imag(UB_0) ** 2 + np.real(UB_0) ** 2)
    # compute the initial guess
    U = np.dot(UA, UB.T)
    return U


def _guess_initial_permutation(A, B, mode, add_noise):
    """
    """
    mode = mode.lower()
    if mode == 'normal1':
        tmp_A = _2sided_1trans_initial_guess_normal1(A)
        tmp_B = _2sided_1trans_initial_guess_normal1(B)
        _, _, U, _, = permutation(tmp_A, tmp_B)
    elif mode == 'normal2':
        tmp_A = _2sided_1trans_initial_guess_normal2(A)
        tmp_B = _2sided_1trans_initial_guess_normal2(B)
        _, _, U, _, = permutation(tmp_A, tmp_B)
    elif mode == 'umeyama':
        U = _2sided_1trans_initial_guess_umeyama(A, B, add_noise)
    elif mode == 'umeyama_approx':
        U = _2sided_1trans_initial_guess_umeyama_approx(A, B, add_noise)
    else:
        raise ValueError(
            "Invalid mode argument"
            "(use 'normal1', 'normal2', 'umeyama' or 'umeyama_approx')")
    return U


def _compute_transform(A, B, guess, tol, iteration):
    """
    """

    # shift the the matrices to avoid negative values
    # otherwise it will cause an error in the Eq. 28

    p_old = guess
    change = np.inf
    step = 0

    while (change > tol and step < iteration):
        # Compute p_new
        tmp1 = np.dot(A, np.dot(p_old, B))
        alpha = np.dot(p_old.T, tmp1)
        alpha = (alpha + alpha.T) / 2
        tmp2 = np.power(tmp1 / np.dot(p_old, alpha), 0.5)
        p_new = p_old * tmp2

        # compute the change
        change = np.trace(np.dot((p_new - p_old).T, (p_new - p_old)))
        step += 1
        # update p_old
        p_old = p_new

        if step == iteration:
            print('Maximum iteration reached! Change={0}'.format(change))

    _, _, p_opt, _ = permutation(np.eye(p_new.shape[0]), p_new)

    return p_opt


def _compute_transform_directed(A, B, guess, tol, iteration):
    r"""
    """
    # shift the the matrices to avoid negative values
    # otherwise it will cause an error in the Eq. 28
    p_old = guess
    change = np.inf
    step = 0
    while (change > tol and step < iteration):
        # Compute p_new
        tmp1 = np.dot(A, np.dot(p_old, B.T))
        tmp2 = np.dot(A.T, np.dot(p_old, B))
        alpha = np.dot(p_old.T, tmp1 + tmp2) + np.dot((tmp1 + tmp2).T, p_old)
        alpha = alpha / 4
        tmp = (tmp1 + tmp2) / (2 * np.dot(p_old, alpha))
        p_new = p_old * np.power(tmp, 0.5)
        # compute the change
        change = np.trace(np.dot((p_new - p_old).T, (p_new - p_old)))
        step += 1
        # update p_old
        p_old = p_new
        if step == iteration:
            print('Maximum iteration reached! Change={0}'.format(change))
    _, _, p_opt, _ = permutation(np.eye(p_new.shape[0]), p_new)

    return p_opt


def permutation_2sided_explicit(A, B,
                                remove_zero_col=True,
                                remove_zero_row=True,
                                pad_mode='row-col', translate=False,
                                scale=False, check_finite=True):
    r"""
    Two sided permutation Procrustes by explicit method.

    Parameters
    ----------
    A : ndarray
        The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be
        transformed.
    B : ndarray
        The 2d-array :math:`\mathbf{B}_{m \times n}` representing the reference.
    remove_zero_col : bool, optional
        If True, the zero columns on the right side will be removed.
        Default=True.
    remove_zero_row : bool, optional
        If True, the zero rows on the top will be removed. Default= True.
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
        If True, both arrays are translated to be centered at origin. Default=False.
    scale : bool, optional
        If True, both arrays are column normalized to unity. Default=False.
    check_finite : bool, optional
        If true, convert the input to an array, checking for NaNs or Infs. Default=True.

    Returns
    -------
    A : ndarray
        The transformed ndarray A.
    B : ndarray
        The transformed ndarray B.
    U : ndarray
        The optimum permutation transformation matrix.
    e_opt : float
        Two-sided orthogonal Procrustes error.

    Notes
    -----
    Given matrix :math:`\mathbf{A}_{n \times n}` and a reference
    :math:`\mathbf{B}_{n \times n}`, find a permutation of rows/columns of
    :math:`\mathbf{A}_{n \times n}` that makes it as close as
    possible to :math:`\mathbf{B}_{n \times n}`. But be careful that we are
    using a brutal way to loop over all the possible permutation matrices and
    return the one that gives the minimum error(distance). This method can be
    used as a checker for small datasets.

    """
    print('Warning: This brute-strength method is computational expensive! \n'
          'But it can be used as a checker for a small dataset.')
    # check inputs
    A, B = _get_input_arrays(A, B, remove_zero_col, remove_zero_row,
                             pad_mode, translate, scale, check_finite)
    perm1 = np.zeros(np.shape(A))
    perm_error1 = np.inf
    for comb in it.permutations(np.arange(np.shape(A)[0])):
        # Compute the permutation matrix
        size = np.shape(A)[1]
        perm2 = np.zeros((size, size))
        perm2[np.arange(size), comb] = 1
        perm_error2 = error(A, B, perm2, perm2)
        if perm_error2 < perm_error1:
            perm_error1 = perm_error2
            perm1 = perm2
    return A, B, perm1, perm_error1

