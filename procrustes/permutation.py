# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2020 The Procrustes Development Team
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

import itertools as it

import numpy as np
from procrustes.utils import error, setup_input_arrays
from scipy.optimize import linear_sum_assignment

__all__ = [
    "permutation",
    "permutation_2sided",
    "permutation_2sided_explicit"
]


def permutation(array_a, array_b, remove_zero_col=True, remove_zero_row=True,
                pad_mode="row-col", translate=False, scale=False, check_finite=True):
    r"""
    Single sided permutation Procrustes.

    Parameters
    ----------
    array_a : ndarray
        The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
    array_b : ndarray
        The 2d-array :math:`\mathbf{B}_{m \times n}` representing the reference.
    remove_zero_col : bool, optional
        If True, the zero columns on the right side will be removed. Default= True.
    remove_zero_row : bool, optional
        If True, the zero rows on the top will be removed. Default= True.
    pad_mode : str, optional
        Zero padding mode when the sizes of two arrays differ. Default="row-col".
        "row": The array with fewer rows is padded with zero rows so that both have the same number
        of rows.
        "col": The array with fewer columns is padded with zero columns so that both have the
        same number of columns.
        "row-col": The array with fewer rows is padded with zero rows, and the array with fewer
        columns is padded with zero columns, so that both have the same dimensions.
        This does not necessarily result in square arrays.
        "square": The arrays are padded with zero rows and zero columns so that they are both
        squared arrays. The dimension of square array is specified based on the highest dimension,
        i.e. :math:`\text{max}(n_a, m_a, n_b, m_b)`."
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
    new_a, new_b = setup_input_arrays(array_a, array_b, remove_zero_col, remove_zero_row,
                                      pad_mode, translate, scale, check_finite)
    # compute permutation Procrustes matrix
    array_p = np.dot(new_a.T, new_b)
    array_c = np.full(array_p.shape, np.max(array_p))
    array_c -= array_p
    array_u = np.zeros(array_p.shape)
    # set elements to 1 according to Hungarian algorithm (linear_sum_assignment)
    array_u[linear_sum_assignment(array_c)] = 1
    e_opt = error(new_a, new_b, array_u)
    return new_a, new_b, array_u, e_opt


def permutation_2sided(array_a, array_b, transform_mode="single_undirected",
                       remove_zero_col=True, remove_zero_row=True,
                       pad_mode="row-col", translate=False, scale=False,
                       mode="normal1", check_finite=True, iteration=500,
                       add_noise=False, tol=1.0e-8):
    r"""
    Single sided permutation Procrustes.

    Parameters
    ----------
    array_a : ndarray
        The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
    array_b : ndarray
        The 2d-array :math:`\mathbf{B}_{m \times n}` representing the reference.
    transform_mode : str
        If transform_mode="single_undirected", two-sided permutation Procrustes with one
        transformation will be performed. If transform_mode="single_directed", two-sided permutation
        for directed graph matching will be used. Otherwise, transform_mode="double", the
        two-sided permutation Procrustes with two transformations will be performed.
        Default="single_undirected".
    remove_zero_col : bool, optional
        If True, the zero columns on the right side will be removed. Default= True.
    remove_zero_row : bool, optional
        If True, the zero rows on the bottom will be removed. Default= True.
    pad_mode : str, optional
        Specifying how to pad the arrays, listed below. Default="row-col".

            - "row"
                The array with fewer rows is padded with zero rows so that both have the same
                number of rows.
            - "col"
                The array with fewer columns is padded with zero columns so that both have the
                same number of columns.
            - "row-col"
                The array with fewer rows is padded with zero rows, and the array with fewer
                columns is padded with zero columns, so that both have the same dimensions.
                This does not necessarily result in square arrays.
            - "square"
                The arrays are padded with zero rows and zero columns so that they are both
                squared arrays. The dimension of square array is specified based on the highest
                dimension, i.e. :math:`\text{max}(n_a, m_a, n_b, m_b)`.
    translate : bool, optional
        If True, both arrays are translated to be centered at origin. Default=False.
    scale : bool, optional
        If True, both arrays are column normalized to unity. Default=False.
    mode : string, optional
        Option for choosing the initial guess methods, including "normal1",
        "normal2", "umeyama" and "umeyama_approx". "umeyama_approx" is the
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
    new_a : ndarray
        The transformed ndarray A.
    new_b : ndarray
        The transformed ndarray B.
    array_u : ndarray
        The optimum permutation transformation matrix.
    array_p : ndarray
        The optimum permutation transformation matrix when using double transform mode.
    array_q : ndarray
        The optimum permutation transformation matrix when using double transform mode.
    e_opt : float
        Two-sided permutation Procrustes error.

    Notes
    -----
    Given matrix :math:`\mathbf{A}_{n \times n}` and a reference :math:`\mathbf{B}_{n \times n}`,
    find a permutation of rows/columns of :math:`\mathbf{A}_{n \times n}` that makes it as close as
    possible to :math:`\mathbf{B}_{n \times n}`. I.e.,

    .. math::
        &\underbrace{\text{min}}_{\left\{\mathbf{P} \left| {p_{ij} \in \{0, 1\}
            \atop \sum_{i=1}^n p_{ij} = \sum_{j=1}^n p_{ij} = 1} \right. \right\}}
            \|\mathbf{P}^\dagger \mathbf{A} \mathbf{P} - \mathbf{B}\|_{F}^2\\
        = &\underbrace{\text{min}}_{\left\{\mathbf{P} \left| {p_{ij} \in \{0, 1\}
            \atop \sum_{i=1}^n p_{ij} = \sum_{j=1}^n p_{ij} = 1} \right. \right\}}
            \text{Tr}\left[\left(\mathbf{P}^\dagger\mathbf{A}\mathbf{P} - \mathbf{B} \right)^\dagger
            \left(\mathbf{P}^\dagger\mathbf{A}\mathbf{P} - \mathbf{B} \right)\right] \\
        = &\underbrace{\text{max}}_{\left\{\mathbf{P} \left| {p_{ij} \in \{0, 1\}
            \atop \sum_{i=1}^n p_{ij} = \sum_{j=1}^n p_{ij} = 1} \right. \right\}}
            \text{Tr}\left[\mathbf{P}^\dagger\mathbf{A}^\dagger\mathbf{P}\mathbf{B} \right]\\

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

    In this case, you would stop the procedure after
    :math:`m = \left\lfloor {\frac{{ - 4\ln 10}}{{\ln p}} + 1} \right \rfloor` rows.

    Then one uses the :class:`procrustes.permutation.PermutationProcrustes` to match the constructed
    matrices :math:`\mathbf{A}^0` and :math:`\mathbf{B}^0` instead of :math:`\mathbf{A}` and
    :math:`\mathbf{B}`. I.e.,

    .. math::
        \underbrace{\text{max}}_{\left\{\mathbf{P} \left| {p_{ij} \in \{0, 1\}
                         \atop \sum_{i=1}^n p_{ij} = \sum_{j=1}^n p_{ij} = 1} \right. \right\}}
          \text{Tr}\left[\mathbf{P}^\dagger \left(\mathbf{A^0}^\dagger\mathbf{B^0}\right)\right]

    Please note that the "umeyama_approx" might give inaccurate permutation
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
    new_a, new_b = setup_input_arrays(array_a, array_b, remove_zero_col, remove_zero_row,
                                      pad_mode, translate, scale, check_finite)
    # np.power() can not handle the negatives values
    # Try to convert the matrices to non-negative
    maximum = np.max(np.abs(new_b)) if np.max(np.abs(new_b)) > np.max(
        np.abs(new_a)) else np.max(np.abs(new_a))
    new_a += maximum
    new_b += maximum
    # A += np.min(A, B)
    # B += np.min(A, B)
    # Do single-transformation computation if requested
    transform_mode = transform_mode.lower()
    if transform_mode == "single_undirected":
        # the initial guess
        guess = _guess_initial_permutation(new_a, new_b, mode, add_noise)
        # Compute the permutation matrix by iterations
        array_u = _compute_transform(new_a, new_b, guess, tol, iteration)
        e_opt = error(new_a, new_b, array_u, array_u)
        return new_a, new_b, array_u, e_opt

    elif transform_mode == "single_directed":
        # the initial guess
        guess = _2sided_1trans_initial_guess_directed(new_a, new_b)
        # Compute the permutation matrix by iterations
        array_u = _compute_transform_directed(new_a, new_b, guess, tol, iteration)
        e_opt = error(new_a, new_b, array_u, array_u)
        return new_a, new_b, array_u, e_opt

    # Do regular computation
    elif transform_mode == "double":
        array_m = new_a
        array_n = new_b
        array_p, array_q, e_opt = _2sided_regular(array_m, array_n, tol, iteration)
        return array_m, array_n, array_p, array_q, e_opt
    else:
        raise ValueError(
            """
            Invalid transform_mode argument, use "single_undirected", "single_directed", or "double"
            """)


def _2sided_regular(array_m, array_n, tol, iteration):
    # Regular two-sided permutation Procrustes
    # :math:` {\(\vert M-PNQ \vert\)}^2_F`
    # taken from page 64 in
    # parallel solution of svd-related problems, with applications
    # Pythagoras Papadimitriou, University of Manchester, 1993

    # Fix P = I first
    # Initial guess for P
    array_p1 = np.eye(array_m.shape[0], array_m.shape[0])
    # Initial guess for Q
    array_q1 = _2sided_hungarian(np.dot(array_n.T, array_m))
    e_opt1 = error(array_n, array_m, array_p1.T, array_q1)
    step1 = 0

    # while loop for the original algorithm
    while e_opt1 > tol and step1 < iteration:
        step1 += 1
        # Update P
        array_p1 = _2sided_hungarian(np.dot(np.dot(array_n, array_q1), array_m.T))
        array_p1 = np.transpose(array_p1)
        # Update the error
        e_opt1 = error(array_n, array_m, array_p1.T, array_q1)
        if e_opt1 > tol:
            # Update Q
            array_q1 = _2sided_hungarian(np.dot(np.dot(array_n.T, array_p1.T), array_m))
            # Update the error
            e_opt1 = error(array_n, array_m, array_p1.T, array_q1)
        else:
            break

        if step1 == iteration:
            print("Maximum iteration reached in the first case! Error={0}".format(e_opt1))

    # Fix Q = I first
    # Initial guess for Q
    array_q2 = np.eye(array_m.shape[1], array_m.shape[1])
    # Initial guess for P
    array_p2 = _2sided_hungarian(np.dot(array_n, array_m.T))
    array_p2 = np.transpose(array_p2)
    e_opt2 = error(array_n, array_m, array_p2.T, array_q2)
    step2 = 0

    # while loop for the original algorithm
    while e_opt2 > tol and step2 < iteration:
        # Update Q
        array_q2 = _2sided_hungarian(np.dot(np.dot(array_n.T, array_p2.T), array_m))
        # Update the error
        e_opt2 = error(array_n, array_m, array_p2.T, array_q1)
        if e_opt2 > tol:
            array_p2 = _2sided_hungarian(np.dot(np.dot(array_n, array_q2), array_m.T))
            array_p2 = np.transpose(array_p2)
            # Update the error
            e_opt2 = error(array_n, array_m, array_p2.T, array_q2)
            step2 += 1
        else:
            break
        if step2 == iteration:
            print("Maximum iteration reached in the second case! Error={0}".format(e_opt2))

    if e_opt1 <= e_opt2:
        array_p = array_p1
        array_q = array_q1
        e_opt = e_opt1
    else:
        array_p = array_p2
        array_q = array_q2
        e_opt = e_opt2

    return array_p, array_q, e_opt


def _2sided_hungarian(profit_matrix):
    # Define the profit array & applying the hungarian algorithm
    cost_matrix = np.ones(profit_matrix.shape) * np.max(profit_matrix) - profit_matrix

    # Obtain the optimum permutation transformation and convert to array
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    perm_optimum = np.zeros(profit_matrix.shape)
    perm_optimum[row_ind, col_ind] = 1

    return perm_optimum


def _2sided_1trans_initial_guess_normal1(array_a):
    # build the empty target array
    array_c = np.zeros(array_a.shape)
    # Fill the first row of array_c with diagonal entries
    array_c[0, :] = array_a.diagonal()
    array_mask = ~np.eye(array_a.shape[0], dtype=bool)
    # get all the non-diagonal element
    array_c_non_diag = (array_a[array_mask]).T.reshape(array_a.shape[0], array_a.shape[1] - 1)
    array_c_non_diag = array_c_non_diag[
        np.arange(np.shape(array_c_non_diag)[0])[:, np.newaxis],
        np.argsort(abs(array_c_non_diag))]

    # form the right format in order to combine with matrix A
    array_c_sorted = np.fliplr(array_c_non_diag).T
    # fill the array_c with array_c_sorted
    array_c[1:, :] = array_c_sorted
    # the weight matrix
    weight_c = np.zeros(array_a.shape)
    weight_p = np.power(2, -0.5)

    for weight in range(array_a.shape[0]):
        weight_c[weight, :] = np.power(weight_p, weight)
    # build the new matrix array_new
    array_new = np.multiply(array_c, weight_c)

    return array_new


def _2sided_1trans_initial_guess_normal2(array_a):
    array_mask_a = ~np.eye(array_a.shape[0], dtype=bool)
    # array_off_diag0 is the off diagonal elements of A
    array_off_diag = array_a[array_mask_a].reshape((array_a.shape[0], array_a.shape[1] - 1))
    # array_off_diag1 is sorted off diagonal elements of A
    array_off_diag = array_off_diag[np.arange(np.shape(array_off_diag)[0])[
                                    :, np.newaxis], np.argsort(
        abs(array_off_diag))]
    array_off_diag = np.fliplr(array_off_diag).T

    # array_c is newly built matrix B without weights
    # build array_c with the expected shape
    col_num_new = array_a.shape[0] * 2 - 1
    array_c = np.zeros((col_num_new, array_a.shape[1]))
    array_c[0, :] = array_a.diagonal()

    # use inf to represent the diagonal element
    a_inf = array_a - np.diag(np.diag(array_a)) + np.diag([-np.inf] * array_a.shape[0])
    index_inf = np.argsort(-np.abs((a_inf)), axis=1)

    # the weight matrix
    weight_p = np.power(2, -0.5)
    weight_c = np.zeros((col_num_new, array_a.shape[1]))
    weight_c[0, :] = np.power(weight_p, 0)

    for index_col in range(1, array_a.shape[0]):
        # the index_col*2 row of array_c
        array_c[index_col * 2, :] = array_off_diag[index_col - 1, :]
        # the index_col*2-1 row of array_c
        array_c[index_col * 2 - 1, :] = array_a[index_inf[:, index_col],
                                                index_inf[:, index_col]]

        # the index_col*2 row of weight_c
        weight_c[index_col * 2, :] = np.power(weight_p, index_col)
        # the index_col*2 row of weight_c
        weight_c[index_col * 2 - 1, :] = np.power(weight_p, index_col)

    # the new matrix B
    array_new = np.multiply(array_c, weight_c)
    return array_new


def _2sided_1trans_initial_guess_umeyama(array_a, array_b, add_noise):
    # add small random noise matrix when matrices are not diagonalizable
    if add_noise:
        array_a = np.float_(array_a)
        array_a += np.random.random(array_a.shape) * np.trace(np.abs(array_a)) /\
            array_a.shape[0] * 1.e-8
        array_b = np.float_(array_b)
        array_b += np.random.random(array_b.shape) * np.trace(np.abs(array_b)) /\
            array_b.shape[0] * 1.e-8
    # calculate the eigenvalue decomposition of A and B
    _, array_ua = np.linalg.eigh(array_a)
    _, array_ub = np.linalg.eigh(array_b)
    # compute U_umeyama
    array_u = np.dot(np.abs(array_ua), np.abs(array_ub.T))
    # compute closest permutation matrix to U
    # In the original paper, it"s not like this
    # _, _, U, _ = permutation(np.eye(U.shape[0], dtype=U.dtype), U)
    return array_u


def _2sided_1trans_initial_guess_umeyama_approx(array_a, array_b, add_noise):
    # compute U_umeyama
    array_u = _2sided_1trans_initial_guess_umeyama(array_a, array_b, add_noise)
    # calculate the approximated umeyama matrix
    array_ua, _, array_vta = np.linalg.svd(array_u)
    u_approx = np.dot(np.abs(array_ua), np.abs(array_vta))
    # compute closest unitary transformation to U
    # _, _, U, _ = permutation(np.eye(U.shape[0], dtype=U.dtype), U)
    return u_approx


def _2sided_1trans_initial_guess_directed(array_a, array_b):
    # Build two new hermitian matrices
    a_0 = (array_a + array_a.T) * 0.5 + (array_a - array_a.T) * 0.5 * 1j
    b_0 = (array_b + array_b.T) * 0.5 + (array_b - array_b.T) * 0.5 * 1j

    _, ua_0 = np.linalg.eigh(a_0)
    _, ub_0 = np.linalg.eigh(b_0)
    # Compute the magnitudes of each element
    array_ua = np.sqrt(np.imag(ua_0) ** 2 + np.real(ua_0) ** 2)
    array_ub = np.sqrt(np.imag(ub_0) ** 2 + np.real(ub_0) ** 2)
    # compute the initial guess
    array_u = np.dot(array_ua, array_ub.T)
    return array_u


def _guess_initial_permutation(array_a, array_b, mode, add_noise):
    mode = mode.lower()
    if mode == "normal1":
        tmp_a = _2sided_1trans_initial_guess_normal1(array_a)
        tmp_b = _2sided_1trans_initial_guess_normal1(array_b)
        _, _, array_u, _, = permutation(tmp_a, tmp_b)
    elif mode == "normal2":
        tmp_a = _2sided_1trans_initial_guess_normal2(array_a)
        tmp_b = _2sided_1trans_initial_guess_normal2(array_b)
        _, _, array_u, _, = permutation(tmp_a, tmp_b)
    elif mode == "umeyama":
        array_u = _2sided_1trans_initial_guess_umeyama(array_a, array_b, add_noise)
    elif mode == "umeyama_approx":
        array_u = _2sided_1trans_initial_guess_umeyama_approx(array_a, array_b, add_noise)
    else:
        raise ValueError(
            """
            Invalid mode argument, use "normal1", "normal2", "umeyama" or "umeyama_approx".
            """)
    return array_u


def _compute_transform(array_a, array_b, guess, tol, iteration):
    # shift the the matrices to avoid negative values
    # otherwise it will cause an error in the Eq. 28
    p_old = guess
    change = np.inf
    step = 0

    while change > tol and step < iteration:
        # Compute p_new
        tmp1 = np.dot(array_a, np.dot(p_old, array_b))
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
            print("Maximum iteration reached! Change={0}".format(change))

    _, _, p_opt, _ = permutation(np.eye(p_new.shape[0]), p_new)

    return p_opt


def _compute_transform_directed(array_a, array_b, guess, tol, iteration):
    # shift the the matrices to avoid negative values
    # otherwise it will cause an error in the Eq. 28
    p_old = guess
    change = np.inf
    step = 0
    while change > tol and step < iteration:
        # Compute p_new
        tmp1 = np.dot(array_a, np.dot(p_old, array_b.T))
        tmp2 = np.dot(array_a.T, np.dot(p_old, array_b))
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
            print("Maximum iteration reached! Change={0}".format(change))
    _, _, p_opt, _ = permutation(np.eye(p_new.shape[0]), p_new)

    return p_opt


def permutation_2sided_explicit(array_a, array_b,
                                remove_zero_col=True,
                                remove_zero_row=True,
                                pad_mode="row-col", translate=False,
                                scale=False, check_finite=True):
    r"""
    Two sided permutation Procrustes by explicit method.

    Parameters
    ----------
    array_a : ndarray
        The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be
        transformed.
    array_b : ndarray
        The 2d-array :math:`\mathbf{B}_{m \times n}` representing the reference.
    remove_zero_col : bool, optional
        If True, the zero columns on the right side will be removed.
        Default=True.
    remove_zero_row : bool, optional
        If True, the zero rows on the bottom will be removed. Default= True.
    pad_mode : str, optional
        Zero padding mode when the sizes of two arrays differ. Default="row-col".
        "row": The array with fewer rows is padded with zero rows so that both have the same number
        of rows.
        "col": The array with fewer columns is padded with zero columns so that both have the
        same number of columns.
        "row-col": The array with fewer rows is padded with zero rows, and the array with fewer
        columns is padded with zero columns, so that both have the same dimensions.
        This does not necessarily result in square arrays.
        "square": The arrays are padded with zero rows and zero columns so that they are both
        squared arrays. The dimension of square array is specified based on the highest dimension,
        i.e. :math:`\text{max}(n_a, m_a, n_b, m_b)`."
    translate : bool, optional
        If True, both arrays are translated to be centered at origin. Default=False.
    scale : bool, optional
        If True, both arrays are column normalized to unity. Default=False.
    check_finite : bool, optional
        If true, convert the input to an array, checking for NaNs or Infs. Default=True.

    Returns
    -------
    new_a : ndarray
        The transformed ndarray A.
    new_b : ndarray
        The transformed ndarray B.
    array_p : ndarray
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
    used as a checker for small dataset.

    """
    print("Warning: This brute-strength method is computational expensive! \n"
          "But it can be used as a checker for a small dataset.")
    # check inputs
    new_a, new_b = setup_input_arrays(array_a, array_b, remove_zero_col, remove_zero_row,
                                      pad_mode, translate, scale, check_finite)
    perm1 = np.zeros(np.shape(new_a))
    perm_error1 = np.inf
    for comb in it.permutations(np.arange(np.shape(new_a)[0])):
        # Compute the permutation matrix
        size = np.shape(new_a)[1]
        perm2 = np.zeros((size, size))
        perm2[np.arange(size), comb] = 1
        perm_error2 = error(new_a, new_b, perm2, perm2)
        if perm_error2 < perm_error1:
            perm_error1 = perm_error2
            perm1 = perm2
    return new_a, new_b, perm1, perm_error1
