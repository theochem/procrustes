# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2021 The QC-Devs Community
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
from procrustes.kopt import kopt_heuristic_double, kopt_heuristic_single
from procrustes.utils import _zero_padding, compute_error, ProcrustesResult, setup_input_arrays
import scipy
from scipy.optimize import linear_sum_assignment

__all__ = ["permutation", "permutation_2sided", "permutation_2sided_explicit"]


def permutation(
    a,
    b,
    pad=True,
    translate=False,
    scale=False,
    unpad_col=False,
    unpad_row=False,
    check_finite=True,
    weight=None,
):
    r"""Perform one-sided permutation Procrustes.

    Given matrix :math:`\mathbf{A}_{m \times n}` and a reference matrix :math:`\mathbf{B}_{m \times
    n}`, find the permutation transformation matrix :math:`\mathbf{P}_{n \times n}`
    that makes :math:`\mathbf{AP}` as close as possible to :math:`\mathbf{B}`. In other words,

    .. math::
       \underbrace{\text{min}}_{\left\{\mathbf{P} \left| {[\mathbf{P}]_{ij} \in \{0, 1\} \atop
       \sum_{i=1}^n [\mathbf{P}]_{ij} = \sum_{j=1}^n [\mathbf{P}]_{ij} = 1} \right. \right\}}
       \|\mathbf{A} \mathbf{P} - \mathbf{B}\|_{F}^2

    This Procrustes method requires the :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices to
    have the same shape, which is guaranteed with the default ``pad=True`` argument for any given
    :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices. In preparing the :math:`\mathbf{A}` and
    :math:`\mathbf{B}` matrices, the (optional) order of operations is: **1)** unpad zero
    rows/columns, **2)** translate the matrices to the origin, **3)** weight entries of
    :math:`\mathbf{A}`, **4)** scale the matrices to have unit norm, **5)** pad matrices with zero
    rows/columns so they have the same shape.

    Parameters
    ----------
    a : ndarray
        The 2d-array :math:`\mathbf{A}` which is going to be transformed.
    b : ndarray
        The 2d-array :math:`\mathbf{B}` representing the reference matrix.
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
        If True, zero columns (with values less than 1.0e-8) on the right-hand side are removed.
    unpad_row : bool, optional
        If True, zero rows (with values less than 1.0e-8) at the bottom are removed.
    check_finite : bool, optional
        If True, convert the input to an array, checking for NaNs or Infs.
    weight : ndarray, optional
        The 1D-array representing the weights of each row of :math:`\mathbf{A}`. This defines the
        elements of the diagonal matrix :math:`\mathbf{W}` that is multiplied by :math:`\mathbf{A}`
        matrix, i.e., :math:`\mathbf{A} \rightarrow \mathbf{WA}`.

    Returns
    -------
    res : ProcrustesResult
        The Procrustes result represented as a class:`utils.ProcrustesResult` object.

    Notes
    -----
    The optimal :math:`n \times n` permutation matrix is obtained by,

    .. math::
        \mathbf{P}^{\text{opt}} =
        \arg \underbrace{\text{min}}_{\left\{\mathbf{P} \left| {[\mathbf{P}]_{ij} \in \{0, 1\}
        \atop \sum_{i=1}^n [\mathbf{P}]_{ij} = \sum_{j=1}^n [\mathbf{P}]_{ij} = 1} \right. \right\}}
            \|\mathbf{A} \mathbf{P} - \mathbf{B}\|_{F}^2
      = \underbrace{\text{max}}_{\left\{\mathbf{P} \left| {[\mathbf{P}]_{ij} \in \{0, 1\}
        \atop \sum_{i=1}^n [\mathbf{P}]_{ij} = \sum_{j=1}^n [\mathbf{P}]_{ij} = 1} \right. \right\}}
            \text{Tr}\left[\mathbf{P}^\dagger\mathbf{A}^\dagger\mathbf{B} \right]

    The solution is found by relaxing the problem into a linear programming problem. The solution
    to a linear programming problem is always at the boundary of the allowed region. So,

    .. math::
       \underbrace{\text{max}}_{\left\{\mathbf{P} \left| {[\mathbf{P}]_{ij} \in \{0, 1\}
       \atop \sum_{i=1}^n [\mathbf{P}]_{ij} = \sum_{j=1}^n [\mathbf{P}]_{ij} = 1} \right. \right\}}
          \text{Tr}\left[\mathbf{P}^\dagger\mathbf{A}^\dagger\mathbf{B} \right] =
       \underbrace{\text{max}}_{\left\{\mathbf{P} \left| {[\mathbf{P}]_{ij} \geq 0
       \atop \sum_{i=1}^n [\mathbf{P}]_{ij} = \sum_{j=1}^n [\mathbf{P}]_{ij} = 1} \right. \right\}}
          \text{Tr}\left[\mathbf{P}^\dagger\left(\mathbf{A}^\dagger\mathbf{B}\right) \right]

    This is a matching problem and can be solved by the Hungarian algorithm. The cost matrix is
    defined as :math:`\mathbf{A}^\dagger\mathbf{B}` and the `scipy.optimize.linear_sum_assignment`
    is used to solve for the permutation that maximizes the linear sum assignment problem.

    """
    # check inputs
    new_a, new_b = setup_input_arrays(
        a, b, unpad_col, unpad_row, pad, translate, scale, check_finite, weight,
    )
    # if number of rows is less than column, the arrays are made square
    if (new_a.shape[0] < new_a.shape[1]) or (new_b.shape[0] < new_b.shape[1]):
        new_a, new_b = _zero_padding(new_a, new_b, "square")

    # compute cost matrix C = A.T B
    c = np.dot(new_a.T, new_b)
    # solve linear sum assignment problem to get the row/column indices of optimal assignment
    row_ind, col_ind = linear_sum_assignment(c, maximize=True)
    # make the permutation matric by setting the corresponding elements to 1
    p = np.zeros(c.shape)
    p[(row_ind, col_ind)] = 1
    # compute one-sided permutation error
    error = compute_error(new_a, new_b, p)

    return ProcrustesResult(new_a=new_a, new_b=new_b, t=p, error=error)


def permutation_2sided(
        a,
        b,
        single=True,
        pad=False,
        unpad_col=False,
        unpad_row=False,
        translate=False,
        scale=False,
        mode="normal1",
        check_finite=True,
        iteration=500,
        tol=1.0e-8,
        kopt=None,
        weight=None
):
    r"""Double sided permutation Procrustes.

    Parameters
    ----------
    a : ndarray
        The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
    b : ndarray
        The 2d-array :math:`\mathbf{B}_{m \times n}` representing the reference.
    single : bool
        If true, the two permutation are assumed to be the same, i.e.
        it is the two-sided permutation Procrustes with one transformation.
        (1). If the input matrices (adjacency matrices) are symmetric within the threshold of 1.e-5,
        undirected graph matching algorithm will be applied.
        (2). If the input matrices (adjacency matrices) are asymmetric, the directed graph
        matching is applied.
        If false, the two permutation matrices can be different, i.e.
        it is the two-sided permutation Procrustes with two transformations
        (known as the regular two-sided permutation Procrustes here). An flip-flop
        algorithm taken from  *Parallel solution of SVD-related problems, with applications,
        Pythagoras Papadimitriou, Ph.D. Thesis, University of Manchester, 1993* is used to solve
        the problem.
        Default=True.
    pad : bool, optional
        Add zero rows (at the bottom) and/or columns (to the right-hand side) of matrices
        :math:`\mathbf{A}` and :math:`\mathbf{B}` so that they have the same shape.
    unpad_col : bool, optional
        If True, zero columns (with values less than 1.0e-8) on the right-hand side of the intial
        :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices are removed.
    unpad_row : bool, optional
        If True, zero rows (with values less than 1.0e-8) at the bottom of the intial
        :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices are removed.
    translate : bool, optional
        If True, both arrays are translated to be centered at origin, ie columns of the arrays
        will have mean zero.
        Default=False.
    scale : bool, optional
        If True, both arrays are normalized to one with respect to the Frobenius norm, ie
        :math:`Tr(A^T A) = 1`.
        Default=False.
    mode : string, optional
        Option for choosing the initial guess methods, including "normal1",
        "normal2", "umeyama" and "umeyama_approx". "umeyama_approx" is the
        approximated umeyama method. Only used when matrices a, b are symmetric.
    check_finite : bool, optional
        If true, convert the input to an array, checking for NaNs or Infs.
        Default=True.
    iteration : int, optional
        Maximum number for iterations. Default=500.
    tol : float, optional
        The tolerance value used for updating the initial guess. Default=1.e-8.
    kopt : (int, None), optional
        Perform a k-opt heuristic search afterwards to further optimize/refine the permutation
        matrix by searching over all k-fold permutations of the rows or columns of each permutation
        matrix. For example, kopt_k=3 searches over all permutations of 3 rows or columns.
        If None, then kopt search is not performed. Default=None.
    weight : ndarray, optional
        The 1D-array representing the weights of each row of :math:`\mathbf{A}`. This defines the
        elements of the diagonal matrix :math:`\mathbf{W}` that is multiplied by :math:`\mathbf{A}`
        matrix, i.e., :math:`\mathbf{A} \rightarrow \mathbf{WA}`.

    Returns
    -------
    res : ProcrustesResult
        The Procrustes result represented as a class:`utils.ProcrustesResult` object.

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
    if not (isinstance(kopt, int) or kopt is None):
        raise TypeError(f"kopt parameter {kopt} should be an positive integer or None.")

    if not isinstance(single, bool):
        raise TypeError(f"single parameter {single} should be a Boolean.")

    # check inputs
    new_a, new_b = setup_input_arrays(a, b, unpad_col, unpad_row,
                                      pad, translate, scale, check_finite, weight)
    if single:
        # Since permutation matrices are square, and its single transformation.
        if new_a.shape != new_b.shape:
            raise ValueError(
                f"Shape of A and B does not match: {new_a.shape} != {new_b.shape} "
                "Check pad, unpad_col, and unpad_row arguments."
            )

    # Do single-transformation computation if requested
    if single:
        # The update formula for _compute_transform and _compute_transform_directed takes
        #   the square root of the matrix entries. To avoid taking the square root of negative
        #   values and dealing with complex numbers, the matrices are translated to be
        #   positive. This causes no change to the objective function, as it's a constant value
        #   being added to all entries of a and b.
        maximum = max(np.amax(np.abs(new_a)), np.amax(np.abs(new_b)))
        new_a_positive = new_a.astype(np.float) + maximum
        new_b_positive = new_b.astype(np.float) + maximum

        # algorithm for undirected graph matching problem
        # check if two matrices are symmetric within a relative tolerance and absolute tolerance.
        if np.allclose(new_a_positive, new_a_positive.T, rtol=1.e-05, atol=1.e-08) and \
                np.allclose(new_b_positive, new_b_positive.T, rtol=1.e-05, atol=1.e-08):
            # the initial guess
            guess = _guess_initial_permutation_undirected(new_a_positive, new_b_positive, mode)
            # Compute the permutation matrix by iterations
            array_u = _compute_transform(new_a_positive, new_b_positive, guess, tol, iteration)
        # algorithm for directed graph matching problem
        else:
            # the initial guess
            guess = _2sided_1trans_initial_guess_directed(new_a_positive, new_b_positive)
            # Compute the permutation matrix by iterations
            array_u = _compute_transform_directed(new_a_positive, new_b_positive,
                                                  guess, tol, iteration)
        # k-opt heuristic
        if kopt is not None:
            fun_error = lambda p: compute_error(new_a_positive, new_b_positive, p, p.T)
            array_u, error = kopt_heuristic_single(fun_error, p0=array_u, k=kopt)
        else:
            error = compute_error(new_a_positive, new_b_positive, array_u, array_u.T)
        return ProcrustesResult(error=error, new_a=new_a, new_b=new_b, t=array_u, s=None)
    # Do regular computation with different permutation matrices.
    array_p, array_q, error = _2sided_regular(new_a, new_b, tol, iteration)
    # perform k-opt heuristic search.
    if kopt is not None:
        fun_error = lambda p1, p2: compute_error(new_a, new_b, p2, p1.T)
        array_p, array_q, error = kopt_heuristic_double(fun_error, p1=array_p, p2=array_q,
                                                        k=kopt)
    # return array_m, array_n, array_p, array_q, error
    return ProcrustesResult(error=error, new_a=new_a, new_b=new_b, t=array_q, s=array_p)


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
    error1 = compute_error(array_n, array_m, array_q1, array_p1)
    step1 = 0

    # while loop for the original algorithm
    while error1 > tol and step1 < iteration:
        step1 += 1
        # Update P
        array_p1 = _2sided_hungarian(np.dot(np.dot(array_n, array_q1), array_m.T))
        array_p1 = np.transpose(array_p1)
        # Update the error
        error1 = compute_error(array_n, array_m, array_q1, array_p1)
        if error1 > tol:
            # Update Q
            array_q1 = _2sided_hungarian(np.dot(np.dot(array_n.T, array_p1.T), array_m))
            # Update the error
            error1 = compute_error(array_n, array_m, array_q1, array_p1)
        else:
            break

        if step1 == iteration:
            print("Maximum iteration reached in the first case! Error={0}".format(error1))

    # Fix Q = I first
    # Initial guess for Q
    array_q2 = np.eye(array_m.shape[1], array_m.shape[1])
    # Initial guess for P
    array_p2 = _2sided_hungarian(np.dot(array_n, array_m.T))
    array_p2 = np.transpose(array_p2)
    error2 = compute_error(array_n, array_m, array_q2, array_p2)
    step2 = 0

    # while loop for the original algorithm
    while error2 > tol and step2 < iteration:
        # Update Q
        array_q2 = _2sided_hungarian(np.dot(np.dot(array_n.T, array_p2.T), array_m))
        # Update the error
        error2 = compute_error(array_n, array_m, array_q1, array_p2)
        if error2 > tol:
            array_p2 = _2sided_hungarian(np.dot(np.dot(array_n, array_q2), array_m.T))
            array_p2 = np.transpose(array_p2)
            # Update the error
            error2 = compute_error(array_n, array_m, array_q2, array_p2)
            step2 += 1
        else:
            break
        if step2 == iteration:
            print("Maximum iteration reached in the second case! Error={0}".format(error2))

    if error1 <= error2:
        array_p = array_p1
        array_q = array_q1
        error = error1
    else:
        array_p = array_p2
        array_q = array_q2
        error = error2

    return array_p, array_q, error


def _2sided_hungarian(profit_matrix):
    # Define the profit array & applying the hungarian algorithm
    cost_matrix = np.ones(profit_matrix.shape) * np.max(profit_matrix) - profit_matrix

    # Obtain the optimum permutation transformation and convert to array
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    perm_optimum = np.zeros(profit_matrix.shape)
    perm_optimum[row_ind, col_ind] = 1

    return perm_optimum


def _2sided_1trans_initial_guess_normal1(array_a):
    # This assumes that array_a has all positive entries, this guess does not match that found
    #    in the notes/paper because it doesn't include the sign function.
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
    # This assumes that array_a has all positive entries, this guess does not match that found
    #    in the notes/paper because it doesn't include the sign function.
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
    index_inf = np.argsort(-np.abs(a_inf), axis=1)

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


def _2sided_1trans_initial_guess_umeyama(array_a, array_b):
    # calculate the eigenvalue decomposition of A and B
    _, array_ua = np.linalg.eigh(array_a)
    _, array_ub = np.linalg.eigh(array_b)
    # compute U_umeyama
    array_u = np.dot(np.abs(array_ua), np.abs(array_ub.T))
    # compute closest permutation matrix to U
    # In the original paper, it"s not like this
    # _, _, U, _ = permutation(np.eye(U.shape[0], dtype=U.dtype), U)
    return array_u


def _2sided_1trans_initial_guess_umeyama_approx(array_a, array_b):
    # compute U_umeyama
    array_u = _2sided_1trans_initial_guess_umeyama(array_a, array_b)
    # calculate the approximated umeyama matrix
    array_ua, _, array_vta = scipy.linalg.svd(array_u, lapack_driver='gesvd')
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


def _guess_initial_permutation_undirected(array_a, array_b, mode):
    mode = mode.lower()
    if mode == "normal1":
        tmp_a = _2sided_1trans_initial_guess_normal1(array_a)
        tmp_b = _2sided_1trans_initial_guess_normal1(array_b)
        array_u = permutation(tmp_a, tmp_b)["t"]
    elif mode == "normal2":
        tmp_a = _2sided_1trans_initial_guess_normal2(array_a)
        tmp_b = _2sided_1trans_initial_guess_normal2(array_b)
        array_u = permutation(tmp_a, tmp_b)["t"]
    elif mode == "umeyama":
        array_u = _2sided_1trans_initial_guess_umeyama(array_a, array_b)
    elif mode == "umeyama_approx":
        array_u = _2sided_1trans_initial_guess_umeyama_approx(array_a, array_b)
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

    p_opt = permutation(
        a=np.eye(p_new.shape[0]),
        b=p_new,
        translate=False,
        scale=False,
        unpad_col=False,
        unpad_row=False,
        check_finite=True,
    )["t"]

    return p_opt


def _compute_transform_directed(array_a, array_b, guess, tol, iteration):
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
    p_opt = permutation(np.eye(p_new.shape[0]), p_new)["t"]

    return p_opt
