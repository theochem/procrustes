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
"""Permutation Procrustes Module."""
from typing import Optional

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

from procrustes.kopt import kopt_heuristic_double, kopt_heuristic_single
from procrustes.utils import ProcrustesResult, _zero_padding, compute_error, setup_input_arrays

__all__ = ["permutation", "permutation_2sided"]


def permutation(
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
        The 2D-array :math:`\mathbf{A}` which is going to be transformed.
    b : ndarray
        The 2D-array :math:`\mathbf{B}` representing the reference matrix.
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
    # if number of rows is less than column, the arrays are made square
    if (new_a.shape[0] < new_a.shape[1]) or (new_b.shape[0] < new_b.shape[1]):
        new_a, new_b = _zero_padding(new_a, new_b, "square")

    # compute cost matrix C = A.T B
    c = np.dot(new_a.T, new_b)
    # compute permutation matrix using Hungarian algorithm
    p = _compute_permutation_hungarian(c)
    # compute one-sided permutation error
    error = compute_error(new_a, new_b, p)

    return ProcrustesResult(new_a=new_a, new_b=new_b, t=p, error=error)


def permutation_2sided(
    a: np.ndarray,
    b: np.ndarray,
    single: bool = True,
    method: str = "kopt",
    guess_p1: Optional[np.ndarray] = None,
    guess_p2: Optional[np.ndarray] = None,
    pad: bool = False,
    unpad_col: bool = False,
    unpad_row: bool = False,
    translate: bool = False,
    scale: bool = False,
    check_finite: bool = True,
    options: Optional[dict] = None,
    weight: Optional[np.ndarray] = None,
    lapack_driver: str = "gesvd",
) -> ProcrustesResult:
    r"""Perform two-sided permutation Procrustes.

    Parameters
    ----------
    a : ndarray
        The 2D-array :math:`\mathbf{A}` which is going to be transformed.
    b : ndarray
        The 2D-array :math:`\mathbf{B}` representing the reference matrix.
    single : bool, optional
        If `True`, the single-transformation Procrustes is performed to obtain :math:`\mathbf{P}`.
        If `False`, the two-transformations Procrustes is performed to obtain :math:`\mathbf{P}_1`
        and :math:`\mathbf{P}_2`.
    method : str, optional
        The method to solve for permutation matrices. For `single=False`, these include "flip-flop"
        and "k-opt" methods. For `single=True`, these include "approx-normal1", "approx-normal2",
        "approx-umeyama", "approx-umeyama-svd", "k-opt", "soft-assign", and "nmf".
    guess_p1 : np.ndarray, optional
        Guess for :math:`\mathbf{P}_1` matrix given as a 2D-array. This is only required for the
        two-transformations case specified by setting `single=False`.
    guess_p2 : np.ndarray, optional
        Guess for :math:`\mathbf{P}_2` matrix given as a 2D-array.
    pad : bool, optional
        Add zero rows (at the bottom) and/or columns (to the right-hand side) of matrices
        :math:`\mathbf{A}` and :math:`\mathbf{B}` so that they have the same shape.
    unpad_col : bool, optional
        If True, zero columns (with values less than 1.0e-8) on the right-hand side are removed.
    unpad_row : bool, optional
        If True, zero rows (with values less than 1.0e-8) at the bottom are removed.
    translate : bool, optional
        If True, both arrays are centered at origin (columns of the arrays will have mean zero).
    scale : bool, optional
        If True, both arrays are normalized with respect to the Frobenius norm, i.e.,
        :math:`\text{Tr}\left[\mathbf{A}^\dagger\mathbf{A}\right] = 1` and
        :math:`\text{Tr}\left[\mathbf{B}^\dagger\mathbf{B}\right] = 1`.
    check_finite : bool, optional
        If True, convert the input to an array, checking for NaNs or Infs.
    options : dict, optional
       A dictionary of method options.
    weight : ndarray, optional
        The 1D-array representing the weights of each row of :math:`\mathbf{A}`. This defines the
        elements of the diagonal matrix :math:`\mathbf{W}` that is multiplied by :math:`\mathbf{A}`
        matrix, i.e., :math:`\mathbf{A} \rightarrow \mathbf{WA}`.
    lapack_driver : {'gesvd', 'gesdd'}, optional
        Whether to use the more efficient divide-and-conquer approach ('gesdd') or the more robust
        general rectangular approach ('gesvd') to compute the singular-value decomposition with
        `scipy.linalg.svd`.

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
    # check single argument
    if not isinstance(single, bool):
        raise TypeError(f"Argument single is not a boolean! Given type={type(single)}")

    # check inputs
    new_a, new_b = setup_input_arrays(
        a, b, unpad_col, unpad_row, pad, translate, scale, check_finite, weight
    )

    # check that A & B are square in case of single transformation
    if single and new_a.shape[0] != new_a.shape[1]:
        raise ValueError(
            f"For single={single}, matrix A should be square but A.shape={new_a.shape}"
            "Check pad, unpad_col, and unpad_row arguments."
        )

    if single and new_b.shape[0] != new_b.shape[1]:
        raise ValueError(
            f"For single={single}, matrix B should be square but B.shape={new_b.shape}"
            "Check pad, unpad_col, and unpad_row arguments."
        )

    # print a statement if user-specified guess is not used
    if method.startswith("approx") and guess_p1 is not None:
        print(f"Method={method} does not use an initial guess, so guess_p1 is ignored!")
    if method.startswith("approx") and guess_p2 is not None:
        print(f"Method={method} does not use an initial guess, so guess_p2 is ignored!")

    # get the number of rows & columns of matrix A
    m, n = new_a.shape

    # assign & check initial guess for P1
    if single and guess_p1 is not None:
        raise ValueError(f"For single={single}, P1 is transpose of P2, so guess_p1 should be None.")
    if not single:
        if guess_p1 is None:
            guess_p1 = np.eye(m)
        if guess_p1.shape != (m, m):
            raise ValueError(f"Argument guess_p1 should be either None or a ({m}, {m}) array.")

    # assign & check initial guess for P2
    if guess_p2 is None:
        guess_p2 = np.eye(n)
    if guess_p2.shape != (n, n):
        raise ValueError(f"Argument guess_p2 should be either None or a ({n}, {n}) array.")

    # check options dictionary & assign default keys
    defaults = {"tol": 1.0e-8, "maxiter": 500, "k": 3}
    if options is not None:
        if not isinstance(options, dict):
            raise ValueError(f"Argument options should be a dictionary. Given type={type(options)}")
        # pylint: disable=C0201
        if not all(k in defaults.keys() for k in options.keys()):
            raise ValueError(
                f"Argument options should only have {defaults.keys()} keys. "
                f"Given options contains {options.keys()} keys!"
            )
        # update defaults dictionary to use the specified options
        defaults.update(options)

    # 2-sided permutation Procrustes with two transformations
    # -------------------------------------------------------
    if not single:
        if method == "flip-flop":
            # compute permutations using flip-flop algorithm
            perm1, perm2, error = _permutation_2sided_2trans_flipflop(
                new_a, new_b, defaults["tol"], defaults["maxiter"], guess_p1, guess_p2
            )
        elif method == "k-opt":
            # compute permutations using k-opt heuristic search
            fun_error = lambda p1, p2: compute_error(new_a, new_b, p2, p1.T)
            perm1, perm2, error = kopt_heuristic_double(
                fun_error, p1=guess_p1, p2=guess_p2, k=defaults["k"]
            )
        else:
            raise ValueError(f"Method={method} not supported for single={single} transformation!")

        return ProcrustesResult(error=error, new_a=new_a, new_b=new_b, t=perm2, s=perm1)

    # 2-sided permutation Procrustes with one transformation
    # ------------------------------------------------------
    # The (un)directed iterative procedure for finding the permutation matrix takes the square
    # root of the matrix entries, which can result in complex numbers if the entries are
    # negative. To avoid this, all matrix entries are shifted (by the smallest amount) to be
    # positive. This causes no change to the objective function, as it's a constant value
    # being added to all entries of a and b.
    shift = 1.0e-6
    if np.min(new_a) < 0 or np.min(new_b) < 0:
        shift += abs(min(np.min(new_a), np.min(new_b)))
    # shift is a float, so even if new_a or new_b are ints, the positive matrices are floats
    # default shift is not zero to avoid division by zero later in the algorithm
    pos_a = new_a + shift
    pos_b = new_b + shift

    if method == "approx-normal1":
        tmp_a = _approx_permutation_2sided_1trans_normal1(a)
        tmp_b = _approx_permutation_2sided_1trans_normal1(b)
        perm = permutation(tmp_a, tmp_b).t

    elif method == "approx-normal2":
        tmp_a = _approx_permutation_2sided_1trans_normal2(a)
        tmp_b = _approx_permutation_2sided_1trans_normal2(b)
        perm = permutation(tmp_a, tmp_b).t

    elif method == "approx-umeyama":
        perm = _approx_permutation_2sided_1trans_umeyama(pos_a, pos_b)

    elif method == "approx-umeyama-svd":
        perm = _approx_permutation_2sided_1trans_umeyama_svd(a, b, lapack_driver)

    elif method == "k-opt":
        fun_error = lambda p: compute_error(pos_a, pos_b, p, p.T)
        perm, error = kopt_heuristic_single(fun_error, p0=guess_p2, k=defaults["k"])

    elif method == "soft-assign":
        raise NotImplementedError

    elif method == "nmf":
        # check whether A & B are symmetric (within a relative & absolute tolerance)
        is_pos_a_symmetric = np.allclose(pos_a, pos_a.T, rtol=1.0e-05, atol=1.0e-08)
        is_pos_b_symmetric = np.allclose(pos_b, pos_b.T, rtol=1.0e-05, atol=1.0e-08)

        if is_pos_a_symmetric and is_pos_b_symmetric:
            # undirected graph matching problem (iterative procedure)
            perm = _permutation_2sided_1trans_undirected(
                pos_a, pos_b, guess_p2, defaults["tol"], defaults["maxiter"]
            )
        else:
            # directed graph matching problem (iterative procedure)
            perm = _permutation_2sided_1trans_directed(
                pos_a, pos_b, guess_p2, defaults["tol"], defaults["maxiter"]
            )
    else:
        raise ValueError(f"Method={method} not supported for single={single} transformation!")

    # some of the methods for 2-sided-1-transformation permutation procrustes does not produce a
    # permutation matrix. So, their output is treated like a guess, and the closest permutation
    # matrix is found using 1-sided permutation procrustes (where A=I & B=perm)
    # Even though this step is not needed for ALL methods (e.g. k-opt, normal1, & normal2), to
    # make the code simple, this step is performed for all methods as its cost is negligible.
    perm = permutation(
        np.eye(perm.shape[0]),
        perm,
        translate=False,
        scale=False,
        unpad_col=False,
        unpad_row=False,
        check_finite=True,
    ).t
    # compute error
    error = compute_error(new_a, new_b, t=perm, s=perm.T)

    return ProcrustesResult(error=error, new_a=new_a, new_b=new_b, t=perm, s=perm.T)


def _permutation_2sided_2trans_flipflop(
    n: np.ndarray,
    m: np.ndarray,
    tol: float,
    max_iter: int,
    p0: Optional[np.ndarray] = None,
    q0: Optional[np.ndarray] = None,
):
    # two-sided permutation Procrustes with 2 transformations :math:` {\(\vert PNQ-M \vert\)}^2_F`
    # taken from page 64 in parallel solution of svd-related problems, with applications
    # Pythagoras Papadimitriou, University of Manchester, 1993

    # initial guesses: set P1 to identity if guess P0 is not given, and compute Q1 using 1-sided
    # permutation procrustes where A=(P1N), B=M, & cost = A.T B
    p1 = p0
    if p1 is None:
        p1 = np.eye(m.shape[0])
    q1 = _compute_permutation_hungarian(np.dot(np.dot(n.T, p1.T), m))
    # compute initial error1 = |(P1)N(Q1) - M|
    error1 = compute_error(n, m, q1, p1)

    step = 0
    while error1 > tol and step < max_iter:
        # update P1 using 1-sided permutation procrustes where A=(NQ1).T, B=M.T, & cost = A.T B
        # 1-sided procrustes finds the right-hand-side transformation T, so to solve for P1, one
        # needs to minimize |Q.T N.T P.T - M.T| which is the same as original objective function.
        p1 = _compute_permutation_hungarian(np.dot(np.dot(n, q1), m.T)).T
        # update Q1 using 1-sided permutation procrustes where A=(P1N).T, B=M, & cost = A.T B
        q1 = _compute_permutation_hungarian(np.dot(np.dot(n.T, p1.T), m))
        error1 = compute_error(n, m, q1, p1)
        step += 1
    if step == max_iter:
        print(f"Maximum iterations reached in 1st case of flip-flop! error={error1} & tol={tol}")

    # initial guesses: set Q2 to identity if guess Q0 is not given, and compute P2 using 1-sided
    # permutation procrustes where A=(NQ2).T, B=M.T, & cost = A.T B
    q2 = q0
    if q2 is None:
        q2 = np.eye(m.shape[1])
    p2 = _compute_permutation_hungarian(np.dot(np.dot(n, q2), m.T)).T
    # compute initial error2 = |(P2)N(Q2) - M|
    error2 = compute_error(n, m, q2, p2)

    step = 0
    while error2 > tol and step < max_iter:
        # update Q2 using 1-sided permutation procrustes where A=(P2N), B=M, & cost = A.T B
        q2 = _compute_permutation_hungarian(np.dot(np.dot(n.T, p2.T), m))
        # update P2 using 1-sided permutation procrustes where A=(NQ2).T, B=M.T, & cost = A.T B
        p2 = _compute_permutation_hungarian(np.dot(np.dot(n, q2), m.T)).T
        error2 = compute_error(n, m, q2, p2)
        step += 1
    if step == max_iter:
        print(f"Maximum iterations reached in 2nd case of flip-flop! error={error1} & tol={tol}")

    # return permutations corresponding to the lowest error
    if error1 <= error2:
        return p1, q1, error1
    return p2, q2, error2


def _compute_permutation_hungarian(cost_matrix: np.ndarray) -> np.ndarray:
    # solve linear sum assignment problem to get the row/column indices of optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    # make the permutation matrix by setting the corresponding elements to 1
    perm = np.zeros(cost_matrix.shape)
    perm[(row_ind, col_ind)] = 1
    return perm


def _approx_permutation_2sided_1trans_normal1(a: np.ndarray) -> np.ndarray:
    # This assumes that array_a has all positive entries, this guess does not match that found
    #    in the notes/paper because it doesn't include the sign function.
    # build the empty target array
    array_c = np.zeros(a.shape)
    # Fill the first row of array_c with diagonal entries
    array_c[0, :] = a.diagonal()
    array_mask = ~np.eye(a.shape[0], dtype=bool)
    # get all the non-diagonal element
    array_c_non_diag = (a[array_mask]).T.reshape(a.shape[0], a.shape[1] - 1)
    array_c_non_diag = array_c_non_diag[
        np.arange(np.shape(array_c_non_diag)[0])[:, np.newaxis], np.argsort(abs(array_c_non_diag))
    ]

    # form the right format in order to combine with matrix A
    array_c_sorted = np.fliplr(array_c_non_diag).T
    # fill the array_c with array_c_sorted
    array_c[1:, :] = array_c_sorted
    # the weight matrix
    weight_c = np.zeros(a.shape)
    weight_p = np.power(2, -0.5)

    for weight in range(a.shape[0]):
        weight_c[weight, :] = np.power(weight_p, weight)
    # build the new matrix array_new
    array_new = np.multiply(array_c, weight_c)

    return array_new


def _approx_permutation_2sided_1trans_normal2(a: np.ndarray) -> np.ndarray:
    # This assumes that array_a has all positive entries, this guess does not match that found
    #    in the notes/paper because it doesn't include the sign function.
    array_mask_a = ~np.eye(a.shape[0], dtype=bool)
    # array_off_diag0 is the off diagonal elements of A
    array_off_diag = a[array_mask_a].reshape((a.shape[0], a.shape[1] - 1))
    # array_off_diag1 is sorted off diagonal elements of A
    array_off_diag = array_off_diag[
        np.arange(np.shape(array_off_diag)[0])[:, np.newaxis], np.argsort(abs(array_off_diag))
    ]
    array_off_diag = np.fliplr(array_off_diag).T

    # array_c is newly built matrix B without weights
    # build array_c with the expected shape
    col_num_new = a.shape[0] * 2 - 1
    array_c = np.zeros((col_num_new, a.shape[1]))
    array_c[0, :] = a.diagonal()

    # use inf to represent the diagonal element
    a_inf = a - np.diag(np.diag(a)) + np.diag([-np.inf] * a.shape[0])
    index_inf = np.argsort(-np.abs(a_inf), axis=1)

    # the weight matrix
    weight_p = np.power(2, -0.5)
    weight_c = np.zeros((col_num_new, a.shape[1]))
    weight_c[0, :] = np.power(weight_p, 0)

    for index_col in range(1, a.shape[0]):
        # the index_col*2 row of array_c
        array_c[index_col * 2, :] = array_off_diag[index_col - 1, :]
        # the index_col*2-1 row of array_c
        array_c[index_col * 2 - 1, :] = a[index_inf[:, index_col], index_inf[:, index_col]]

        # the index_col*2 row of weight_c
        weight_c[index_col * 2, :] = np.power(weight_p, index_col)
        # the index_col*2 row of weight_c
        weight_c[index_col * 2 - 1, :] = np.power(weight_p, index_col)

    # the new matrix B
    array_new = np.multiply(array_c, weight_c)
    return array_new


def _approx_permutation_2sided_1trans_umeyama(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # check whether A & B are symmetric (within a relative & absolute tolerance)
    is_a_symmetric = np.allclose(a, a.T, rtol=1.0e-05, atol=1.0e-08)
    is_b_symmetric = np.allclose(b, b.T, rtol=1.0e-05, atol=1.0e-08)
    # symmetrize A & B if not symmetric
    if not (is_a_symmetric and is_b_symmetric):
        a = _symmetrize_matrix(a)
        b = _symmetrize_matrix(b)

    # compute normalized normalized eigenvector of A & B
    # in some cases, A and B can be complex matrix (when symmetrizing non-symmetric A & B),
    # the np.linalg.eigh returns the eigenvalues and eigenvectors of a complex Hermitian
    # (conjugate symmetric) or a real symmetric matrix.
    _, ua = np.linalg.eigh(a)
    _, ub = np.linalg.eigh(b)
    # for complex input, x + iy, the absolute value is np.sqrt(x**2 + y**2)
    u_umeyama = np.dot(np.abs(ua), np.abs(ub.T))
    return u_umeyama


def _approx_permutation_2sided_1trans_umeyama_svd(
    a: np.ndarray, b: np.ndarray, lapack_driver: str
) -> np.ndarray:
    # compute u_umeyama
    perm = _approx_permutation_2sided_1trans_umeyama(a, b)
    # compute approximated umeyama matrix
    u, _, vt = scipy.linalg.svd(perm, lapack_driver=lapack_driver)
    u_umeyama_approx = np.dot(np.abs(u), np.abs(vt))
    return u_umeyama_approx


def _symmetrize_matrix(a: np.ndarray) -> np.ndarray:
    # symmetrized matrix A would be complex
    return (a + a.T) * 0.5 + (a - a.T) * 0.5 * 1j


def _permutation_2sided_1trans_undirected(
    a: np.ndarray, b: np.ndarray, guess: np.ndarray, tol: float, iteration: int
) -> np.ndarray:
    """Solve for 2-sided permutation Procrustes with 1-transformation when A & B are symmetric."""

    p_old = guess
    change = np.inf
    step = 0

    while change > tol and step < iteration:
        # compute alpha matrix
        temp = np.dot(a, np.dot(p_old, b))
        alpha = np.dot(p_old.T, temp)
        alpha = (alpha + alpha.T) / 2
        # compute new permutation matrix & change
        p_new = p_old * np.sqrt(temp / np.dot(p_old, alpha))
        change = np.trace(np.dot((p_new - p_old).T, (p_new - p_old)))
        # update permutation matrix
        p_old = p_new
        step += 1

    if step == iteration:
        print(f"Maximum iteration reached! change={change} & tolerance={tol}")

    return p_new


def _permutation_2sided_1trans_directed(
    a: np.ndarray, b: np.ndarray, guess: np.ndarray, tol: float, iteration: int
) -> np.ndarray:
    """Solve for 2-sided permutation Procrustes with 1-transformation."""

    # Algorithm 2 from Appendix of Procrustes paper
    p_old = guess
    change = np.inf
    step = 0
    while change > tol and step < iteration:
        # compute alpha matrix
        tmp1 = np.dot(a, np.dot(p_old, b.T))
        tmp2 = np.dot(a.T, np.dot(p_old, b))
        alpha = 0.25 * np.dot(p_old.T, tmp1 + tmp2) + np.dot((tmp1 + tmp2).T, p_old)
        # compute new permutation matrix & change
        p_new = p_old * np.sqrt((tmp1 + tmp2) / (2 * np.dot(p_old, alpha)))
        change = np.trace(np.dot((p_new - p_old).T, (p_new - p_old)))
        # update permutation matrix
        p_old = p_new
        step += 1

    if step == iteration:
        print(f"Maximum iteration reached! change={change} & tolerance={tol}")

    return p_new
