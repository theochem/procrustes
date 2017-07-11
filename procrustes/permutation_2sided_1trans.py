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

from procrustes.base import Procrustes
from procrustes.orthogonal_2sided_1trans import TwoSidedOrthogonalSingleTransformationProcrustes
from procrustes.permutation import PermutationProcrustes
import numpy as np
from math import log
from math import isnan


class TwoSidedPermutationSingleTransformationProcrustes(Procrustes):
    r"""
    Given a symmetric :math:`n \times n` matrix :math:`A` and a reference :math:`n \times n` matrix :math:`A^0` , find a
    permutation of the rows\/columns of :math:`A` that makes it as close as possible to :math:`A^0`.

    .. math::
       \begin{array}{c}
       \underbrace {\min }_{\left\{{{\bf{P}}\left| {\scriptstyle{p_{ij}} \in \left\{{0,1} \right\} \atop
       \scriptstyle\sum\limits_{i = 1}^n {{p_{ij}}}  = \sum\limits_{j = 1}^n {{p_{ij}}}  = 1 } \right.} \right\}}\left\|
       {{\bf{P}}^T{\bf{A}}{{\bf{P}}} - {{\bf{A}}^0}} \right\|_F^2 = \underbrace {\min }_{\left\{ {{\bf{P}}\left|
       {\scriptstyle{p_{ij}} \in \left\{ {0,1}\right\} \atop \scriptstyle\sum\limits_{i = 1}^n {{p_{ij}}}  =
       \sum\limits_{j = 1}^n {{p_{ij}}}  = 1 } \right.} \right\}}{\mathop{\rm Tr}\nolimits} \left[ {\left(
       {{\bf{P}}^T{\bf{A}}{{\bf{P}}} - {{\bf{A}}^0}} \right)^T\left({{\bf{P}}^T{\bf{A}}{{\bf{P}}} -
       {{\bf{A}}^0}} \right)} \right]\\
       = \underbrace {\max }_{\left\{ {{\bf{P}}\left| {\scriptstyle{p_{ij}} \in \left\{ {0,1} \right\} \atop
       \scriptstyle\sum\limits_{i = 1}^n {{p_{ij}}}  = \sum\limits_{j = 1}^n{{p_{ij}}}  = 1 } \right.}
       \right\}}{\mathop{\rm Tr}\nolimits} \left[ {{{\bf{P}}^T}{\bf{A}}^T{\bf{P}}{{\bf{A}}^0}} \right]
       \end{array}

    Given an intial guess, the best local minimum can be obtained by the iterative procedure.

    .. math::
       p_{ij}^{\left( {n + 1} \right)} = p_{ij}^{\left( n \right)}\sqrt {\frac{{2{{\left[ {{{\bf{T}}^{\left( n
       \right)}}} \right]}_{ij}}}}{{{{\left[ {{{\bf{P}}^{\left( n \right)}}\left( {{{\left( {{{\bf{P}}^{\left( n
       \right)}}} \right)}^T}{\bf{T}} + {{\left( {{{\left( {{{\bf{P}}^{\left( n \right)}}} \right)}^T}{\bf{T}}}
       \right)}^T}} \right)} \right]}_{ij}}}}}

    where

    .. math::
       {\bf{T}}^{\left(n\right)}={\bf{A}}{\bf{P}}^{\left(n\right)}{\bf{A}}^0

    **Step 1. Initial Guess**

    Two possible initial guesses are inferred from the Umeyama procedure. One can find either the closest permutation
    matrix to  :math:`{\bf{U}}_{Umeyama}` (Eq.23) or to :math:`{\bf{U}}_{Umeyama}^{\left(approx\right)}`.  I.e., two
    choices come from the permutation Procrustes problem with:

    .. math::
       \begin{array}{c}
       \underbrace {\min }_{\left\{ {{\bf{P}}\left| {\scriptstyle{p_{ij}} \in \left\{ {0,1} \right\} \atop
       \scriptstyle\sum\limits_{i = 1}^n {{p_{ij}}}  = \sum\limits_{j = 1}^n {{p_{ij}}}  = 1 } \right.} \right\}}\left\|
       {{\bf{P}} - {\bf{U}}} \right\|_F^2 = \underbrace {\min }_{\left\{ {{\bf{P}}\left| {\scriptstyle{p_{ij}} \in
       \left\{ {0,1} \right\} \atop
       \scriptstyle\sum\limits_{i = 1}^n {{p_{ij}}}  = \sum\limits_{j = 1}^n {{p_{ij}}}  = 1 } \right.}
       \right\}}{\mathop{\rm Tr}\nolimits} \left[ {\left( {{\bf{P}} - {\bf{U}}} \right)^\dagger \left( {{\bf{P}} -
       {\bf{U}}} \right)} \right]\\
       = \underbrace {\max }_{\left\{ {{\bf{P}}\left| {\scriptstyle{p_{ij}} \in \left\{ {0,1} \right\} \atop
       \scriptstyle\sum\limits_{i = 1}^n {{p_{ij}}}  = \sum\limits_{j = 1}^n {{p_{ij}}}  = 1 } \right.}
       \right\}}{\mathop{\rm Tr}\nolimits} \left[ {{\bf{P}}^\dagger {\bf{U}}} \right]
       \end{array}

    which gives two different assignment problems for the Hungarian algorithm,

    .. math::
       \underbrace {\max }_{\left\{ {{\bf{P}}\left| {\scriptstyle0 \le {p_{ij}} \atop
       \scriptstyle\sum\limits_{i = 1}^n {{p_{ij}}}
       = \sum\limits_{j = 1}^n {{p_{ij}}}  = 1 } \right.} \right\}}{\mathop{\rm Tr}\nolimits} \left[
       {{\bf{P}}^\dagger {{\bf{U}}_{{\rm{Umeyama}}}}} \right]

    .. math::
       \underbrace {\max }_{\left\{ {{\bf{P}}\left| {\scriptstyle0 \le {p_{ij}} \atop
       \scriptstyle\sum\limits_{i = 1}^n {{p_{ij}}}  = \sum\limits_{j = 1}^n {{p_{ij}}}  = 1 } \right.}
       \right\}}{\mathop{\rm Tr}\nolimits} \left[ {{\bf{P}}^\dagger {\bf{U}}_{{\rm{Umeyama}}}^{\left(
       {{\rm{approx}}{\rm{.}}} \right)}} \right]

    The permutations matrix that solves the problem is used as input into Eq.(28).

    Another choice is to start by solving a normal permutation Procrustes problem.  E.g., write new matrices, :math:`\bf{B}` and :math:`{\bf{B}}^0`, with columns like

    .. math::
       \left[
       {\begin{array}{*{20}{c}}
       {{a_{ii}}}\\
       {p \cdot {\mathop{\rm sgn}} \left( {{a_{i{j_{{\rm{max}}}}}}} \right)\underbrace {\max }_{1 \le j \le n}\left( {\left| {{a_{ij}}} \right|} \right)}\\
       {{p^2} \cdot {\mathop{\rm sgn}} \left( {{a_{i{j_{{\rm{\left( max-1 \right)}}}}}}} \right)\underbrace {\max {\rm{ - 1}}}_{1 \le j \le n}\left( {\left| {{a_{ij}}} \right|} \right)}\\
       \vdots
       \end{array}}
       \right]

    Here max-1 refers to the second-largest element (in absolute value), max-2 is the third-largest element in absolute
    value, etc..

    The matrix :math:`\bf{B}` (or :math:`{\bf{B}}^0` ) has the diagonal elements of :math:`\bf{A}` (or
    :math:`{\bf{A}}^0`) in the first row and below the first row has the largest off-diagonal element in row I, the
    second-largest off-diagonal element, etc.. These elements are weighted by a factor 0 < p < 1, so that smaller
    elements are considered less important for matching. (Perhaps choose :math:`p = 2^{-0.5}`.)  The matrices can be
    truncated after a few terms (perhaps after the size of the elements falls below some threshold; a reasonable choice
    would be to stop after :math:`m = \left\lfloor {\frac{{ - 2\ln 10}}{{\ln p}} + 1} \right\rfloor`) rows; this ensures
    that the size of the elements in the last row is less than 1% of those in first off-diagonal row.

    Then one uses the normal permutation Procrustes procedure to match the matrices :math:`\bf{B}` and
    :math:`{\bf{B}}^0` constructed by the preceding procedure. I.e.,

    .. math::
       \underbrace {\min }_{\left\{ {{\bf{P}}\left| {\scriptstyle{p_{ij}} \in \left\{ {0,1} \right\} \atop
       \scriptstyle\sum\limits_{i = 1}^n {{p_{ij}}}  = \sum\limits_{j = 1}^n{{p_{ij}}}  = 1 } \right.} \right\}}\left\|
       {{\bf{BP}} - {{\bf{B}}^0}} \right\|_F^2 = \underbrace {\max }_{\left\{ {{\bf{P}}\left| {\scriptstyle{p_{ij}} \in
       \left\{ {0,1} \right\} \atop \scriptstyle\sum\limits_{i = 1}^n {{p_{ij}}}  = \sum\limits_{j = 1}^n {{p_{ij}}}  =
       1 } \right.} \right\}}{\mathop{\rm Tr}\nolimits} \left[ {{\bf{P}}^\dagger {\bf{B}}^\dagger {{\bf{B}}^0}}
       \right]

    which we solve with the Hungarian methods,

    .. math::
       \underbrace {\max }_{\left\{ {{\bf{P}}\left| {\scriptstyle0 \le {p_{ij}} \atop \scriptstyle\sum\limits_{i = 1}^n
       {{p_{ij}}}  = \sum\limits_{j = 1}^n {{p_{ij}}}  = 1 } \right.} \right\}}{\mathop{\rm Tr}\nolimits} \left[
       {{\bf{P}}^\dagger \left( {{\bf{B}}^\dagger {{\bf{B}}^0}} \right)} \right]

    There are obviously many different ways to construct the matrices B.  Another, even better, method would be to try
    to encode not only what the off-diagonal elements are, but which element in the matrix they correspond to. One could
    do that by replacing each row in Eq.  by two rows, one of which lists the diagonal element and the other of which
    lists the associated off-diagonal element. I.e., the columns of :math:`\bf{B}` (or :math:`{\bf{B}}^0` ) would
    be,

    .. math::
       \left[
         {\begin{array}{*{20}{c}}
           {{a_{ii}}}\\
           {p \cdot {a_{{j_{\max }}{j_{\max }}}}}\\
           {p \cdot {\mathop{\rm sgn}}
           \left(
           {{a_{i{j_{{\rm{max}}}}}}}
           \right)
           \underbrace {\max }_{1  \le j \le n}\left( {\left| {{a_{ij}}}
           \right|} \right)}\\
           {{p^2} \cdot {a_{{j_{{\rm{\left( max-1 \right)}}}}{j_{{\rm{\left( max-1 \right)}}}}}}}\\
           {{p^2} \cdot {\mathop{\rm sgn}}
           \left(
           {{a_{i{j_{{\rm{max -  1}}}}}}}
           \right)
           \underbrace {\max {\rm{ - 1}}}_{1 \le j \le  n}
           \left(
           {\left|    {{a_{ij}}} \right|}
           \right)}\\
           \vdots
         \end{array}}
       \right]

    **Step 2. Iteration**

    Using one of the initial guesses obtained by solving the assignment problems in , , or , use the iteration procedure
    in

    .. math::
       {\mathop{\rm Tr}\nolimits} \left[ {{{\left( {{{\bf{P}}^{\left( {n + 1} \right)}} - {{\bf{P}}^{\left( n \right)}}}
       \right)}^T}\left( {{{\bf{P}}^{\left( {n + 1} \right)}} - {{\bf{P}}^{\left( n \right)}}} \right)} \right]

    Stop when the change in  is small enough.

    **Step 3. Refinment**

    The result of step 2 is not a permutation matrix.  So we have to find the closest permutation matrix, corresponding
    to the problem,

    .. math::
       \underbrace {\min }_{\left\{ {{\bf{P}}\left| {\scriptstyle{p_{ij}} \in \left\{ {0,1} \right\} \atop
       \scriptstyle\sum\limits_{i = 1}^n {{p_{ij}}}  = \sum\limits_{j = 1}^n {{p_{ij}}}  = 1 } \right.} \right\}}\left\|
       {{\bf{P}} - {{\bf{P}}^{\left( \infty  \right)}}} \right\|_F^2 = \underbrace {\max }_{\left\{ {{\bf{P}}\left|
       {\scriptstyle{p_{ij}} \in \left\{ {0,1} \right\} \atop \scriptstyle\sum\limits_{i = 1}^n {{p_{ij}}}  =
       \sum\limits_{j = 1}^n  {{p_{ij}}}  = 1 } \right.} \right\}}{\mathop{\rm Tr}\nolimits} \left[ {{\bf{P}}^\dagger
       {{\bf{P}}^{\left( \infty  \right)}}} \right]

    where :math:`{\bf{P}}^\infty` is the solution of step 2. We now have the Hungarian problem,

    .. math::
       \underbrace {\max }_{\left\{ {{\bf{P}}\left| {\scriptstyle0 \le {p_{ij}} \atop \scriptstyle\sum\limits_{i = 1}^n
           {{p_{ij}}}
       = \sum\limits_{j = 1}^n {{p_{ij}}}  = 1 } \right.} \right\}}{\mathop{\rm Tr}\nolimits} \left[
       {{\bf{P}}^\dagger {{\bf{P}}^{\left( \infty  \right)}}} \right]

    This method deals with the two-sided orthogonal Procrustes problem
    limited to a single transformation.
    We require symmetric input arrays to perform this analysis.

    Map_a_to_b is set to True by default. For the two-sided single transformation procrustes analyses, this is crucial
    for accuracy. When set to False, the input arrays both undergo centroid translation to the origin and
    Frobenius normalization, prior to further analysis. Something about this transformation skews the accuracy
    of the results.

    translate_scale for this analysis is False by default. The reason is that the inputs are really the outputs of two-
    sided single orthogonal procrustes, where translate_scale is True.
    """

    def __init__(self, array_a, array_b, translate=False, scale=False):
        r"""
        Initialize class.

        Parameters
        ----------
        array_a : ndarray
            The 2d-array :math: `A_{m \times n}` which is going to be transformed.
        array_b : ndarray
            The 2d-array :math: `A^0_{m \times n}` represents the reference matrix.
        translate : bool, default='False'
            If True, both arrays are translated to be centered at origin.
        scale : bool, default='False'
            If True, both arrays are column normalized to unity.
        """

        Procrustes.__init__(self, array_a, array_b,
                            translate=translate, scale=scale)

        if (abs(self.array_a - self.array_a.T) > 1.e-10).all() or (abs(self.array_b - self.array_b.T) > 1.e-10).all():
            raise ValueError(
                'Arrays array_a and array_b must both be symmetric for this analysis.')

    def calculate(self, tol=1e-5, p=2.**(-.5)):
        """
        Calculate the single optimum two-sided permuation transformation matrix in the
        double-sided procrustes problem

        Parameters
        ----------
        array_a : ndarray
            A 2D array representing the array to be transformed (as close as possible to array_b)

        array_b : ndarray
            A 2D array representing the reference array
        tol : float, default=1e-5
            Tolerance value.
        p : float, default=2.**(-.5)
            Weighting factor to scale the rows of the matrix.

        Returns
        -------
        perm_optimum : ndarray
            The optimum permutation transformation array satisfying the double
            sided procrustes problem. Array represents the closest permutation
            array to u_umeyama given by the permutation procrustes problem.
        array_ transformed : ndarroay
            The transformed input array after transformation by perm_optimum.
        error : float
             The error as described by the double-sided procrustes problem.
        """

        # Finding initial guess

        # Method 1

        # Solve for the optimum initial permutation transformation array by finding the closest permutation
        # array to u_umeyama_approx given by the two-sided orthogonal single
        # transformation problem
        twosided_ortho_single_trans = TwoSidedOrthogonalSingleTransformationProcrustes(
            self.array_a, self.array_b)
        u_approx, u_best, array_transformed_approx, array_transformed_best, error_approx, error_best = \
            twosided_ortho_single_trans.calculate(
                return_u_approx=True, return_u_best=True)

        # Find the closest permutation matrix to the u_optima obtained from
        # TSSTO with permutation procrustes analysis
        perm1 = PermutationProcrustes(self.array_a, u_approx)
        perm2 = PermutationProcrustes(self.array_a, u_best)

        # Differentiate between exact and approx u_optima from two-sided single transformation orthogonal
        # initial guesses
        perm_optimum_trans1, array_transformed1, total_potential1, error1 = perm1.calculate()
        perm_optimum_trans2, array_transformed2, total_potential2, error2 = perm2.calculate()
        perm_optimum1 = perm_optimum_trans1.T  # Initial guess due to u_approx
        perm_optimum2 = perm_optimum_trans2.T  # Initial guess due to u_exact

        # Method two returns too high an error to be correctly coded
        # Method 2
        n_a, m_a = self.array_a.shape
        n_a0, m_a0 = self.array_b.shape
        diagonals_a = np.diagonal(self.array_a)
        diagonals_a0 = np.diagonal(self.array_b)
        b = np.zeros((n_a, m_a))
        b0 = np.zeros((n_a0, m_a0))
        b[0, :] = diagonals_a
        b0[0, :] = diagonals_a0
        # Populate remaining rows with columns of array_a sorted from greatest
        # to least (excluding diagonals)
        for i in range(n_a):
            col_a = self.array_a[i, :]  # Get the ith column of array_a
            col_a0 = self.array_b[i, :]
            col_a = np.delete(col_a, i)  # Remove the diagonal component
            col_a0 = np.delete(col_a0, i)
            # Sort the column from greatest to least
            idx_a = col_a.argsort()[::-1]
            idx_a0 = col_a0.argsort()[::-1]
            ordered_col_a = col_a[idx_a]
            ordered_col_a0 = col_a0[idx_a0]
            b[i, 1:n_a] = ordered_col_a  # Append the ordered column to array B
            b0[i, 1:n_a0] = ordered_col_a0
        for i in range(1, m_a):
            # Scale each row by appropriate weighting factor
            b[i, :] = p**i * b[i, :]
            b0[i, :] = p**i * b0[i, :]
        # Truncation criteria ; Truncate after this many rows
        n_truncate = -2 * log(10) / log(p) + 1
        truncate_rows = range(int(n_truncate), n_a)
        b = np.delete(b, truncate_rows, axis=0)
        b0 = np.delete(b0, truncate_rows, axis=0)

        # Match the matrices b and b0 via the permutation procrustes problem
        perm = PermutationProcrustes(b, b0)
        perm_optimum3, array_transformed, total_potential, error = perm.calculate()

        # Arbitrarily initiate the least error perm. Will be adjusted in
        least_error_perm = perm_optimum1
        #  following procedure
        initial_perm_list = [perm_optimum1, perm_optimum2, perm_optimum3]
        min_error = 1.e8  # Arbitrarily initialize error ; will be adjusted in following procedure
        # Arbitrarily initiate the least error transformed array
        least_error_array_transformed = array_transformed1

        error_to_beat = 1.  # Initialize error-to-beat

        for k in range(3):

            perm_optimum = initial_perm_list[k]
            """Beginning Iterative Procedure"""
            n, m = perm_optimum.shape

            # Initializing updated arrays. See literature for a full
            # description of algorithm
            t_array = np.dot(np.dot(self.array_a, perm_optimum), self.array_b)
            p_new = perm_optimum
            p_old = perm_optimum
            # For simplicity, shorten t_array to t.
            t = t_array
            # Arbitrarily initialize error
            # Define breakouter, a boolean value which will skip the current
            # method if NaN values occur

            break_outer = True
            while break_outer:
                while error > tol:
                    for i in range(n):
                        for j in range(m):
                            # compute sqrt factor in (28)
                            num = 2 * t[i, j]
                            if isnan(num):
                                break_outer = False
                                break
                            # If the numerator (denominator) is NaN, skip the current method and
                            # move onto the next.
                            denom = np.dot(
                                p_old, (np.dot(p_old.T, t)) + (np.dot(p_old.T, t)).T)[i, j]
                            if isnan(denom):
                                break_outer = False
                                break
                            factor = np.sqrt(abs(num / denom))
                            p_new[i, j] = p_old[i, j] * factor
                    error = np.trace(
                        np.dot((p_new - p_old).T, (p_new - p_old)))
                break_outer = False

            #Converting optimal permutation (step 2) into permutation matrix
            # Convert the array found above into a permutation matrix with
            # permutation procrustes analysis
            perm = PermutationProcrustes(np.eye(n), p_new)
            perm_optimum, array_transformed, total_potential, error = perm.calculate()

            # Calculate the error
            error_perm_optimum = self.double_sided_error(
                perm_optimum, perm_optimum)

            if error_perm_optimum < error_to_beat:
                least_error_perm = perm_optimum
                min_error = error_perm_optimum
                least_error_array_transformed = array_transformed
                error_to_beat = error_perm_optimum
            else:
                continue
        return least_error_perm, least_error_array_transformed, min_error
