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
from procrustes.orthogonal import OrthogonalProcrustes
from procrustes.utils import eigenvalue_decomposition
import numpy as np


class TwoSidedPermutationSingleTransformationProcrustes(Procrustes):
    r"""
    Two-Sided Permutation Procrustes with Single-Transformation Class.

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
                              \atop \sum_{i=1}^n p_{ij} = \sum_{j=1}^n p_{ij} = 1} \right. \right\}}
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

    The answer ro this problem is a heuristic solution for the matrix-matching problem that seems
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
    """

    def __init__(self, array_a, array_b, translate=False, scale=False, guess='umeyama', maxiter=500,
                 threshold=1.e-5, refinement=True):
        r"""
        Initialize the class and transfer/scale the arrays followed by computing transformaions.

        Parameters
        ----------
        array_a : ndarray
            The symmetric 2d-array :math:`A_{m \times n}` which is going to be transformed.
        array_b : ndarray
            The symmetric 2d-array :math:`B_{m \times n}` represents the reference matrix.
        translate : bool, default=False
            If True, both arrays are translated to be centered at origin.
        scale : bool, default=False
            If True, both arrays are column normalized to unity.
        """
        super(self.__class__, self).__init__(array_a, array_b, translate, scale)

        # check arrau_a and array_b are symmetric
        diff_a = abs(self.array_a - self.array_a.T)
        diff_b = abs(self.array_b - self.array_b.T)
        if np.all(diff_a) > 1.e-10:
            raise ValueError('Array array_a should be symmetric.')
        if np.all(diff_b) > 1.e-10:
            raise ValueError('Array array_b should be symmetric.')

        self._guess = guess
        self.maxiter = maxiter
        self.threshold = threshold
        self.refinement = refinement

        # compute initial guess array, array_p and error
        self._array_g = self._get_initial_array()
        self._array_p = self._compute_transformation(self._array_g)
        self._error = self.double_sided_error(self._array_p, self._array_p)

    @property
    def array_p(self):
        r"""Transformation matrix :math:`\mathbf{P}`."""
        return self._array_p

    @property
    def error(self):
        """Double-sided Procrustes error."""
        return self._error

    @property
    def array_g(self):
        r"""Initial guess of matrix :math:`\mathbf{P}`."""
        return self._array_g

    @property
    def guess(self):
        """Initial guess scheme."""
        return self._guess

    def _get_initial_array(self):
        """
        """
        if self._guess == 'umeyama' or self._guess == 'umeyama_approx':
            # calculate the eigenvalue decomposition of array_a and array_b
            sigma_a, u_a = eigenvalue_decomposition(self.array_a, two_sided_single=True)
            sigma_b, u_b = eigenvalue_decomposition(self.array_b, two_sided_single=True)
            # compute u_umeyama
            array_u = np.multiply(abs(u_a), abs(u_b).T)
            if self._guess == 'umeyama_approx':
                # compute u_umeyama_apprxo (the closet unitary transformation to u_umeyama)
                ortho = OrthogonalProcrustes(np.eye(array_u.shape[0]), array_u)
                array_u = ortho.array_u
            # compute closest permutation matrix to array_u
            perm = PermutationProcrustes(np.eye(array_u.shape[0]), array_u)
            guess = perm.array_p

        elif guess == 'normal':
            raise NotImplementedError('')
            #
            # diag_a = np.diagonal(self.array_a)
            # diag_b = np.diagonal(self.array_b)

            # n_a, m_a = self.array_a.shape
            # n_a0, m_a0 = self.array_b.shape
            # diagonals_a = np.diagonal(self.array_a)
            # diagonals_a0 = np.diagonal(self.array_b)
            # b = np.zeros((n_a, m_a))
            # b0 = np.zeros((n_a0, m_a0))
            # b[0, :] = diagonals_a
            # b0[0, :] = diagonals_a0
            # # Populate remaining rows with columns of array_a sorted from greatest
            # # to least (excluding diagonals)
            # for i in range(n_a):
            #     col_a = self.array_a[i, :]  # Get the ith column of array_a
            #     col_a0 = self.array_b[i, :]
            #     col_a = np.delete(col_a, i)  # Remove the diagonal component
            #     col_a0 = np.delete(col_a0, i)
            #     # Sort the column from greatest to least
            #     idx_a = col_a.argsort()[::-1]
            #     idx_a0 = col_a0.argsort()[::-1]
            #     ordered_col_a = col_a[idx_a]
            #     ordered_col_a0 = col_a0[idx_a0]
            #     b[i, 1:n_a] = ordered_col_a  # Append the ordered column to array B
            #     b0[i, 1:n_a0] = ordered_col_a0
            # for i in range(1, m_a):
            #     # Scale each row by appropriate weighting factor
            #     b[i, :] = p**i * b[i, :]
            #     b0[i, :] = p**i * b0[i, :]
            # # Truncation criteria ; Truncate after this many rows
            # n_truncate = -2 * log(10) / log(p) + 1
            # truncate_rows = range(int(n_truncate), n_a)
            # b = np.delete(b, truncate_rows, axis=0)
            # b0 = np.delete(b0, truncate_rows, axis=0)

            # # Match the matrices b and b0 via the permutation procrustes problem
            # perm = PermutationProcrustes(b, b0)
            # perm_optimum3, array_transformed, total_potential, error = perm.calculate()
        return guess

    def _compute_transformation(self, guess):
        """
        """
        p_old = guess
        change = np.inf
        iteration = 0
        while change > self.threshold and iteration <= self.maxiter:
            # copute array_t
            t_old = np.dot(np.dot(self.array_a, p_old), self.array_b)
            # compute new array_p
            denom = np.dot(p_old, np.dot(p_old.T, t_old) + np.dot(p_old.T, t_old).T)
            p_new = p_old * np.sqrt(2 * t_old / denom)
            # compute change
            change = np.trace(np.dot((p_new - p_old).T, (p_new - p_old)))
            iteration += 1
        if iteration == self.maxiter:
            raise ValueError('Maximum iteration reached! Change={0}'.format(change))

        if self.refinement:
            # find closest permutation matrix to p_new
            proc = PermutationProcrustes(np.eye(p_new.shape[0]), p_new)
            p_new = proc.array_p

        return p_new
