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
Permutation Procrustes Module.
"""


import numpy as np
from scipy.optimize import linear_sum_assignment
from procrustes.base import Procrustes


class PermutationProcrustes(Procrustes):
    r"""
    Permutation Procrustes Class.

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

    References
    ----------
    1. Harold W. Kuhn. The Hungarian Method for the assignment problem.
       Naval Research Logistics Quarterly, 2:83-97, 1955.
    2. Harold W. Kuhn. Variants of the Hungarian method for assignment problems.
       Naval Research Logistics Quarterly, 3: 253-258, 1956.
    3. Munkres, J. Algorithms for the Assignment and Transportation Problems.
       J. SIAM, 5(1):32-38, March, 1957.
    """

    def __init__(self, array_a, array_b, translate=False, scale=False):
        r"""
        Initialize the class and transfer/scale the arrays followed by computing transformation.

        Parameters
        ----------
        array_a : np.ndarray
            The 2d-array :math:`\mathbf{A}_{n \times n}` which is going to be transformed.
        array_b : np.ndarray
            The 2d-array :math:`\mathbf{B}_{n \times n}` representing the reference.
        translate : bool, default=False
            If True, both arrays are translated to be centered at origin.
        scale : bool, default=False
            If True, both arrays are column normalized to unity.

        Notes
        -----
        The Procrustes analysis requires two 2d-arrays with the same number of
        rows, so the array with the smaller number of rows will automatically
        be padded with zero rows.
        """
        super(self.__class__, self).__init__(array_a, array_b, translate, scale)

        # compute transformation
        self._array_p = self._compute_transformation()

        # calculate the single-sided error
        self._error = self.single_sided_error(self._array_p)

    @property
    def array_p(self):
        r"""Transformation matrix :math:`\mathbf{P}_{n \times n}`."""
        return self._array_p

    @property
    def error(self):
        """Procrustes error."""
        return self._error

    def _compute_transformation(self):
        """
        Compute optimum permutation transformation matrix in the single-sided procrustes problem.

        Returns
        -------
        perm_optimum : ndarray
            Permutation transformation matrix that satisfies the single sided procrustes problem.
        """
        # Define the profit array & applying the hungarian algorithm
        profit_matrix = np.dot(self.array_a.T, self.array_b)
        cost_matrix = np.ones(profit_matrix.shape) * np.max(profit_matrix) - profit_matrix

        # Obtain the optimum permutation transformation and convert to array
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        perm_optimum = np.zeros(profit_matrix.shape)
        perm_optimum[row_ind, col_ind] = 1

        return perm_optimum
