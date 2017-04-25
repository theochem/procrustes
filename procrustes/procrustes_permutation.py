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
"""Solve the Permutation Procrustes Problem.

Find the permutation of the rows and columns of one matrix such that it most closely resembles
another matrix
"""


import numpy as np
from scipy.optimize import linear_sum_assignment
from procrustes.base import Procrustes


class PermutationProcrustes(Procrustes):
    """
    Permutation Procrustes Class.
    """

    def __init__(self, array_a, array_b, translate=False, scale=False):
        """
        Initialize the class and transfer/scale the arrays followed by computing transformation.

        Parameters
        ----------
        array_a : np.ndarray(m,n)
            The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
        array_b : np.ndarray(m,n)
            The 2d-array :math:`\mathbf{A}^0_{m \times n}` representing the reference.
        translate : bool
            If True, both arrays are translated to be centered at origin, default=False.
        scale : bool
            If True, both arrays are column normalized to unity, default=False.

        Notes
        -----
        The Procrustes analysis requires two 2d-arrays with the same number of rows, so the
        array with the smaller number of rows will automatically be padded with zero rows.
        """
        super(PermutationProcrustes, self).__init__(array_a, array_b, translate, scale)

        # compute transformation
        self.array_p = self.compute_transformation()

        # calculate the single-sided error
        self.error = self.single_sided_error(self.array_p)

    def compute_transformation(self):
        """
        Find the optimum permutation transformation matrix in the single-sided procrustes problem.

        Returns
        -------
        perm_optimum : np.ndarray(N, M)
            Permutation transformation matrix that satisfies the single sided procrustes problem
        """

        # Define the profit array & applying the hungarian algorithm
        profit_array = np.dot(self.array_a.T, self.array_b)
        cost_matrix = np.ones(profit_matrix.shape) * np.max(profit_matrix) - profit_matrix

        # Obtain the optimum permutation transformation and convert to array form
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        perm_optimum = np.zeros(profit_matrix.shape)
        perm_optimum[row_ind, col_ind] = 1

        return perm_optimum
