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
__author__ = 'Jonny'

from procrustes.base import Procrustes
from procrustes.procrustes_two_sided_orthogonal_single_transformation import TwoSidedOrthogonalSingleTransformationProcrustes
from procrustes.procrustes_permutation import PermutationProcrustes
import numpy as np
from math import log
from math import isnan


class TwoSidedPermutationSingleTransformationProcrustes(Procrustes):

    """
    This method deals with the two-sided orthogonal Procrustes problem
    limited to a single transformation

    We require symmetric input arrays to perform this analysis
    """

    """
    map_a_to_b is set to True by default. For the two-sided single transformation procrustes analyses, this is crucial
    for accuracy. When set to False, the input arrays both undergo centroid translation to the origin and
    Frobenius normalization, prior to further analysis. Something about this transformation skews the accuracy
    of the results.
    """

    """
    translate_scale for this analysis is False by default. The reason is that the inputs are really the outputs of two-
    sided single orthogonal procrustes, where translate_scale is True.
    """

    def __init__(self, array_a, array_b, translate=False, scale=False):

        Procrustes.__init__(self, array_a, array_b, translate=translate, scale=scale)


        if (abs(self.array_a - self.array_a.T) > 1.e-10).all() or (abs(self.array_b - self.array_b.T) > 1.e-10).all():
            raise ValueError('Arrays array_a and array_b must both be symmetric for this analysis.')

    def calculate(self, tol=1e-5, p=2.**(-.5)):
        """
        Calculates the single optimum two-sided permuation transformation matrix in the
        double-sided procrustes problem

        Parameters
        ----------
        array_a : ndarray
            A 2D array representing the array to be transformed (as close as possible to array_b)

        array_b : ndarray
            A 2D array representing the reference array

        Returns
        ----------
        perm_optimum, array_transformed, error
        perm_optimum= the optimum permutation transformation array satisfying the double
             sided procrustes problem. Array represents the closest permutation array to
             u_umeyama given by the permutation procrustes problem
        array_ transformed = the transformed input array after transformation by perm_optimum
        error = the error as described by the double-sided procrustes problem
        """

        """Finding initial guess"""

        # Method 1

        # Solve for the optimum initial permutation transformation array by finding the closest permutation
        # array to u_umeyama_approx given by the two-sided orthogonal single transformation problem
        twosided_ortho_single_trans = TwoSidedOrthogonalSingleTransformationProcrustes(self.array_a, self.array_b)
        u_approx, u_best, array_transformed_approx, array_transformed_best, error_approx, error_best = \
        twosided_ortho_single_trans.calculate(return_u_approx=True, return_u_best=True)

        # Find the closest permutation matrix to the u_optima obtained from TSSTO with permutation procrustes analysis
        perm1 = PermutationProcrustes(self.array_a, u_approx)
        perm2 = PermutationProcrustes(self.array_a, u_best)

        # Differentiate between exact and approx u_optima from two-sided single transformation orthogonal
        # initial guesses
        perm_optimum_trans1, array_transformed1, total_potential1, error1 = perm1.calculate()
        perm_optimum_trans2, array_transformed2, total_potential2, error2 = perm2.calculate()
        perm_optimum1 = perm_optimum_trans1.T  # Initial guess due to u_approx
        perm_optimum2 = perm_optimum_trans2.T  # Initial guess due to u_exact

        """
        Method two returns too high an error to be correctly coded
        """
        # Method 2
        n_a, m_a = self.array_a.shape
        n_a0, m_a0 = self.array_b.shape
        diagonals_a = np.diagonal(self.array_a)
        diagonals_a0 = np.diagonal(self.array_b)
        b = np.zeros((n_a, m_a))
        b0 = np.zeros((n_a0, m_a0))
        b[0, :] = diagonals_a
        b0[0, :] = diagonals_a0
        # Populate remaining rows with columns of array_a sorted from greatest to least (excluding diagonals)
        for i in range(n_a):
            col_a = self.array_a[i, :]  # Get the ith column of array_a
            col_a0 = self.array_b[i, :]
            col_a = np.delete(col_a, i)  # Remove the diagonal component
            col_a0 = np.delete(col_a0, i)
            idx_a = col_a.argsort()[::-1]  # Sort the column from greatest to least
            idx_a0 = col_a0.argsort()[::-1]
            ordered_col_a = col_a[idx_a]
            ordered_col_a0 = col_a0[idx_a0]
            b[i, 1:n_a] = ordered_col_a  # Append the ordered column to array B
            b0[i, 1:n_a0] = ordered_col_a0
        for i in range(1, m_a):
            b[i, :] = p**i * b[i, :]  # Scale each row by appropriate weighting factor
            b0[i, :] = p**i * b0[i, :]
        n_truncate = -2*log(10) / log(p) + 1  # Truncation criteria ; Truncate after this many rows
        truncate_rows = range(int(n_truncate), n_a)
        b = np.delete(b, truncate_rows, axis=0)
        b0 = np.delete(b0, truncate_rows, axis=0)

        # Match the matrices b and b0 via the permutation procrustes problem
        perm = PermutationProcrustes(b, b0)
        perm_optimum3, array_transformed, total_potential, error = perm.calculate()

        least_error_perm = perm_optimum1  # Arbitrarily initiate the least error perm. Will be adjusted in
        #  following procedure
        initial_perm_list = [perm_optimum1, perm_optimum2, perm_optimum3]
        min_error = 1.e8  # Arbitrarily initialize error ; will be adjusted in following procedure
        least_error_array_transformed = array_transformed1  # Arbitrarily initiate the least error transformed array

        error_to_beat = 1.  # Initialize error-to-beat

        for k in range(3):

            perm_optimum = initial_perm_list[k]
            """Beginning Iterative Procedure"""
            n, m = perm_optimum.shape

            # Initializing updated arrays. See literature for a full description of algorithm
            t_array = np.dot(np.dot(self.array_a, perm_optimum), self.array_b)
            p_new = perm_optimum
            p_old = perm_optimum
            # For simplicity, shorten t_array to t.
            t = t_array
            # Arbitrarily initialize error
            # Define breakouter, a boolean value which will skip the current method if NaN values occur

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
                            denom = np.dot(p_old, (np.dot(p_old.T, t)) + (np.dot(p_old.T, t)).T)[i, j]
                            if isnan(denom):
                                break_outer = False
                                break
                            factor = np.sqrt(abs(num / denom))
                            p_new[i, j] = p_old[i, j] * factor
                    error = np.trace(np.dot((p_new - p_old).T, (p_new - p_old)))
                break_outer = False

            """Converting optimal permutation (step 2) into permutation matrix """
            # Convert the array found above into a permutation matrix with permutation procrustes analysis
            perm = PermutationProcrustes(np.eye(n), p_new)
            perm_optimum, array_transformed, total_potential, error = perm.calculate()

            # Calculate the error
            error_perm_optimum = self.double_sided_error(perm_optimum, perm_optimum)

            if error_perm_optimum < error_to_beat:
                least_error_perm = perm_optimum
                min_error = error_perm_optimum
                least_error_array_transformed = array_transformed
                error_to_beat = error_perm_optimum
            else:
                continue
        return least_error_perm, least_error_array_transformed, min_error
