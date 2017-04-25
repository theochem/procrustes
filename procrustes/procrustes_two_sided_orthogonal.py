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
from procrustes.utils import singular_value_decomposition
import numpy as np


class TwoSidedOrthogonalProcrustes(Procrustes):

    """
    This method deals with the orthogonal Procrustes problem

    """

    def __init__(self, array_a, array_b, translate=False, scale=False):

        Procrustes.__init__(self, array_a, array_b, translate=translate, scale=scale)

    def calculate(self):
        """
        Calculates the two optimum two-sided orthogonal transformation arrays in the
        double-sided procrustes problem

        Parameters
        ----------
        array_a : ndarray
            A 2D array representing the array to be transformed (as close as possible to array_b)

        array_b : ndarray
            A 2D array representing the reference array

        Returns
        ----------
        u1, u2, array_transformed, error
        u1 = the optimum orthogonal left-multiplying transformation array satisfying the double
             sided procrustes problem
        u2 = the optimum orthogonal right-multiplying transformation array satisfying the double
             sided procrustes problem
        array_transformed = the transformed input array after the transformation U1* array_a*U2
        error = the error as described by the double-sided procrustes problem
        """
        # Calculate the SVDs of array_a and array_b & solve for the optimum orthogonal transformation arrays
        u_a, sigma_a, v_trans_a = singular_value_decomposition(self.array_a)
        u_a0, sigma_a0, v_trans_a0 = singular_value_decomposition(self.array_b)
        u1 = np.dot(u_a, u_a0.T)
        u2 = np.dot(v_trans_a.T, v_trans_a0)

        # Calculate the error
        error = self.double_sided_error(u1, u2)

        # Calculate the transformed input array
        array_transformed = np.dot(np.dot(u1.T, self.array_a), u2)

        return u1, u2, array_transformed, error
