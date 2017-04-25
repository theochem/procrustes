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
Base Procrustes Module.
"""


import numpy as np
from utils import zero_padding, hide_zero_padding, translate_array, scale_array


class Procrustes(object):
    """
    Base Procrustes Class.
    """

    def __init__(self, array_a, array_b, translate=False, scale=False):
        r"""
        Initialize the class and transfer/scale the arrays.

        Parameters
        ----------
        array_a : ndarray
            The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
        array_b : ndarray
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
        # sanity checks of type and dimension
        if not isinstance(array_a, np.ndarray) or not isinstance(array_b, np.ndarray):
            raise ValueError('Arguments array_a and array_b should be of numpy.ndarray type.')
        if array_a.ndim != 2 or array_b.ndim != 2:
            raise ValueError('Arguments array_a and array_b should be 2D arrays.')

        # remove already padded zero rows and columns (important for translation)
        array_a = hide_zero_padding(array_a)
        array_b = hide_zero_padding(array_b)

        if translate:
            array_a, self.translate_a = translate_array(array_a)
            array_b, self.translate_b = translate_array(array_b)

        if scale:
            array_a, self.scale_a = scale_array(array_a)
            array_b, self.scale_b = scale_array(array_b)

        # make sure the arrays have the same number of rows
        # if not, pad the one with smaller number of rows with zero.
        if array_a.shape[0] != array_b.shape[0]:
            array_a, array_b = zero_padding(array_a, array_b, mode='row')

        self.array_a = array_a
        self.array_b = array_b

    def single_sided_error(self, array_u):
        r"""
        Return the single-sided procrustes error.

         .. math::
            \text{Tr}\left[\left(\mathbf{AU} - \mathbf{A}^0\right)^\dagger
                           \left(\mathbf{AU} - \mathbf{A}^0\right)\right]

        Parameters
        ----------
        array_u : ndarray
           A 2D array representing the transformation :math:`\mathbf{U}`.

        Returns
        -------
        float
            The single-sided error.
        """
        au = np.dot(self.array_a, array_u)
        error = np.trace(np.dot((au - self.array_b).T, au - self.array_b))
        return error

    def double_sided_error(self, array_u1, array_u2):
        r"""
        Return the double-sided procrustes error.

         .. math::
            \text{Tr}\left[\left(\mathbf{U_1}^\dagger \mathbf{AU_2} - \mathbf{A}^0\right)^\dagger
                           \left(\mathbf{U_1}^\dagger \mathbf{AU_2} - \mathbf{A}^0\right)\right]

        Parameters
        ----------
        array_u1 : ndarray
           A 2D array representing the 1st transformation :math:`\mathbf{U_1}`.
        array_u2 : ndarray
           A 2D array representing the 2nd transformation :math:`\mathbf{U_2}`.

        Returns
        -------
        float
            The double-sided error.
        """
        u1au2 = np.dot(np.dot(array_u1.T, self.array_a), array_u2)
        error = np.trace(np.dot((u1au2 - self.array_b).T, u1au2 - self.array_b))
        return error
