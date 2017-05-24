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


from procrustes.base import Procrustes
from procrustes.utils import hide_zero_padding, singular_value_decomposition
import numpy as np


class SymmetricProcrustes(Procrustes):
    r"""
    Given an :math:`m \times n` matrix :math:`A^0`, with
    :math:`m \geqslant n`,
    find the symmetric :math:`n \times n` matrix :math:`X` for
    which :math:`AX` is as close as possible to :math:`A^0`.

    .. math::
       \underbrace {\min }_{\left\{ {{\bf{X}}\left| {{\bf{X}} =
       {\bf{X}}_{}^\dagger } \right.} \right\}}\left\| {{\bf{AX}} -
       {{\bf{A}}^0}} \right\|_F^2 = \underbrace {\min }_{\left\{ {{\bf{X}}\left| {{\bf{X}} =
       {\bf{X}}_{}^\dagger } \right.} \right\}}{\mathop{\rm Tr}\nolimits} \left[
       {\left( {{\bf{AX}} - {{\bf{A}}^0}} \right)_{}^\dagger \left( {{\bf{AX}} -
       {{\bf{A}}^0}} \right)}
       \right]

    Define the singular value decomposition of :math:`A` as

    .. math::
       {\bf{A}} = {{\bf{U}}_{m \times m}}\left[
       \begin{array}{l}{\Sigma _{m \times m}}\\{{\bf{0}}_{m \times \left(
       {n - m} \right)}}\end{array} \right]
       {\bf{V}}_{n \times n}^\dagger

    A square diagonal :math:`n \times n` matrix with nonnegative elements is represented
    by :math:`\Sigma`, and it is consisted of :math:`\sigma_{i}` , listed in decreasing order.

    Define

    .. math::
       {{\bf{C}}_{m \times n}} = {\bf{U}}_{m \times m}^\dagger
       {\bf{A}}_{m \times n}^0{{\bf{V}}_{n \times n}}

    Then the elements of the optimal :math:`n \times n` matrix :math:`X` are

    .. math::
       {x_{ij}} =
       \left\{
       \begin{array}{l}
        0 & & i {\text{ and }} j > {\rm{rank}}\left( {{{\bf{A}}^0}}\right)\\
       \frac{{{\sigma_i}{c_{ij}} + {\sigma_j}{c_{ji}}}}{{\sigma_i^2 + \sigma_j^2}}
       &  & {\rm{otherwise}}
       \end{array}
       \right.

    Notice that the first part of this constrain works only only in the unusual case where
    :math:`A^0` has rank less than :math:`n`.
    """

    def __init__(self, array_a, array_b, translate=False, scale=False):
        r"""
        Initialize the class and transfer/scale the arrays followed by computing transformaion.

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
        array_a = hide_zero_padding(array_a)
        array_b = hide_zero_padding(array_b)

        if array_a.shape[0] < array_a.shape[1]:
            raise ValueError('The unpadding array_a should cannot have more columns than rows.')
        if array_a.shape[0] != array_b.shape[0]:
            raise ValueError('Arguments array_a & array_b should have the same number of rows.')
        if array_a.shape[1] != array_b.shape[1]:
            raise ValueError('Arguments array_a & array_b should have the same number of columns.')
        if np.linalg.matrix_rank(array_b) >= array_a.shape[1]:
            raise ValueError('Rand of array_b should be less than number of columns of array_a.')

        super(SymmetricProcrustes, self).__init__(array_a, array_b, translate, scale)

        # compute transformation
        self.array_x = self.compute_transformation()

        # calculate the single-sided error
        self.error = self.single_sided_error(self.array_x)

    def compute_transformation(self):
        r"""
        Return optimum right hand sided symmetric transformation array.

        Returns
        -------
        ndarray
            The symmetric transformation array.
        """
        # compute SVD of A
        u, s, v_trans = singular_value_decomposition(self.array_a)

        # add zeros to the eigenvalue array so it has length n
        n = self.array_a.shape[1]
        if len(s) < self.array_a.shape[1]:
            s = np.concatenate((s, np.zeros(n - len(s))))

        c = np.dot(np.dot(u.T, self.array_b), v_trans.T)
        # Create the intermediate array Y and the optimum symmetric transformation array X
        y = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if s[i]**2 + s[j]**2 == 0:
                    y[i, j] = 0
                else:
                    y[i, j] = (s[i]*c[i, j] + s[j]*c[j, i]) / (s[i]**2 + s[j]**2)

        x = np.dot(np.dot(v_trans.T, y), v_trans)

        return x


# Reference
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.112.4378&rep=rep1&type=pdf
