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
"""
Symmetric Procrustes Module.
"""


from procrustes.base import Procrustes
from procrustes.utils import hide_zero_padding, singular_value_decomposition
import numpy as np


class SymmetricProcrustes(Procrustes):
    r"""
    Symmetric Procrustes Class.

    Given matrix :math:`\mathbf{A}_{m \times n}` and a reference :math:`\mathbf{B}_{m \times n}`,
    with :math:`m \geqslant n`, find the symmetric matrix :math:`\mathbf{X}_{n \times n}` for which
    :math:`\mathbf{AX}` is as close as possible to :math:`\mathbf{B}_{m \times n}`. I.e.,

    .. math::
       \underbrace{\text{min}}_{\left\{\mathbf{X} \left| \mathbf{X} = \mathbf{X}^\dagger
                                \right. \right\}} \|\mathbf{A} \mathbf{X} - \mathbf{B}\|_{F}^2 =
       \underbrace{\text{min}}_{\left\{\mathbf{X} \left| \mathbf{X} = \mathbf{X}^\dagger
                                \right. \right\}}
                \text{Tr}\left[\left(\mathbf{A}\mathbf{X} - \mathbf{B} \right)^\dagger
                         \left(\mathbf{A}\mathbf{X} - \mathbf{B} \right)\right]

    Considering the singular value decomposition of :math:`\mathbf{A}_{m \times n}` as

    .. math::
       \mathbf{A}_{m \times n} = \mathbf{U}_{m \times m} \begin{bmatrix}
                                 \mathbf{\Sigma}_{m \times m} \\
                                 \mathbf{0}_{m \times (n - m)} \end{bmatrix}
                                 \mathbf{V}_{n \times n}^\dagger

    where :math:`\mathbf{\Sigma}_{n \times n}` is a square diagonal matrix with nonnegative elements
    denoted by :math:`\sigma_i` listed in decreasing order, define

    .. math::
       \mathbf{C}_{m \times n} = \mathbf{U}_{m \times m}^\dagger
                                 \mathbf{A}_{m \times n}^0 \mathbf{V}_{n \times n}

    Then the elements of the optimal matrix :math:`\mathbf{X}_{n \times n}` are

    .. math::
       x_{ij} = \begin{cases}
              0 && i \text{ and } j > \text{rank} \left(\mathbf{B}\right) \\
              \frac{\sigma_i c_{ij} + \sigma_j c_{ji}}{\sigma_i^2 + \sigma_j^2} && \text{otherwise}
              \end{cases}

    Notice that the first part of this constrain only works in the unusual case where
    :math:`\mathbf{B}` has rank less than :math:`n`.

    References
    ----------
    1. Higham, Nicholas J. The Symmetric Procrustes problem.
       BIT Numerical Mathematics, 28 (1), 133-143, 1988.
    """

    def __init__(self, array_a, array_b, translate=False, scale=False):
        r"""
        Initialize the class and transfer/scale the arrays followed by computing transformaion.

        Parameters
        ----------
        array_a : ndarray
            The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
        array_b : ndarray
            The 2d-array :math:`\mathbf{B}_{m \times n}` representing the reference.
        translate : bool, default=False
            If True, both arrays are translated to be centered at origin.
        scale : bool, default=False
            If True, both arrays are column normalized to unity.

        Notes
        -----
        The symmetric procrustes analysis requires two 2d-arrays with the same number of rows, so
        the array with the smaller number of rows will automatically be padded with zero rows.
        """
        array_a = hide_zero_padding(array_a)
        array_b = hide_zero_padding(array_b)

        if array_a.shape[0] < array_a.shape[1]:
            raise ValueError('The unpadding array_a cannot have more columns than rows.')

        if array_a.shape[0] != array_b.shape[0]:
            raise ValueError('Arguments array_a & array_b should have the same number of rows.')

        if array_a.shape[1] != array_b.shape[1]:
            raise ValueError('Arguments array_a & array_b should have the same number of columns.')

        if np.linalg.matrix_rank(array_b) >= array_a.shape[1]:
            raise ValueError('Rand of array_b should be less than number of columns of array_a.')

        super(self.__class__, self).__init__(array_a, array_b, translate, scale)

        # compute transformation
        self._array_x = self._compute_transformation()

        # calculate the single-sided error
        self._error = self.single_sided_error(self._array_x)

    @property
    def array_x(self):
        r"""Transformation matrix :math:`\mathbf{X}_{n \times n}`."""
        return self._array_x

    @property
    def error(self):
        """Procrustes error."""
        return self._error

    def _compute_transformation(self):
        """
        Compute optimum right hand sided symmetric transformation array.

        Returns
        -------
        x_opt : ndarray
            The optimum symmetric transformation array.
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
                    y[i, j] = (s[i] * c[i, j] + s[j] * c[j, i]) / \
                        (s[i]**2 + s[j]**2)

        x_opt = np.dot(np.dot(v_trans.T, y), v_trans)

        return x_opt
