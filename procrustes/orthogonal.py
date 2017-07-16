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
Orthogonal Procrustes Module.
"""


from procrustes.base import Procrustes
from procrustes.utils import singular_value_decomposition
import numpy as np


class OrthogonalProcrustes(Procrustes):
    r"""
    Orthogonal Procrustes Class.

    Given matrix :math:`\mathbf{A}_{m \times n}` and a reference :math:`\mathbf{B}_{m \times n}`,
    find the unitary/orthogonal transformation matrix :math:`\mathbf{U}_{n \times n}` that makes
    :math:`\mathbf{A}_{m \times n}` as close as possible to :math:`\mathbf{B}_{m \times n}`. I.e.,

    .. math::
       \underbrace{\min}_{\left\{\mathbf{U} | \mathbf{U}^{-1} = {\mathbf{U}}^\dagger \right\}}
                          \|\mathbf{A}\mathbf{U} - \mathbf{B}\|_{F}^2
       &= \underbrace{\text{min}}_{\left\{\mathbf{U} | \mathbf{U}^{-1} = {\mathbf{U}}^\dagger
                                   \right\}}
          \text{Tr}\left[\left(\mathbf{A}\mathbf{U} - \mathbf{B} \right)^\dagger
                         \left(\mathbf{A}\mathbf{U} - \mathbf{B} \right)\right] \\
       &= \underbrace{\text{max}}_{\left\{\mathbf{U} | \mathbf{U}^{-1} = {\mathbf{U}}^\dagger
                                   \right\}}
          \text{Tr}\left[\mathbf{U}^\dagger {\mathbf{A}}^\dagger \mathbf{B} \right]

    The solution is obtained by taking the singular value decomposition (SVD) of the product of the
    matrices,

    .. math::
       \mathbf{A}^\dagger \mathbf{B} &= \tilde{\mathbf{U}} \tilde{\mathbf{\Sigma}}
                                          \tilde{\mathbf{V}}^{\dagger} \\
       \mathbf{U}_{\text{optimum}} &= \tilde{\mathbf{U}} \tilde{\mathbf{V}}^{\dagger}

    The singular values are always listed in decreasing order, with the smallest singular
    value in the bottom-right-hand corner of :math:`\tilde{\mathbf{\Sigma}}`.
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
        The Procrustes analysis requires two 2d-arrays with the same number of rows, so the
        array with the smaller number of rows will automatically be padded with zero rows.
        """
        super(self.__class__, self).__init__(array_a, array_b, translate, scale)

        # compute transformation
        self._array_u = self._compute_transformation()

        # calculate the single-sided error
        self._error = self.single_sided_error(self._array_u)

    @property
    def array_u(self):
        r"""Transformation matrix :math:`\mathbf{U}_{n \times n}`."""
        return self._array_u

    @property
    def error(self):
        """Procrustes error."""
        return self._error

    def _compute_transformation(self):
        r"""
        Compute optimal orthogonal transformation array.

        Returns
        -------
        u_opt : ndarray
            The optimum orthogonal transformation array.
        """
        # calculate SVD of A.T * B
        product = np.dot(self.array_a.T, self.array_b)
        u, s, v_trans = singular_value_decomposition(product)

        # compute optimum orthogonal transformation
        u_opt = np.dot(u, v_trans)
        return u_opt
