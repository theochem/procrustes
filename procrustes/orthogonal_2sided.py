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
Two-Sided Orthogonal Procrustes Module.
"""


from procrustes.base import Procrustes
from procrustes.utils import singular_value_decomposition
import numpy as np


class TwoSidedOrthogonalProcrustes(Procrustes):
    r"""
    Two-Sided Orthogonal Procrustes.

    Given matrix :math:`\mathbf{A}_{m \times n}` and a reference :math:`\mathbf{B}_{m \times n}`,
    find two unitary/orthogonal transformation of :math:`\mathbf{A}_{m \times n}` that makes it as
    as close as possible to :math:`\mathbf{B}_{m \times n}`. I.e.,

    .. math::
          \underbrace{\text{min}}_{\left\{ {\mathbf{U}_1 \atop \mathbf{U}_2} \left|
            {\mathbf{U}_1^{-1} = \mathbf{U}_1^\dagger \atop \mathbf{U}_2^{-1} =
            \mathbf{U}_2^\dagger} \right. \right\}}
            \|\mathbf{U}_1^\dagger \mathbf{A} \mathbf{U}_2 - \mathbf{B}\|_{F}^2
       &= \underbrace{\text{min}}_{\left\{ {\mathbf{U}_1 \atop \mathbf{U}_2} \left|
             {\mathbf{U}_1^{-1} = \mathbf{U}_1^\dagger \atop \mathbf{U}_2^{-1} =
             \mathbf{U}_2^\dagger} \right. \right\}}
        \text{Tr}\left[\left(\mathbf{U}_1^\dagger\mathbf{A}\mathbf{U}_2 - \mathbf{B} \right)^\dagger
                   \left(\mathbf{U}_1^\dagger\mathbf{A}\mathbf{U}_2 - \mathbf{B} \right)\right] \\
       &= \underbrace{\text{min}}_{\left\{ {\mathbf{U}_1 \atop \mathbf{U}_2} \left|
             {\mathbf{U}_1^{-1} = \mathbf{U}_1^\dagger \atop \mathbf{U}_2^{-1} =
             \mathbf{U}_2^\dagger} \right. \right\}}
          \text{Tr}\left[\mathbf{U}_2^\dagger\mathbf{A}^\dagger\mathbf{U}_1\mathbf{B} \right]

    We can get the solution by taking singular value decomposition of the matrices. Having,

    .. math::
       \mathbf{A} = \mathbf{U}_A \mathbf{\Sigma}_A \mathbf{V}_A^\dagger \\
       \mathbf{B} = \mathbf{U}_B \mathbf{\Sigma}_B \mathbf{V}_B^\dagger

    The transformation is foubd by,

    .. math::
       \mathbf{U}_1 = \mathbf{U}_A \mathbf{U}_B^\dagger \\
       \mathbf{U}_2 = \mathbf{V}_B \mathbf{V}_B^\dagger

    References
    ----------
    1. Schönemann, Peter H. "A generalized solution of the orthogonal Procrustes problem."
       *Psychometrika* 31.1:1-10, 1966.
    """

    def __init__(self, array_a, array_b, translate=False, scale=False):
        r"""
        Initialize the class and transfer/scale the arrays followed by computing transformaions.

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
        """
        super(self.__class__, self).__init__(array_a, array_b, translate, scale)

        # compute transformation
        self._array_u1, self._array_u2 = self._compute_transformation()

        # calculate the single-sided error
        self._error = self.double_sided_error(self._array_u1, self._array_u2)

    @property
    def array_u1(self):
        r"""Transformation matrix :math:`\mathbf{U}_1`."""
        return self._array_u1

    @property
    def array_u2(self):
        r"""Transformation matrix :math:`\mathbf{U}_2`."""
        return self._array_u2

    @property
    def error(self):
        """Procrustes error."""
        return self._error

    def _compute_transformation(self):
        """
        Compute optimal two-sided orthogonal transformation arrays.

        Returns
        -------
        u1_opt : ndarray
           The optimum orthogonal left-multiplying transformation array.
        u2_opt : ndarray
           The optimum orthogonal right-multiplying transformation array.
        """
        # calculate the SVDs of array_a and array_b
        u_a, sigma_a, v_trans_a = singular_value_decomposition(self.array_a)
        u_b, sigma_b, v_trans_b = singular_value_decomposition(self.array_b)
        # compute optimal orthogonal transformation arrays
        u1_opt = np.dot(u_a, u_b.T)
        u2_opt = np.dot(v_trans_a.T, v_trans_b)
        return u1_opt, u2_opt
