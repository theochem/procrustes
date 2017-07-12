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
Rotational Orthogonal Procrustes Module.
"""


from procrustes.base import Procrustes
from procrustes.utils import singular_value_decomposition
import numpy as np


class RotationalOrthogonalProcrustes(Procrustes):
    r"""
    Rotational Orthogonal Procrustes Class.

    Given matrix :math:`\mathbf{A}_{m \times n}` and a reference :math:`\mathbf{B}_{m \times n}`,
    find the transformation of :math:`\mathbf{A}_{m \times n}` that makes it as close as possible
    to :math:`\mathbf{B}_{m \times n}`. I.e.,

    .. math::
       \underbrace{\min}_{\left\{\mathbf{U} \left| {\mathbf{U}^{-1} = {\mathbf{U}}^\dagger
                                \atop \left| \mathbf{U} \right| = 1} \right. \right\}}
          \|\mathbf{A}\mathbf{U} - \mathbf{B}\|_{F}^2
       &= \underbrace{\min}_{\left\{\mathbf{U} \left| {\mathbf{U}^{-1} = {\mathbf{U}}^\dagger
                                   \atop \left| \mathbf{U} \right| = 1} \right. \right\}}
          \text{Tr}\left[\left(\mathbf{A}\mathbf{U} - \mathbf{B} \right)^\dagger
                         \left(\mathbf{A}\mathbf{U} - \mathbf{B} \right)\right] \\
       &= \underbrace{\max}_{\left\{\mathbf{U} \left| {\mathbf{U}^{-1} = {\mathbf{U}}^\dagger
                                   \atop \left| \mathbf{U} \right| = 1} \right. \right\}}
          \text{Tr}\left[\mathbf{U}^\dagger {\mathbf{A}}^\dagger \mathbf{B} \right]

    Here, :math:`\mathbf{U}_{n \times n}` is the permutation matrix. The solution is obtained by
    taking the singular value decomposition (SVD) of the product of the matrix,

    .. math::
       \mathbf{A}^\dagger \mathbf{B} &= \tilde{\mathbf{U}} \tilde{\mathbf{\Sigma}}
                                          \tilde{\mathbf{V}}^{\dagger} \\
       \mathbf{U}_{\text{optimum}} &= \tilde{\mathbf{U}} \tilde{\mathbf{S}}
                                      \tilde{\mathbf{V}}^{\dagger}

    Where :math:`\tilde{\mathbf{S}}_{n \times m}` is almost an identity matrix,

    .. math::
       \tilde{\mathbf{S}}_{m \times n} \equiv
       \begin{bmatrix}
           1  &  0  &  \cdots  &  0   &  0 \\
           0  &  1  &  \ddots  & \vdots &0 \\
           0  & \ddots &\ddots & 0 &\vdots \\
           \vdots&0 & 0        & 1     &0 \\
           0 & 0 & 0 \cdots &0 &\operatorname{sgn}
                                \left(\left|\mathbf{U}\mathbf{V}^\dagger\right|\right)
       \end{bmatrix}

    I.e. the smallest singular value is replaced by

    .. math::
       \operatorname{sgn} \left(\left|\tilde{\mathbf{U}} \tilde{\mathbf{V}}^\dagger\right|\right) =
       \begin{cases}
        +1 \qquad \left|\tilde{\mathbf{U}} \tilde{\mathbf{V}}^\dagger\right| \geq 0 \\
        -1 \qquad \left|\tilde{\mathbf{U}} \tilde{\mathbf{V}}^\dagger\right| < 0
       \end{cases}
    """

    def __init__(self, array_a, array_b, translate=False, scale=False):
        r"""
        Initialize the class and transfer/scale the arrays followed by computing transformaion.

        Parameters
        ----------
        array_a : ndarray
            The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
        array_b : ndarray
            The 2d-array :math:`\mathbf{B}_{m \times n}` represents the reference matrix.
        translate : bool
            If True, both arrays are translated to be centered at origin, default=False.
        scale : bool
            If True, both arrays are column normalized to unity, default=False.

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
        """
        Compute optimal rotational-orthogonal transformation array.

        Returns
        -------
        u_opt : ndarray
            The optimum rotational-orthogonal transformation array.
        """
        # compute SVD of A.T * A
        product = np.dot(self.array_a.T, self.array_b)
        u, s, v_trans = singular_value_decomposition(product)

        # construct S which is an identity matrix with the smallest
        # singular value replaced by sgn(|U*V^t|).
        array_s = np.eye(self.array_a.shape[1])
        array_s[-1, -1] = np.sign(np.linalg.det(np.dot(u, v_trans)))

        # compute optimum rotation matrix
        u_opt = np.dot(np.dot(u, array_s), v_trans)

        return u_opt
