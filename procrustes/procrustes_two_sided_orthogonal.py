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

from procrustes.base import Procrustes
from procrustes.utils import singular_value_decomposition
import numpy as np


class TwoSidedOrthogonalProcrustes(Procrustes):
    r"""
    This class deals with the orthogonal Procrustes problem.
    Given an :math:`m \times n \ \text{matrix} \ A`
    and a reference :math:`m \times n \ \text{matrix} \ A^0`, find two unitary/orthogonal transformation of
    :math:`A` that makes it as close as possible to :math:`A^0`. I.e.,

    .. math::
       \begin{array}{c} \underbrace{\min }_{\left\{{\begin{array}{*{20}{c}}
       {{{\bf{U}}_1}}\\{{{\bf{U}}_2}}
       \end{array}\left| {\begin{array}{*{20}{c}}
       {{\bf{U}}_1^{ - 1} = {\bf{U}}_1^\dagger }\\
       {{\bf{U}}_2^{ - 1} = {\bf{U}}_2^\dagger }
       \end{array}} \right.} \right\}}\left\| {{\bf{U}}_1^\dagger {\bf{A}}{{\bf{U}}_2} -
       {{\bf{A}}^0}} \right\|_F^2 = \underbrace {\min }_{\left\{{\begin{array}{*{20}{c}}
       {{{\bf{U}}_1}}\\{{{\bf{U}}_2}}
       \end{array}\left| {\begin{array}{*{20}{c}}
       {{\bf{U}}_1^{ - 1} = {\bf{U}}_1^\dagger }\\
       {{\bf{U}}_2^{ - 1} = {\bf{U}}_2^\dagger }
       \end{array}} \right.} \right\}}{\mathop{\rm Tr}\nolimits}
       \left[ {\left({{\bf{U}}_1^\dagger {\bf{A}}{{\bf{U}}_2} -
       {{\bf{A}}^0}} \right)_{}^\dagger \left({{\bf{U}}_1^\dagger {\bf{A}}{{\bf{U}}_2} -
       {{\bf{A}}^0}} \right)} \right]\\
        = \underbrace {\max }_{\left\{ {\begin{array}{*{20}{c}}
       {{{\bf{U}}_1}}\\
       {{{\bf{U}}_2}}
       \end{array}\left| {\begin{array}{*{20}{c}}
       {{\bf{U}}_1^{ - 1} = {\bf{U}}_1^\dagger }\\
       {{\bf{U}}_2^{ - 1} = {\bf{U}}_2^\dagger }
       \end{array}} \right.} \right\}}{\mathop{\rm Tr}\nolimits} \left[
       {{\bf{U}}_2^\dagger {\bf{A}}_{}^\dagger {{\bf{U}}_1}{{\bf{A}}^0}} \right]
       \end{array}

    We can get the solution by taking singular value decomposition of the matrices,

    .. math::
       \begin{array}{c}
       {\bf{A}} = {{\bf{U}}_A}{\Sigma _A}{\bf{V}}_A^\dagger \\
       {{\bf{A}}^0} = {{\bf{U}}_{{A^0}}}{\Sigma _{{A^0}}}{\bf{V}}_{{A^0}}^\dagger
       \end{array}

       \begin{array}{l}
       {{\bf{U}}_1} = {\bf{U}}_A^{}{\bf{U}}_{{A^0}}^\dagger \\
       {{\bf{U}}_2} = {\bf{V}}_A^{}{\bf{V}}_{{A^0}}^\dagger
       \end{array}

    References
    ----------
    1. Schönemann, Peter H. "A generalized solution of the orthogonal Procrustes problem." *Psychometrika* 31.1:1-10, 
    1966
    """

    def __init__(self, array_a, array_b, translate=False, scale=False):
        """
        Parameters
        ----------
        array_a : ndarray
            The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
        array_b : ndarray
            The 2d-array :math:`\mathbf{A}^0_{m \times n}` representing the reference.
        translate : bool, default = 'False'
            If True, both arrays are translated to be centered at origin.
        scale : bool, default = 'False'
            If True, both arrays are column normalized to unity.
        """

        Procrustes.__init__(self, array_a, array_b,
                            translate=translate, scale=scale)

    def calculate(self):
        """
        Calculates the two optimum two-sided orthogonal transformation arrays in the double-sided procrustes problem.

        Returns
        -------
        u1_opt : ndarray
           The optimum orthogonal left-multiplying transformation array satisfying the double sided procrustes problem.
        u2_opt : ndarray
           The optimum orthogonal right-multiplying transformation array satisfying the double sided procrustes problem.
        array_transformed : ndarray
           The transformed input array after the transformation.
        error : float
           The error as described by the double-sided procrustes problem.
        """

        # Calculate the SVDs of array_a and array_b & solve for the optimum
        # orthogonal transformation arrays
        u_a, sigma_a, v_trans_a = singular_value_decomposition(self.array_a)
        u_a0, sigma_a0, v_trans_a0 = singular_value_decomposition(self.array_b)
        u1_opt = np.dot(u_a, u_a0.T)
        u2_opt = np.dot(v_trans_a.T, v_trans_a0)

        # Calculate the error
        error = self.double_sided_error(u1_opt, u2_opt)

        # Calculate the transformed input array
        array_transformed = np.dot(np.dot(u1_opt.T, self.array_a), u2_opt)

        return u1_opt, u2_opt, array_transformed, error
