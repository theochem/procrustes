# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2020 The Procrustes Development Team
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
"""Generic Procrustes Module."""

import numpy as np
from procrustes.utils import error, setup_input_arrays


def generic(array_a, array_b, remove_zero_col=True, remove_zero_row=True,
            pad_mode='row-col', translate=False, scale=False, check_finite=True):
    r"""
    Solve the generic right-sided Procrustes problem.

    The generic Procrustes solves the least squares optimization problem without any constraints. It
    assumed that each matrix has the same dimension, if not padding will occur.

    Parameters
    ----------
    array_a : ndarray
        The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
    array_b : ndarray
        The 2d-array :math:`\mathbf{B}_{m \times n}` representing the reference.
    remove_zero_col : bool, optional
        If True, the zero columns on the right side will be removed.
        Default=True.
    remove_zero_row : bool, optional
        If True, the zero rows on the top will be removed.
        Default=True.
    pad_mode : str, optional
        Specifying how to pad the arrays, listed below. Default="row-col".

            - "row"
                The array with fewer rows is padded with zero rows so that both have the same
                number of rows.
            - "col"
                The array with fewer columns is padded with zero columns so that both have the
                same number of columns.
            - "row-col"
                The array with fewer rows is padded with zero rows, and the array with fewer
                columns is padded with zero columns, so that both have the same dimensions.
                This does not necessarily result in square arrays.
            - "square"
                The arrays are padded with zero rows and zero columns so that they are both
                squared arrays. The dimension of square array is specified based on the highest
                dimension, i.e. :math:`\text{max}(n_a, m_a, n_b, m_b)`.
    translate : bool, optional
        If True, both arrays are translated to be centered at origin.
        Default=False.
    scale : bool, optional
        If True, both arrays are column normalized to unity. Default=False.
    check_finite : bool, optional
        If true, convert the input to an array, checking for NaNs or Infs.
        Default=True.

    Returns
    -------
    new_a : ndarray
        The transformed ndarray array_a.
    new_b : ndarray
        The transformed ndarray array_b.
    array_x : ndarray
        The optimum symmetric transformation array.
    e_opt : float
        One-sided Procrustes error.

    Notes
    -----
    Given a source matrix :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` and a target matrix
    :math:`\mathbf{B} \in \mathbb{R}^{m \times n}`, find the transformation matrix
    :math:`\mathbf{X} \in \mathbb{R}^{n \times n}` for which :math:`\mathbf{AX}` is as close as
    possible to :math:`\mathbf{B}`. I.e.,

    .. math::
       \text{min} \quad \|\mathbf{A} \mathbf{X} - \mathbf{B}\|_{F}^2

    The optimal transformation matrix :math:`\mathbf{X}` is given by

    .. math::
        \mathbf{X} = {(\mathbf{A}^{\top}\mathbf{A})}^{-1} \mathbf{A}^{\top} \mathbf{B}

    References
    ----------
    1. Gower, J. C. Procrustes Methods. Wiley Interdisciplinary Reviews: Computational Statistics,
       2(4), 503-508, 2010.

    """
    # check inputs
    new_a, new_b = setup_input_arrays(array_a, array_b, remove_zero_col, remove_zero_row,
                                      pad_mode, translate, scale, check_finite)
    # compute the generic solution
    a_inv = np.linalg.pinv(np.dot(new_a.T, new_a))
    array_x = np.linalg.multi_dot([a_inv, new_a.T, new_b])
    e_opt = error(new_a, new_b, array_x)
    return new_a, new_b, array_x, e_opt
