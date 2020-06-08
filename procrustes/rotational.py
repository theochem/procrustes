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
"""Rotational-Orthogonal Procrustes Module."""

import numpy as np
from procrustes.utils import error, setup_input_arrays


def rotational(array_a, array_b, remove_zero_col=True, remove_zero_row=True,
               pad_mode="row-col", translate=False, scale=False, check_finite=True):
    r"""
    Compute optimal rotational transformation array.

    The Procrustes analysis requires two 2d-arrays with the same number of rows, so the
    array with the smaller number of rows will automatically be padded with zero rows.

    Parameters
    ----------
    array_a : ndarray
        The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
    array_b : ndarray
        The 2d-array :math:`\mathbf{B}_{m \times n}` representing the reference array.
    remove_zero_col : bool, optional
        If True, zero columns (values less than 1e-8) on the right side will be removed.
        Default= True.
    remove_zero_row : bool, optional
        If True, zero rows (values less than 1e-8) on the bottom will be removed.
        Default= True.
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
        If True, both arrays are translated to be centered at origin, ie columns of the arrays
        will have mean zero.
        Default=False.
    scale : bool, optional
        If True, both arrays are normalized to one with respect to the Frobenius norm, ie
        :math:`Tr(A^T A) = 1`.
        Default=False.
    check_finite : bool, optional
        If true, convert the input to an array, checking for NaNs or Infs.

    Returns
    -------
    new_a : ndarray
        The transformed ndarray :math:`A`.
    new_b : ndarray
        The transformed ndarray :math:`B`.
    u_opt : ndarray
        The optimum rotation transformation matrix.
    e_opt : float
        One-sided orthogonal Procrustes error.

    Notes
    -----
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

    Examples
    --------
    >>> import numpy as np
    >>> array_a = np.array([[1.5, 7.4], [8.5, 4.5]])
    >>> array_b = np.array([[6.29325035,  4.17193001, 0., 0,], \
                            [9.19238816, -2.82842712, 0., 0.], \
                            [0.,          0.,         0., 0.]])
    >>> new_a, new_b, array_u, e_opt = rotational(array_a, array_b, translate=False, scale=False)
    >>> array_u # rotational array
    array([[ 0.70710678, -0.70710678],
           [ 0.70710678,  0.70710678]])
    >>> e_opt # error
    1.483808210011695e-17

    """
    # check inputs
    new_a, new_b = setup_input_arrays(array_a, array_b, remove_zero_col, remove_zero_row,
                                      pad_mode, translate, scale, check_finite)
    # compute SVD of A.T * A
    array_u, _, array_vt = np.linalg.svd(np.dot(new_a.T, new_b))
    # construct S which is an identity matrix with the smallest
    # singular value replaced by sgn(|U*V^t|).
    s_value = np.eye(new_a.shape[1])
    s_value[-1, -1] = np.sign(np.linalg.det(np.dot(array_u, array_vt)))
    # compute optimum rotation matrix
    u_opt = np.dot(np.dot(array_u, s_value), array_vt)
    # compute single-sided error error
    e_opt = error(new_a, new_b, u_opt)

    return new_a, new_b, u_opt, e_opt
