# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2021 The QC-Devs Community
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
from procrustes.utils import compute_error, ProcrustesResult, setup_input_arrays


def rotational(
    a,
    b,
    remove_zero_col=True,
    remove_zero_row=True,
    pad_mode="row-col",
    translate=False,
    scale=False,
    check_finite=True,
    weight=None,
):
    r"""Perform rotational Procrustes.

    This Procrustes method requires the :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices
    to have the same shape. If this is not the case, the arguments `pad_mode`, `remove_zero_row`,
    and `remove_zero_col` can be used to make them have the same shape.

    Parameters
    ----------
    a : ndarray
        The 2D-array :math:`\mathbf{A}` which is going to be transformed.
    b : ndarray
        The 2D-array :math:`\mathbf{B}` representing the reference array.
    remove_zero_col : bool, optional
        If True, zero columns (values less than 1e-8) on the right side will be removed.
    remove_zero_row : bool, optional
        If True, zero rows (values less than 1e-8) on the bottom will be removed.
    translate : bool, optional
        If True, arrays are centered at origin, i.e., columns of the arrays will have mean zero.
    pad_mode : str, optional
        Specifying how to pad the arrays, listed below.

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
    scale : bool, optional
        If True, arrays are normalized to one with respect to the Frobenius norm, i.e.,
        :math:`\text{Tr}\left[\mathbf{A}^\dagger\mathbf{A}\right] = 1` and
        :math:`\text{Tr}\left[\mathbf{B}^\dagger\mathbf{B}\right] = 1`.
    check_finite : bool, optional
        If True, convert the input to an array, checking for NaNs or Infs.
    weight : ndarray
        The weighting matrix.

    Returns
    -------
    res : ProcrustesResult
        The Procrustes result represented as a class:`utils.ProcrustesResult` object.

    Notes
    -----
    Given matrix :math:`\mathbf{A}_{m \times n}` and a reference matrix :math:`\mathbf{B}_{m \times
    n}`, find the rotational transformation matrix :math:`\mathbf{R}_{n \times n}` that makes
    :math:`\mathbf{A}` as close as possible to :math:`\mathbf{B}`. In other words,

    .. math::
       \underbrace{\min}_{\left\{\mathbf{R} \left| {\mathbf{R}^{-1} = {\mathbf{R}}^\dagger
                                \atop \left| \mathbf{R} \right| = 1} \right. \right\}}
          \|\mathbf{A}\mathbf{R} - \mathbf{B}\|_{F}^2
       &= \underbrace{\min}_{\left\{\mathbf{R} \left| {\mathbf{R}^{-1} = {\mathbf{R}}^\dagger
                                   \atop \left| \mathbf{R} \right| = 1} \right. \right\}}
          \text{Tr}\left[\left(\mathbf{A}\mathbf{R} - \mathbf{B} \right)^\dagger
                         \left(\mathbf{A}\mathbf{R} - \mathbf{B} \right)\right] \\
       &= \underbrace{\max}_{\left\{\mathbf{R} \left| {\mathbf{R}^{-1} = {\mathbf{R}}^\dagger
                                   \atop \left| \mathbf{R} \right| = 1} \right. \right\}}
          \text{Tr}\left[\mathbf{R}^\dagger {\mathbf{A}}^\dagger \mathbf{B} \right]

    The optimal rotational matrix :math:`\mathbf{R}_{\text{opt}}` is obtained using the singular
    value decomposition (SVD) of the :math:`\mathbf{A}^\dagger \mathbf{B}` matrix through,

    .. math::
       \mathbf{A}^\dagger \mathbf{B} &= \tilde{\mathbf{U}} \tilde{\mathbf{\Sigma}}
                                          \tilde{\mathbf{V}}^{\dagger} \\
       \mathbf{R}_{\text{opt}} &= \tilde{\mathbf{U}} \tilde{\mathbf{S}}
                                      \tilde{\mathbf{V}}^{\dagger}

    where :math:`\tilde{\mathbf{S}}_{n \times m}` is almost an identity matrix,

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

    in which the smallest singular value is replaced by

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
    >>> array_b = np.array([[6.29325035,  4.17193001, 0., 0,],
    ...                     [9.19238816, -2.82842712, 0., 0.],
    ...                     [0.,          0.,         0., 0.]])
    >>> res = rotational(array_a,array_b,translate=False,scale=False)
    >>> res.t   # rotational array
    array([[ 0.70710678, -0.70710678],
           [ 0.70710678,  0.70710678]])
    >>> res.error   # one-sided Procrustes error
    1.483808210011695e-17

    """
    # check inputs
    new_a, new_b = setup_input_arrays(
        a,
        b,
        remove_zero_col,
        remove_zero_row,
        pad_mode,
        translate,
        scale,
        check_finite,
        weight,
    )
    if new_a.shape != new_b.shape():
        raise ValueError(f"Shape of A and B does not match: {new_a.shape} != {new_b.shape} "
                         "Check pad_mode, remove_zero_col, and remove_zero_row options.")
    # compute SVD of A.T * B
    u, _, vt = np.linalg.svd(np.dot(new_a.T, new_b))
    # construct S: an identity matrix with the smallest singular value replaced by sgn( |U*V^t|)
    s = np.eye(new_a.shape[1])
    s[-1, -1] = np.sign(np.linalg.det(np.dot(u, vt)))
    # compute optimal rotation matrix
    r_opt = np.dot(np.dot(u, s), vt)
    # compute single-sided error error
    error = compute_error(new_a, new_b, r_opt)

    return ProcrustesResult(error=error, new_a=new_a, new_b=new_b, t=r_opt, s=None)
