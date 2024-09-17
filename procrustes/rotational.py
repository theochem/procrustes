# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2024 The QC-Devs Community
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

from typing import Optional

import numpy as np
import scipy

from procrustes.utils import ProcrustesResult, compute_error, setup_input_arrays

__all__ = [
    "rotational",
]


def rotational(
    a: np.ndarray,
    b: np.ndarray,
    pad: bool = True,
    translate: bool = False,
    scale: bool = False,
    unpad_col: bool = False,
    unpad_row: bool = False,
    check_finite: bool = True,
    weight: Optional[np.ndarray] = None,
    lapack_driver: str = "gesvd",
) -> ProcrustesResult:
    r"""Perform rotational Procrustes.

    Given a matrix :math:`\mathbf{A}_{m \times n}` and a reference matrix :math:`\mathbf{B}_{m
    \times n}`, find the rotational transformation matrix :math:`\mathbf{R}_{n \times n}` that
    makes :math:`\mathbf{A}` as close as possible to :math:`\mathbf{B}`. In other words,

    .. math::
       \underbrace{\min}_{\left\{\mathbf{R} \left| {\mathbf{R}^{-1} = {\mathbf{R}}^\dagger
                                \atop \left| \mathbf{R} \right| = 1} \right. \right\}}
          \|\mathbf{A}\mathbf{R} - \mathbf{B}\|_{F}^2

    This Procrustes method requires the :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices to
    have the same shape, which is gauranteed with the default ``pad`` argument for any given
    :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices. In preparing the :math:`\mathbf{A}` and
    :math:`\mathbf{B}` matrices, the (optional) order of operations is: **1)** unpad zero
    rows/columns, **2)** translate the matrices to the origin, **3)** weight entries of
    :math:`\mathbf{A}`, **4)** scale the matrices to have unit norm, **5)** pad matrices with zero
    rows/columns so they have the same shape.

    Parameters
    ----------
    a : ndarray
        The 2D-array :math:`\mathbf{A}` which is going to be transformed.
    b : ndarray
        The 2D-array :math:`\mathbf{B}` representing the reference matrix.
    pad : bool, optional
        Add zero rows (at the bottom) and/or columns (to the right-hand side) of matrices
        :math:`\mathbf{A}` and :math:`\mathbf{B}` so that they have the same shape.
    translate : bool, optional
        If True, both arrays are centered at origin (columns of the arrays will have mean zero).
    scale : bool, optional
        If True, both arrays are normalized with respect to the Frobenius norm, i.e.,
        :math:`\text{Tr}\left[\mathbf{A}^\dagger\mathbf{A}\right] = 1` and
        :math:`\text{Tr}\left[\mathbf{B}^\dagger\mathbf{B}\right] = 1`.
    unpad_col : bool, optional
        If True, zero columns (with values less than 1.0e-8) on the right-hand side of the intial
        :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices are removed.
    unpad_row : bool, optional
        If True, zero rows (with values less than 1.0e-8) at the bottom of the intial
        :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices are removed.
    check_finite : bool, optional
        If True, convert the input to an array, checking for NaNs or Infs.
    weight : ndarray, optional
        The 1D-array representing the weights of each row of :math:`\mathbf{A}`. This defines the
        elements of the diagonal matrix :math:`\mathbf{W}` that is multiplied by :math:`\mathbf{A}`
        matrix, i.e., :math:`\mathbf{A} \rightarrow \mathbf{WA}`.
    lapack_driver : {'gesvd', 'gesdd'}, optional
        Whether to use the more efficient divide-and-conquer approach ('gesdd') or the more robust
        general rectangular approach ('gesvd') to compute the singular-value decomposition with
        `scipy.linalg.svd`.

    Returns
    -------
    res : ProcrustesResult
        The Procrustes result represented as a class:`utils.ProcrustesResult` object.

    Notes
    -----
    The optimal rotational matrix is obtained by,

    .. math::
       \mathbf{R}_{\text{opt}} =
       \arg \underbrace{\min}_{\left\{\mathbf{R} \left| {\mathbf{R}^{-1} = {\mathbf{R}}^\dagger
                               \atop \left| \mathbf{R} \right| = 1} \right. \right\}}
                               \|\mathbf{A}\mathbf{R} - \mathbf{B}\|_{F}^2 =
       \arg \underbrace{\max}_{\left\{\mathbf{R} \left| {\mathbf{R}^{-1} = {\mathbf{R}}^\dagger
                               \atop \left| \mathbf{R} \right| = 1} \right. \right\}}
                      \text{Tr}\left[\mathbf{R}^\dagger {\mathbf{A}}^\dagger \mathbf{B} \right]

    The solution is obtained by taking the singular value decomposition (SVD) of the
    :math:`\mathbf{A}^\dagger \mathbf{B}` matrix,

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
    >>> res.t   # rotational transformation
    array([[ 0.70710678, -0.70710678],
           [ 0.70710678,  0.70710678]])
    >>> res.error   # one-sided Procrustes error
    1.483808210011695e-17

    """
    # check inputs
    new_a, new_b = setup_input_arrays(
        a,
        b,
        unpad_col,
        unpad_row,
        pad,
        translate,
        scale,
        check_finite,
        weight,
    )
    if new_a.shape != new_b.shape:
        raise ValueError(
            f"Shape of A and B does not match: {new_a.shape} != {new_b.shape} "
            "Check pad, unpad_col, and unpad_row arguments."
        )
    # compute SVD of A.T * B
    u, _, vt = scipy.linalg.svd(np.dot(new_a.T, new_b), lapack_driver=lapack_driver)
    # construct S: an identity matrix with the smallest singular value replaced by sgn(|U*V^t|)
    s = np.eye(new_a.shape[1])
    s[-1, -1] = np.sign(np.linalg.det(np.dot(u, vt)))
    # compute optimal rotational transformation
    r_opt = np.dot(np.dot(u, s), vt)
    # compute one-sided error
    error = compute_error(new_a, new_b, r_opt)

    return ProcrustesResult(error=error, new_a=new_a, new_b=new_b, t=r_opt, s=None)
