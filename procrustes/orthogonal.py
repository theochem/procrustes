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
"""Orthogonal Procrustes Module."""

# import warnings

from typing import Optional

import numpy as np
import scipy

from procrustes.utils import ProcrustesResult, compute_error, setup_input_arrays

__all__ = [
    "orthogonal",
    "orthogonal_2sided",
]


def orthogonal(
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
    r"""Perform orthogonal Procrustes.

    Given a matrix :math:`\mathbf{A}_{m \times n}` and a reference matrix :math:`\mathbf{B}_{m
    \times n}`, find the orthogonal transformation matrix :math:`\mathbf{Q}_{n
    \times n}` that makes :math:`\mathbf{AQ}` as close as possible to :math:`\mathbf{B}`.
    In other words,

    .. math::
       \underbrace{\min}_{\left\{\mathbf{Q} | \mathbf{Q}^{-1} = {\mathbf{Q}}^\dagger \right\}}
                          \|\mathbf{A}\mathbf{Q} - \mathbf{B}\|_{F}^2

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
    The optimal orthogonal matrix is obtained by,

    .. math::
        \mathbf{Q}^{\text{opt}} =
        \arg \underbrace{\min}_{\left\{\mathbf{Q} \left| {\mathbf{Q}^{-1} = {\mathbf{Q}}^\dagger}
             \right. \right\}} \|\mathbf{A}\mathbf{Q} - \mathbf{B}\|_{F}^2 =
        \arg \underbrace{\max}_{\left\{\mathbf{Q} \left| {\mathbf{Q}^{-1} = {\mathbf{Q}}^\dagger}
             \right. \right\}} \text{Tr}\left[\mathbf{Q^\dagger}\mathbf{A^\dagger}\mathbf{B}\right]

    The solution is obtained using the singular value decomposition (SVD) of the
    :math:`\mathbf{A}^\dagger \mathbf{B}` matrix,

    .. math::
       \mathbf{A}^\dagger \mathbf{B} &= \tilde{\mathbf{U}} \tilde{\mathbf{\Sigma}}
                                          \tilde{\mathbf{V}}^{\dagger} \\
       \mathbf{Q}^{\text{opt}} &= \tilde{\mathbf{U}} \tilde{\mathbf{V}}^{\dagger}

    The singular values are always listed in decreasing order, with the smallest singular
    value in the bottom-right-hand corner of :math:`\tilde{\mathbf{\Sigma}}`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import ortho_group
    >>> from procrustes import orthogonal
    >>> a = np.random.rand(5, 3)   # random input matrix
    >>> q = ortho_group.rvs(3)     # random orthogonal transformation
    >>> b = np.dot(a, q) + np.random.rand(1, 3)   # random target matrix
    >>> result = orthogonal(a, b, translate=True, scale=False)
    >>> print(result.error)      # error (should be zero)
    >>> print(result.t)          # transformation matrix (same as q)
    >>> print(result.new_a)      # translated array a
    >>> print(result.new_b)      # translated array b

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
    # calculate SVD of A.T * B
    u, _, vt = scipy.linalg.svd(np.dot(new_a.T, new_b), lapack_driver=lapack_driver)
    # compute optimal orthogonal transformation
    u_opt = np.dot(u, vt)
    # compute one-sided error
    error = compute_error(new_a, new_b, u_opt)

    return ProcrustesResult(error=error, new_a=new_a, new_b=new_b, t=u_opt, s=None)


def orthogonal_2sided(
    a: np.ndarray,
    b: np.ndarray,
    single: bool = True,
    pad: bool = True,
    translate: bool = False,
    scale: bool = False,
    unpad_col: bool = False,
    unpad_row: bool = False,
    check_finite: bool = True,
    weight: Optional[np.ndarray] = None,
    lapack_driver: str = "gesvd",
) -> ProcrustesResult:
    r"""Perform two-sided orthogonal Procrustes with one- or two-transformations.

    **Two Transformations:** Given a matrix :math:`\mathbf{A}_{m \times n}` and a reference matrix
    :math:`\mathbf{B}_{m \times n}`, find two :math:`n \times n` orthogonal
    transformation matrices :math:`\mathbf{Q}_1^\dagger` and :math:`\mathbf{Q}_2` that makes
    :math:`\mathbf{Q}_1^\dagger\mathbf{A}\mathbf{Q}_2` as close as possible to :math:`\mathbf{B}`.
    In other words,

    .. math::
          \underbrace{\text{min}}_{\left\{ {\mathbf{Q}_1 \atop \mathbf{Q}_2} \left|
            {\mathbf{Q}_1^{-1} = \mathbf{Q}_1^\dagger \atop \mathbf{Q}_2^{-1} =
            \mathbf{Q}_2^\dagger} \right. \right\}}
            \|\mathbf{Q}_1^\dagger \mathbf{A} \mathbf{Q}_2 - \mathbf{B}\|_{F}^2

    **Single Transformations:** Given a **symmetric** matrix :math:`\mathbf{A}_{n \times n}` and
    a reference :math:`\mathbf{B}_{n \times n}`, find one orthogonal transformation
    matrix :math:`\mathbf{Q}_{n \times n}` that makes :math:`\mathbf{A}` as close as possible to
    :math:`\mathbf{B}`. In other words,

    .. math::
       \underbrace{\min}_{\left\{\mathbf{Q} | \mathbf{Q}^{-1} = {\mathbf{Q}}^\dagger \right\}}
                          \|\mathbf{Q}^\dagger\mathbf{A}\mathbf{Q} - \mathbf{B}\|_{F}^2

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
    single : bool, optional
        If True, single transformation is used (i.e., :math:`\mathbf{Q}_1=\mathbf{Q}_2=\mathbf{Q}`),
        otherwise, two transformations are used.
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
    lapack_driver : {"gesvd", "gesdd"}, optional
        Used in the singular value decomposition function from SciPy. Only allowed two options,
        with "gesvd" being less-efficient than "gesdd" but is more robust. Default is "gesvd".

    Returns
    -------
    res : ProcrustesResult
        The Procrustes result represented as a class:`utils.ProcrustesResult` object.

    Notes
    -----
    **Two-Sided Orthogonal Procrustes with Two Transformations:**
    The optimal orthogonal transformations are obtained by:

    .. math::
       \mathbf{Q}_{1}^{\text{opt}}, \mathbf{Q}_{2}^{\text{opt}} = \arg
          \underbrace{\text{min}}_{\left\{ {\mathbf{Q}_1 \atop \mathbf{Q}_2} \left|
            {\mathbf{Q}_1^{-1} = \mathbf{Q}_1^\dagger \atop \mathbf{Q}_2^{-1} =
            \mathbf{Q}_2^\dagger} \right. \right\}}
            \|\mathbf{Q}_1^\dagger \mathbf{A} \mathbf{Q}_2 - \mathbf{B}\|_{F}^2 = \arg
       \underbrace{\text{max}}_{\left\{ {\mathbf{Q}_1 \atop \mathbf{Q}_2} \left|
             {\mathbf{Q}_1^{-1} = \mathbf{Q}_1^\dagger \atop \mathbf{Q}_2^{-1} =
             \mathbf{Q}_2^\dagger} \right. \right\}}
          \text{Tr}\left[\mathbf{Q}_2^\dagger\mathbf{A}^\dagger\mathbf{Q}_1\mathbf{B} \right]

    This is solved by taking the singular value decomposition (SVD) of :math:`\mathbf{A}` and
    :math:`\mathbf{B}`,

    .. math::
       \mathbf{A} = \mathbf{U}_A \mathbf{\Sigma}_A \mathbf{V}_A^\dagger \\
       \mathbf{B} = \mathbf{U}_B \mathbf{\Sigma}_B \mathbf{V}_B^\dagger

    Then the two optimal orthogonal matrices are given by,

    .. math::
       \mathbf{Q}_1^{\text{opt}} = \mathbf{U}_A \mathbf{U}_B^\dagger \\
       \mathbf{Q}_2^{\text{opt}} = \mathbf{V}_A \mathbf{V}_B^\dagger

    **Two-Sided Orthogonal Procrustes with Single-Transformation:**
    The optimal orthogonal transformation is obtained by:

    .. math::
       \mathbf{Q}^{\text{opt}} = \arg
       \underbrace{\min}_{\left\{\mathbf{Q} | \mathbf{Q}^{-1} = {\mathbf{Q}}^\dagger \right\}}
                          \|\mathbf{Q}^\dagger\mathbf{A}\mathbf{Q} - \mathbf{B}\|_{F}^2 = \arg
       \underbrace{\text{max}}_{\left\{\mathbf{Q} | \mathbf{Q}^{-1} = {\mathbf{Q}}^\dagger\right\}}
          \text{Tr}\left[\mathbf{Q}^\dagger\mathbf{A}^\dagger\mathbf{Q}\mathbf{B} \right]

    Using the singular value decomposition (SVD) of :math:`\mathbf{A}` and :math:`\mathbf{B}`,

    .. math::
       \mathbf{A} = \mathbf{U}_A \mathbf{\Lambda}_A \mathbf{U}_A^\dagger \\
       \mathbf{B} = \mathbf{U}_B \mathbf{\Lambda}_B \mathbf{U}_B^\dagger

    The optimal orthogonal matrix :math:`\mathbf{Q}^\text{opt}` is obtained through,

    .. math::
       \mathbf{Q}^\text{opt} = \mathbf{U}_A \mathbf{S} \mathbf{U}_B^\dagger

    where :math:`\mathbf{S}` is a diagonal matrix with :math:`\pm{1}` elements,

    .. math::
       \mathbf{S} =
       \begin{bmatrix}
        { \pm 1} & 0       &\cdots &0 \\
        0        &{ \pm 1} &\ddots &\vdots \\
        \vdots   &\ddots   &\ddots &0\\
        0        &\cdots   &0      &{ \pm 1}
       \end{bmatrix}

    The matrix :math:`\mathbf{S}` is chosen to be the identity matrix.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[30, 33, 20], [33, 53, 43], [20, 43, 46]])
    >>> b = np.array([[ 22.78131838, -0.58896768,-43.00635291, 0., 0.],
    ...               [ -0.58896768, 16.77132475,  0.24289990, 0., 0.],
    ...               [-43.00635291,  0.2428999 , 89.44735687, 0., 0.],
    ...               [  0.        ,  0.        ,  0.        , 0., 0.]])
    >>> res = orthogonal_2sided(a, b, single=True, pad=True, unpad_col=True)
    >>> res.t
    array([[ 0.25116633,  0.76371527,  0.59468855],
           [-0.95144277,  0.08183302,  0.29674906],
           [ 0.17796663, -0.64034549,  0.74718507]])
    >>> res.error
    1.9646186414076689e-26

    """
    # if translate:
    #     warnings.warn(
    #         "The translation matrix was not well defined. \
    #             Two sided rotation and translation don't commute.",
    #         stacklevel=2,
    #     )

    # Check inputs
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

    # check symmetry if single_transform=True
    if single:
        if not np.allclose(new_a.T, new_a):
            raise ValueError(
                f"Array A with {new_a.shape} shape is not symmetric. "
                "Check pad, unpad_col, and unpad_row arguments."
            )
        if not np.allclose(new_b.T, new_b):
            raise ValueError(
                f"Array B with {new_b.shape} shape is not symmetric. "
                "Check pad, unpad_col, and unpad_row arguments."
            )

    # two-sided orthogonal Procrustes with one-transformations
    if single:
        _, ua = np.linalg.eigh(new_a)
        _, ub = np.linalg.eigh(new_b)
        u_opt = np.dot(ua, ub.T)
        # compute one-sided error
        error = compute_error(new_a, new_b, u_opt, u_opt.T)
        return ProcrustesResult(error=error, new_a=new_a, new_b=new_b, t=u_opt, s=u_opt.T)

    # two-sided orthogonal Procrustes with two-transformations
    ua, _, vta = scipy.linalg.svd(new_a, lapack_driver=lapack_driver)
    ub, _, vtb = scipy.linalg.svd(new_b, lapack_driver=lapack_driver)
    u_opt1 = np.dot(ua, ub.T)
    u_opt2 = np.dot(vta.T, vtb)
    error = compute_error(new_a, new_b, u_opt2, u_opt1.T)
    return ProcrustesResult(error=error, new_a=new_a, new_b=new_b, t=u_opt2, s=u_opt1.T)
