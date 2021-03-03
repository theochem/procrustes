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
"""Orthogonal Procrustes Module."""

import warnings

import numpy as np
from procrustes.utils import compute_error, ProcrustesResult, setup_input_arrays

__all__ = [
    "orthogonal",
    "orthogonal_2sided",
]


def orthogonal(
    a,
    b,
    pad=True,
    translate=False,
    scale=False,
    remove_zero_col=False,
    remove_zero_row=False,
    check_finite=True,
    weight=None,
):
    r"""Perform orthogonal Procrustes.

    This Procrustes method requires the :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices to
    have the same shape. If this is not the case, the arguments `pad`, `remove_zero_col`, and
    `remove_zero_row` can be used to make them have the same shape.

    Parameters
    ----------
    a : ndarray
        The 2d-array :math:`\mathbf{A}` which is going to be transformed.
    b : ndarray
        The 2d-array :math:`\mathbf{B}` representing the reference matrix.
    pad : bool, optional
        Add zero rows (at the bottom) and/or columns (to the right-hand side) of matrices
        :math:`\mathbf{A}` and :math:`\mathbf{B}` so that they have the same shape.
    translate : bool, optional
        If True, both arrays are centered at origin (columns of the arrays will have mean zero).
    scale : bool, optional
        If True, both arrays are normalized with respect to the Frobenius norm, i.e.,
        :math:`\text{Tr}\left[\mathbf{A}^\dagger\mathbf{A}\right] = 1` and
        :math:`\text{Tr}\left[\mathbf{B}^\dagger\mathbf{B}\right] = 1`.
    remove_zero_col : bool, optional
        If True, zero columns (with values less than 1.0e-8) on the right-hand side are removed.
    remove_zero_row : bool, optional
        If True, zero rows (with values less than 1.0e-8) at the bottom are removed.
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
    n}`, find the orthogonal (i.e., unitary) transformation matrix :math:`\mathbf{U}_{n \times n}`
    that makes :math:`\mathbf{A}` as close as possible to :math:`\mathbf{B}`. In other words,

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

    The optimal orthogonal matrix :math:`\mathbf{U}_{\text{opt}}` is obtained using the singular
    value decomposition (SVD) of the :math:`\mathbf{A}^\dagger \mathbf{B}` matrix through,

    .. math::
       \mathbf{A}^\dagger \mathbf{B} &= \tilde{\mathbf{U}} \tilde{\mathbf{\Sigma}}
                                          \tilde{\mathbf{V}}^{\dagger} \\
       \mathbf{U}_{\text{opt}} &= \tilde{\mathbf{U}} \tilde{\mathbf{V}}^{\dagger}

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
        remove_zero_col,
        remove_zero_row,
        pad,
        translate,
        scale,
        check_finite,
        weight,
    )
    if new_a.shape != new_b.shape:
        raise ValueError(
            f"Shape of A and B does not match: {new_a.shape} != {new_b.shape} "
            "Check pad, remove_zero_col, and remove_zero_row arguments."
        )
    # calculate SVD of A.T * B
    u, _, vt = np.linalg.svd(np.dot(new_a.T, new_b))
    # compute optimal orthogonal transformation
    u_opt = np.dot(u, vt)
    # compute one-sided error
    error = compute_error(new_a, new_b, u_opt)

    return ProcrustesResult(error=error, new_a=new_a, new_b=new_b, t=u_opt, s=None)


def orthogonal_2sided(
    a,
    b,
    single_transform=True,
    pad=True,
    translate=False,
    scale=False,
    remove_zero_col=False,
    remove_zero_row=False,
    check_finite=True,
    weight=None,
):
    r"""Perform two-sided orthogonal Procrustes with one- or two-transformations.

    This Procrustes method requires the :math:`\mathbf{A}` and :math:`\mathbf{B}` matrices to
    have the same shape and be **symmetric** when using one-transformation matrix. If this is not
    the case, the arguments `pad`, `remove_zero_col`, and `remove_zero_row` can be used to make them
    have the same shape.

    Parameters
    ----------
    a : ndarray
        The 2d-array :math:`\mathbf{A}` which is going to be transformed.
    b : ndarray
        The 2d-array :math:`\mathbf{B}` representing the reference matrix.
    single_transform : bool, optional
        If True, one transformation is used, otherwise, two transformations are used.
    pad : bool, optional
        Add zero rows (at the bottom) and/or columns (to the right-hand side) of matrices
        :math:`\mathbf{A}` and :math:`\mathbf{B}` so that they have the same shape.
    translate : bool, optional
        If True, both arrays are centered at origin (columns of the arrays will have mean zero).
    scale : bool, optional
        If True, both arrays are normalized with respect to the Frobenius norm, i.e.,
        :math:`\text{Tr}\left[\mathbf{A}^\dagger\mathbf{A}\right] = 1` and
        :math:`\text{Tr}\left[\mathbf{B}^\dagger\mathbf{B}\right] = 1`.
    remove_zero_col : bool, optional
        If True, zero columns (with values less than 1.0e-8) on the right-hand side are removed.
    remove_zero_row : bool, optional
        If True, zero rows (with values less than 1.0e-8) at the bottom are removed.
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
    **Two-Sided Orthogonal Procrustes:** Given a matrix :math:`\mathbf{A}_{m \times n}` and a
    reference matrix :math:`\mathbf{B}_{m \times n}`, find two orthogonal (i.e., unitary)
    transformation matrices :math:`\mathbf{U}_{n \times n}` that makes :math:`\mathbf{A}` as
    close as possible to :math:`\mathbf{B}`. In other words,

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

    Using the singular value decomposition (SVD) of :math:`\mathbf{A}` and :math:`\mathbf{B}`,

    .. math::
       \mathbf{A} = \mathbf{U}_A \mathbf{\Sigma}_A \mathbf{V}_A^\dagger \\
       \mathbf{B} = \mathbf{U}_B \mathbf{\Sigma}_B \mathbf{V}_B^\dagger

    The two optimal orthogonal matrices are obtained through,

    .. math::
       \mathbf{U}_1 = \mathbf{U}_A \mathbf{U}_B^\dagger \\
       \mathbf{U}_2 = \mathbf{V}_B \mathbf{V}_B^\dagger

    **Two-Sided Orthogonal Procrustes with Single-Transformation:** Given a **symmetric** matrix
    :math:`\mathbf{A}_{n \times n}` and a reference :math:`\mathbf{B}_{n \times n}`, find one
    orthogonal (i.e., unitary) transformation matrix :math:`\mathbf{U}_{n \times n}` that makes
    :math:`\mathbf{A}` as close as possible to :math:`\mathbf{B}`. I.e.,

    .. math::
       \underbrace{\min}_{\left\{\mathbf{U} | \mathbf{U}^{-1} = {\mathbf{U}}^\dagger \right\}}
                          \|\mathbf{U}^\dagger\mathbf{A}\mathbf{U} - \mathbf{B}\|_{F}^2
       &= \underbrace{\text{min}}_{\left\{\mathbf{U} | \mathbf{U}^{-1} = {\mathbf{U}}^\dagger
                                   \right\}}
          \text{Tr}\left[\left(\mathbf{U}^\dagger\mathbf{A}\mathbf{U} - \mathbf{B} \right)^\dagger
                         \left(\mathbf{U}^\dagger\mathbf{A}\mathbf{U} - \mathbf{B} \right)\right] \\
       &= \underbrace{\text{max}}_{\left\{\mathbf{U} | \mathbf{U}^{-1} = {\mathbf{U}}^\dagger
                                   \right\}}
          \text{Tr}\left[\mathbf{U}^\dagger\mathbf{A}^\dagger\mathbf{U}\mathbf{B} \right]

    Using the singular value decomposition (SVD) of :math:`\mathbf{A}` and :math:`\mathbf{B}`,

    .. math::
       \mathbf{A} = \mathbf{U}_A \mathbf{\Lambda}_A \mathbf{U}_A^\dagger \\
       \mathbf{B} = \mathbf{U}_B \mathbf{\Lambda}_B \mathbf{U}_B^\dagger

    The optimal orthogonal matrix :math:`\mathbf{U}_\text{opt}` is obtained through,

    .. math::
       \mathbf{U}_\text{opt} = \mathbf{U}_A \mathbf{S} \mathbf{U}_A^\dagger

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

    Please note that the translation operation is not well defined for two sided orthogonal
    procrustes since two sided rotation and translation don't commute. Therefore, please be careful
    when setting translate=True.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[30, 33, 20], [33, 53, 43], [20, 43, 46]])
    >>> b = np.array([[ 22.78131838, -0.58896768,-43.00635291, 0., 0.],
    ...               [ -0.58896768, 16.77132475,  0.24289990, 0., 0.],
    ...               [-43.00635291,  0.2428999 , 89.44735687, 0., 0.],
    ...               [  0.        ,  0.        ,  0.        , 0., 0.]])
    >>> res = orthogonal_2sided(a, b, single_transform=True, pad=True, remove_zero_col=True)
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
        remove_zero_col,
        remove_zero_row,
        pad,
        translate,
        scale,
        check_finite,
        weight,
    )

    # check symmetry if single_transform=True
    if single_transform:
        if not np.allclose(new_a.T, new_a):
            raise ValueError(
                f"Array A with {new_a.shape} shape is not symmetric. "
                "Check pad, remove_zero_col, and remove_zero_row arguments."
            )
        if not np.allclose(new_b.T, new_b):
            raise ValueError(
                f"Array B with {new_b.shape} shape is not symmetric. "
                "Check pad, remove_zero_col, and remove_zero_row arguments."
            )

    if single_transform:
        # two-sided orthogonal Procrustes with one-transformations
        _, ua = np.linalg.eigh(new_a)
        _, ub = np.linalg.eigh(new_b)
        u_opt = np.dot(ua, ub.T)
        # compute one-sided error
        error = compute_error(new_a, new_b, u_opt, u_opt)
        return ProcrustesResult(error=error, new_a=new_a, new_b=new_b, t=u_opt, s=u_opt)

    # two-sided orthogonal Procrustes with two-transformations
    ua, _, vta = np.linalg.svd(new_a)
    ub, _, vtb = np.linalg.svd(new_b)
    u_opt1 = np.dot(ua, ub.T)
    u_opt2 = np.dot(vta.T, vtb)
    error = compute_error(new_a, new_b, u_opt1, u_opt2)
    return ProcrustesResult(error=error, new_a=new_a, new_b=new_b, t=u_opt2, s=u_opt1)
