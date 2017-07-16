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
from procrustes.utils import eigenvalue_decomposition, singular_value_decomposition
from procrustes.orthogonal import OrthogonalProcrustes
import numpy as np
from itertools import product


class TwoSidedOrthogonalSingleTransformationProcrustes(Procrustes):
    r"""
    Two-Sided Orthogonal Procrustes with Single-Transformation Class.

    Given matrix :math:`\mathbf{A}_{n \times n}` and a reference :math:`\mathbf{B}_{n \times n}`,
    find one unitary/orthogonal transformation matrix :math:`\mathbf{U}_{n \times n}` that makes
    :math:`\mathbf{A}_{n \times n}` as close as possible to :math:`\mathbf{B}_{n \times n}`. I.e.,

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

    Taking the eigenvalue decomposition of the matrices:

    .. math::
       \mathbf{A} = \mathbf{U}_A \mathbf{\Lambda}_A \mathbf{U}_A^\dagger \\
       \mathbf{B} = \mathbf{U}_B \mathbf{\Lambda}_B \mathbf{U}_B^\dagger

    the solution is obtained by,

    .. math::
       \mathbf{U} = \mathbf{U}_A \mathbf{S} \mathbf{U}_A^\dagger

    where :math:`\mathbf{S}` is a diagonal matrix for which every diagonal element is
    :math:`\pm{1}`,

    .. math::
       \mathbf{S} =
       \begin{bmatrix}
        { \pm 1} & 0       &\cdots &0 \\
        0        &{ \pm 1} &\ddots &\vdots \\
        \vdots   &\ddots   &\ddots &0\\
        0        &\cdots   &0      &{ \pm 1}
       \end{bmatrix}

    Finding the best choice of :math:`\mathbf{S}` requires :math:`2^n` trial-and-error tests.
    This is called the ``exact`` scheme for solving the probelm.

    A heuristic, due to Umeyama, is to take the element-wise absolute value of the elements
    of the unitary transformations,

    .. math::
       \mathbf{U}_\text{Umeyama} = \text{abs}(\mathbf{U}_A) \cdot \text{abs}(\mathbf{U}_B^\dagger)

    This is not actually a unitary matrix. But we can use the orthogonal procrustes problem
    to find the closest unitray matrix (i.e., the closest matrix that is unitarily equivalent
    to the identity matrix),

    .. math::
       \underbrace{\min}_{\left\{\mathbf{U} | \mathbf{U}^{-1} = {\mathbf{U}}^\dagger \right\}}
                          \|\mathbf{I}\mathbf{U} -  \mathbf{U}_\text{Umeyama}\|_{F}^2
       &= \underbrace{\text{min}}_{\left\{\mathbf{U} | \mathbf{U}^{-1} = {\mathbf{U}}^\dagger
                                   \right\}}
          \text{Tr}\left[\left(\mathbf{U} - \mathbf{U}_\text{Umeyama} \right)^\dagger
                         \left(\mathbf{U} - \mathbf{U}_\text{Umeyama} \right)\right] \\
       &= \underbrace{\text{max}}_{\left\{\mathbf{U} | \mathbf{U}^{-1} = {\mathbf{U}}^\dagger
                                   \right\}}
          \text{Tr}\left[\mathbf{U}^\dagger \mathbf{U}_\text{Umeyama} \right]

    considering the singular value decomposition of :math:`\mathbf{U}_\text{Umeyama}`,

    .. math::
       \mathbf{U}_\text{Umeyama} =
            \tilde{\mathbf{U}} \tilde{\mathbf{\Sigma}} \tilde{\mathbf{V}}^\dagger

    the solution is give by,

    .. math::
       \mathbf{U}_\text{Umeyama}^\text{approx} = \tilde{\mathbf{U}} \tilde{\mathbf{V}}^\dagger

    This is called the ``approx`` scheme for solving the probelm.
    """

    def __init__(self, array_a, array_b, translate=False, scale=False, scheme='approx', tol=1.e-12):
        r"""
        Initialize the class and transfer/scale the arrays followed by computing transformaions.

        Parameters
        ----------
        array_a : ndarray
            The 2d-array symmetric :math:`\mathbf{A}_{n \times n}` which is going to be transformed.
        array_b : ndarray
            The 2d-array symmetric :math:`\mathbf{B}_{n \times n}` representing the reference.
        translate : bool, default=False
            If True, both arrays are translated to be centered at origin.
        scale : bool, default=False
            If True, both arrays are column normalized to unity.
        scheme : str, default='approx'
            The scheme to solve for unitary transformation. Options: 'exact' and 'approx'.
        tol : float, default=1.e-12
            The tolerace used by ``approx`` scheme.
        """
        super(self.__class__, self).__init__(array_a, array_b, translate, scale)
        self._scheme = scheme
        self._tol = tol

        # check arrau_a and array_b are symmetric
        diff_a = abs(self.array_a - self.array_a.T)
        diff_b = abs(self.array_b - self.array_b.T)
        if np.all(diff_a) > 1.e-10:
            raise ValueError('Array array_a should be symmetric.')
        if np.all(diff_b) > 1.e-10:
            raise ValueError('Array array_b should be symmetric.')

        # compute matrix u and double sided error
        if self._scheme == 'approx':
            self._array_u = self._compute_transformation_approx()
            self._error = self.double_sided_error(self._array_u, self._array_u)

        elif self._scheme == 'exact':
            self._array_u, self._error = self._compute_transformation_exact()

        else:
            raise ValueError('Scheme={0} not recognozed!'.format(scheme))

    @property
    def array_u(self):
        r"""Transformation matrix :math:`\mathbf{U}`."""
        return self._array_u

    @property
    def error(self):
        """Procrustes error."""
        return self._error

    @property
    def scheme(self):
        """Scheme to solve for unitary transformation."""
        return self._scheme

    def _compute_transformation_approx(self):
        """
        Compute approximate two-sided orthogonal single transformation array.

        Returns
        -------
        u_opt : ndarray
           The optimum transformation array.
        """
        # calculate the eigenvalue decomposition of array_a and array_b
        sigma_a, u_a = eigenvalue_decomposition(self.array_a, two_sided_single=True)
        sigma_b, u_b = eigenvalue_decomposition(self.array_b, two_sided_single=True)

        # compute u_umeyama
        u_umeyama = np.multiply(abs(u_a), abs(u_b).T)
        # compute the closet unitary transformation to u_umeyama
        ortho = OrthogonalProcrustes(np.eye(u_umeyama.shape[0]), u_umeyama)
        ortho.array_u[abs(ortho.array_u) < 1.e-8] = 0
        return ortho.array_u

    def _compute_transformation_exact(self):
        """
        Compute exact two-sided orthogonal single transformation array.

        Returns
        -------
        u_opt : ndarray
           The optimum transformation array.
        e_opt : float
           The optimum double-sided procrustes error.
        """
        # svd of array_a and array_b
        u_a, sigma_a, v_trans_a = singular_value_decomposition(self.array_a)
        u_b, sigma_b, v_trans_b = singular_value_decomposition(self.array_b)

        # 2^n trial-and-error test to find optimum S array
        diags = product((-1., 1.), repeat=self.array_a.shape[0])
        for index, diag in enumerate(diags):
            if index == 0:
                u_opt = np.dot(np.dot(u_a, np.diag(diag)), u_b.T)
                e_opt = self.double_sided_error(u_opt, u_opt)
            else:
                # compute trial transformation and error
                u_trial = np.dot(np.dot(u_a, np.diag(diag)), u_b.T)
                e_trial = self.double_sided_error(u_trial, u_trial)
                if e_trial < e_opt:
                    u_opt = u_trial
                    e_opt = e_trial
            # stop trial-and-error if error is below treshold
            if e_opt < self._tol:
                break
        return u_opt, e_opt
