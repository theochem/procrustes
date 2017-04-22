__author__ = 'Jonny'

from procrustes.base import Procrustes
from procrustes.utils import singular_value_decomposition
import numpy as np


class RotationalOrthogonalProcrustes(Procrustes):



    def __init__(self, array_a, array_b, translate=False, scale=False):

        r"""
        This method deals with the Rotational-Orthogonal Procrustes Problem Constrain the transformation matrix :math:`U` to be pure rotational.

        .. math::

            \underbrace {\min }_{\left\{ {U\left| {\scriptstyle{U^{ - 1}} = U\atop \scriptstyle\left| U \right| = 1} \right.} \right\}}\left\| {AU - {A^0}} \right\|_F^2 = \underbrace {\min }_{\left\{ {U\left| {\scriptstyle{U^{ - 1}} = U\atop \scriptstyle\left| U \right| = 1} \right.} \right\}}Tr\left[ {{{(AU - {A^0})}^\dagger }(AU - {A^0})} \right] = \underbrace {\min }_{\left\{ {U\left| {\scriptstyle{U^{ - 1}} = U\atop \scriptstyle\left| U \right| = 1} \right.} \right\}}Tr\left[ {{U^\dagger }{A^\dagger }{A^0}} \right]

        The solution of obtained by taking the singular value decomposition (SVD) of the product of the matrices, :math:`A^{\dagger}A^0`,

        .. math::
           A^{\dagger}A^0 = \tilde{U}\tilde{\Sigma}\tilde{V^{\dagger}}\\
           U_{optimum}=\tilde{U}\tilde{V^{\dagger}}

        where :math:`\tilde{S}` is the n :math:`\times` n matrix that is almost an identity matrix,

        .. math::
           \tilde{S} \equiv
           \begin{bmatrix}
               1  &  0  &  \cdots  &  0   &  0\\
               0  &  1  &  \ddots  & \vdots &0\\
               0  & \ddots &\ddots & 0 &\vdots\\
               \vdots&0 & 0        & 1     &0\\
               0 & 0 \cdots &0 & {\operatorname{sgn}} \left( {\left| {U{V^{\dagger} }} \right|} \right)
           \end{bmatrix}
        I.e. the smallest singular value is replaced by

        .. math::
           {\operatorname{sgn}} \left( {\left|{U{V^{\dagger} }} \right|} \right)= \left\{ {\begin{array}{*{40}{c}}{ + 1}\\{ - 1}\end{array}} \right.\begin{array}{*{40}{c}}{\left| {U{V^\dagger }} \right| \geq 0}\\{\left| {U{V^\dagger }} \right|< 0}\end{array}



        """
        Procrustes.__init__(self, array_a, array_b,
                            translate=translate, scale=scale)

    def calculate(self):
        """
        Calculates the optimum rotational-orthogonal transformation array in the
        single-sided procrustes problem

        Parameters
        ----------
        array_a : ndarray
            A 2D array representing the array to be transformed (as close as possible to array_b)

        array_b : ndarray
            A 2D array representing the reference array

        Returns
        ----------
        r, array_transformed, error
        r = the optimum rotation transformation array satisfying the single
             sided procrustes problem
        array_ transformed = the transformed input array after transformation by r
        error = the error as described by the single-sided procrustes problem
        """
        # form the product matrix & compute SVD
        prod_matrix = np.dot(self.array_a.T, self.array_b)
        u, s, v_trans = singular_value_decomposition(prod_matrix)

        # Constrain transformation matrix to be a pure rotation matrix by replacing the least
        # significant singular value with sgn(|U*V^t|). The rest of the
        # diagonal entries are ones
        replace_singular_value = np.sign(np.linalg.det(np.dot(u, v_trans)))
        n = self.array_a.shape[1]
        s = np.eye(n)
        s[n - 1, n - 1] = replace_singular_value

        # Calculate the optimum rotation matrix r
        r = np.dot(np.dot(u, s), v_trans)

        # Calculate the error
        error = self.single_sided_error(r)

        # Calculate the transformed input array
        array_transformed = np.dot(self.array_a, r)

        return r, array_transformed, error
