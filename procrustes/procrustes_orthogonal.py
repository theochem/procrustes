"""
"""

from procrustes.base import Procrustes
from procrustes.utils import singular_value_decomposition
import numpy as np


class OrthogonalProcrustes(Procrustes):
    r"""
    This method deals with the orthogonal Procrustes problem.
    Given an :math:`\text{m}\times\text{n }A` and a reference matrix :math:`A^0`, find the unitary/orthogonla transformation of :math:`A` that makes it as close as possible to :math:`A^0`. I.e.,

    .. math::

       \underbrace{min}_{\left\{U|U^{-1} = {U}^\dagger \right\}}\|AU-A^0\|_{F}^2=\underbrace{min}_{\left\{U|U^{-1} = {U}^\dagger \right\}}\text{Tr}[({AU-A^0})^\dagger(AU-A^0)]=\underbrace{max}_{\left\{U|U^{-1} = {U}^\dagger\right\}}\text{Tr}[ U^\dagger {A}^\dagger A^0]

    The solution of obtained by taking the singular value decomposition (SVD) of the product of the matrices, :math:`A^{\dagger}A^0`,

    .. math::

       A^{\dagger}A^0 = \tilde{U}\tilde{\Sigma}\tilde{V^{\dagger}}\\
       U_{optimum}=\tilde{U}\tilde{V^{\dagger}}

    These singular values are always listed in decreasing order, with the smallest singular value in the bottom=right-hand corner of :math:`\tilde{\Sigma}`.
    """

    def __init__(self, array_a, array_b, translate=False, scale=False):

        Procrustes.__init__(self, array_a, array_b, translate=translate,
                            scale=scale)
        # allows one to call Procrustes methods

    def calculate(self):
        """
        Calculates the optimum orthogonal transformation array in the
        single-sided procrustes problem

        Parameters
        ----------
        array_a : ndarray
            A 2D array representing the array to be transformed (as close as possible to array_b)

        array_b : ndarray
            A 2D array representing the reference array

        Returns
        ----------
        u_optimum, array_transformed, error
        u_optimum = the optimum orthogonal transformation array satisfying the single
             sided procrustes problem
        array_ transformed = the transformed input array after transformation by u_optimum
        error = the error as described by the single-sided procrustes problem
        """
        array_a = self.array_a
        array_b = self.array_b

        # Calculate SVD
        prod_matrix = np.dot(self.array_a.T, self.array_b)
        u, s, v_trans = singular_value_decomposition(prod_matrix)

        # Define the optimum orthogonal transformation
        u_optimum = np.dot(u, v_trans)

        # Calculate the error
        error = self.single_sided_procrustes_error(self.array_a, self.array_b, u_optimum)

        # Calculate the transformed input array
        array_transformed = np.dot(array_a, u_optimum)

        return u_optimum, array_transformed, error, self.translate_and_or_scale
