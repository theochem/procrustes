__author__ = 'Jonny'

from procrustes.base import Procrustes
from procrustes.utils import singular_value_decomposition
import numpy as np


class TwoSidedOrthogonalProcrustes(Procrustes):

    """
    This method deals with the orthogonal Procrustes problem

    """

    def __init__(self, array_a, array_b, translate=False, scale=False):

        Procrustes.__init__(self, array_a, array_b, translate=translate, scale=scale)

    def calculate(self):
        """
        Calculates the two optimum two-sided orthogonal transformation arrays in the
        double-sided procrustes problem

        Parameters
        ----------
        array_a : ndarray
            A 2D array representing the array to be transformed (as close as possible to array_b)

        array_b : ndarray
            A 2D array representing the reference array

        Returns
        ----------
        u1, u2, array_transformed, error
        u1 = the optimum orthogonal left-multiplying transformation array satisfying the double
             sided procrustes problem
        u2 = the optimum orthogonal right-multiplying transformation array satisfying the double
             sided procrustes problem
        array_transformed = the transformed input array after the transformation U1* array_a*U2
        error = the error as described by the double-sided procrustes problem
        """

        array_a = self.array_a
        array_b = self.array_b
        # Calculate the SVDs of array_a and array_b & solve for the optimum orthogonal transformation arrays
        u_a, sigma_a, v_trans_a = singular_value_decomposition(array_a)
        u_a0, sigma_a0, v_trans_a0 = singular_value_decomposition(array_b)
        u1 = np.dot(u_a, u_a0.T)
        u2 = np.dot(v_trans_a.T, v_trans_a0)

        # Calculate the error
        error = self.double_sided_procrustes_error(array_a, array_b, u1, u2)

        # Calculate the transformed input array
        array_transformed = np.dot(np.dot(u1.T, array_a), u2)

        return u1, u2, array_transformed, error, self.translate_and_or_scale
