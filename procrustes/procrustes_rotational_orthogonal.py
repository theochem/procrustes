__author__ = 'Jonny'

from procrustes.base import Procrustes
import numpy as np


class RotationalOrthogonalProcrustes(Procrustes):

    '''
    This method deals with the Rotational-Orthogonal Procrustes Problem
    Constrain the transformation matrix U to be pure rotational
    '''

    def __init__(self, array_a, array_b, translate_scale=False, translate=False, scale=False):

        Procrustes.__init__(self, array_a, array_b, translate_scale=translate_scale, translate=translate, scale=scale)

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

        array_a = self.array_a
        array_b = self.array_b
        # form the product matrix & compute SVD
        prod_matrix = np.dot(array_a.T, array_b)
        u, s, v_trans = self.singular_value_decomposition(prod_matrix)

        # Constrain transformation matrix to be a pure rotation matrix by replacing the least
        # significant singular value with sgn(|U*V^t|). The rest of the diagonal entries are ones
        replace_singular_value = np.sign(np.linalg.det(np.dot(u, v_trans)))
        n = array_a.shape[1]
        s = np.eye(n)
        s[n-1, n-1] = replace_singular_value
        # Remove zero entries
        s[abs(s) <= 1e-8] = 0

        # Calculate the optimum rotation matrix r
        r = np.dot(np.dot(u, s), v_trans)
        r[abs(r) <= 1e-8] = 0

        # Calculate the error
        error = self.single_sided_procrustes_error(array_a, array_b, r)

        # Calculate the transformed input array
        array_transformed = np.dot(array_a, r)

        return r, array_transformed, error
