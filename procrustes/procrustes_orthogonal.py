__author__ = 'Jonny'

from procrustes.base import Procrustes
import numpy as np


class OrthogonalProcrustes(Procrustes):

    '''
    This method deals with the orthogonal Procrustes problem
    '''

    def __init__(self, array_a, array_b, translate_scale=False, translate=False, scale=False):

        Procrustes.__init__(self, array_a, array_b, translate_scale=translate_scale, translate=translate,
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
        prod_matrix = np.dot(array_a.T, array_b)
        u, s, v_trans = self.singular_value_decomposition(prod_matrix)

        # Remove array elements close to zero
        u[abs(u) <= 1.e-8] = 0
        v_trans[abs(v_trans) <= 1.e-8] = 0

        # Define the optimum orthogonal transformation
        u_optimum = np.dot(u, v_trans)
        u_optimum[abs(u_optimum) <= 1.e-8] = 0

        # Calculate the error
        error = self.single_sided_procrustes_error(array_a, array_b, u_optimum)

        # Calculate the transformed input array
        array_transformed = np.dot(array_a, u_optimum)

        return u_optimum, array_transformed, error
