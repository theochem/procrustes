__author__ = 'Jonny'

import hungarian.hungarian_algorithm as hm
from procrustes.base import Procrustes
import numpy as np


class PermutationProcrustes(Procrustes):
    '''
    This method deals with the Permutation Procrustes problem
    '''
    def __init__(self, array_a, array_b, translate=False, scale=False):

        Procrustes.__init__(self, array_a, array_b, translate=translate,
                            scale=scale)

    def calculate(self):
        """
        Calculates the optimum permutation transformation array in the
        single-sided procrustes problem.

        Parameters
        ----------
        array_a : ndarray
            A 2D array representing the array to be transformed (as close as possible to array_b)

        array_b : ndarray
            A 2D array representing the reference array

        Returns
        ----------
        perm_optimum, array_transformed, total_potential error
        perm_optimum = the optimum permutation transformation matrix satisfying the single
             sided procrustes problem
        array_ transformed = the transformed input array after transformation by perm_optimum
        total_potential = The total 'profit', i.e. the trace of the transformed input array
        error = the error as described by the single-sided procrustes problem.
        """
        array_a = self.array_a
        array_b = self.array_b

        # Define the profit array & applying the hungarian algorithm
        profit_array = np.dot(array_a.T, array_b)
        hungary = hm.Hungarian(profit_array, is_profit_matrix=True)
        hungary.calculate()

        # Obtain the optimum permutation transformation and convert to array form
        perm_hungarian = hungary.get_results()
        perm_optimum = np.zeros(profit_array.shape)
        # convert hungarian output into array form
        for k in range(len(perm_hungarian)):
            i, j = perm_hungarian[k]
            perm_optimum[i, j] = 1

        # Calculate the total potential (trace)
        total_potential = hungary.get_total_potential()

        # Calculate the error
        error = self.single_sided_error(perm_optimum)

        # Calculate the transformed input array
        array_transformed = np.dot(array_a, perm_optimum)

        return perm_optimum, array_transformed, total_potential, error, self.translate_and_or_scale