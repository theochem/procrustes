"""
"""


import hungarian.hungarian_algorithm as hm
from procrustes.base import Procrustes
import numpy as np


class PermutationProcrustes(Procrustes):
    """
    Permutation Procrustes Class.
    """

    def __init__(self, array_a, array_b, translate=False, scale=False):
        """
        Parameters
        ----------
        array_a : ndarray
            The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
        array_b : ndarray
            The 2d-array :math:`\mathbf{A}^0_{m \times n}` representing the reference.
        translate : bool
            Whether to translate input arrays to origin; default=False.
        scale : bool
            Whether to scale the input arrays; default=False.
        """
        super(PermutationProcrustes, self).__init__(array_a, array_b, translate, scale)

        # compute transformation
        self.array_p = self.compute_transformation()

        # calculate the single-sided error
        self.error = self.single_sided_error(self.array_p)

    def compute_transformation(self):
        """
        Return optimum right hand sided permutation array.

        Returns
        -------
        ndarray
            The permutation array.
        """
        # Define the profit array & applying the hungarian algorithm
        profit_array = np.dot(self.array_a.T, self.array_b)
        hungary = hm.Hungarian(profit_array, is_profit_matrix=True)
        hungary.calculate()

        # Obtain the optimum permutation transformation and convert to array form
        perm_hungarian = hungary.get_results()
        perm_optimum = np.zeros(profit_array.shape)
        # convert hungarian output into array form
        for k in range(len(perm_hungarian)):
            i, j = perm_hungarian[k]
            perm_optimum[i, j] = 1

        # # Calculate the total potential (trace)
        # total_potential = hungary.get_total_potential()

        return perm_optimum
