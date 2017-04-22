__author__ = 'Jonny'

from procrustes.base import Procrustes
from procrustes.utils import hide_zero_padding, singular_value_decomposition
import numpy as np


class SymmetricProcrustes(Procrustes):
    """
    This method deals with the symmetric Procrustes problem

    We require symmetric input arrays for this problem

    """

    def __init__(self, array_a, array_b, translate=False, scale=False):

        self.array_a = hide_zero_padding(array_a)
        m_a, n_a = self.array_a.shape
        self.array_b = hide_zero_padding(array_b)
        m_b, n_b = self.array_b.shape
        assert(m_a == m_b and n_a == n_b)
        if m_a < n_a:
            raise ValueError('This analysis requires than m must be less than n,'
                             ' where m, n are the number of rows, columns of the unpadded input arrays')
        assert(np.linalg.matrix_rank(self.array_b) < n_a)

        Procrustes.__init__(self, self.array_a, self.array_b, translate=translate,
                            scale=scale)

    def calculate(self):
        """
        Calculates the optimum symmetric transformation array in the
        single-sided procrustes problem

        Parameters
        ----------
        array_a : ndarray
            A 2D array representing the array to be transformed (as close as possible to array_b)

        array_b : ndarray
            A 2D array representing the reference array

        Returns
        ----------
        x, array_transformed, error
        x = the optimum symmetric transformation array satisfying the single
             sided procrustes problem
        array_ transformed = the transformed input array after transformation by x
        error = the error as described by the single-sided procrustes problem
        """
        array_a = self.array_a
        array_b = self.array_b
        m, n = self.array_a.shape
        # Compute SVD of array_a
        u, s, v_trans = singular_value_decomposition(array_a)
        # Add zeros to the eigenvalue array s so that total length(s) = n
        s_concatenate = np.zeros(n - len(s))
        np.concatenate((s, s_concatenate))
        # Define array C
        c = np.dot(np.dot(u.T, array_b), v_trans.T)
        # Create the intermediate array Y and the optimum symmetric transformation array X
        y = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if s[i]**2 + s[j]**2 == 0:
                    y[i, j] = 0
                else:
                    y[i, j] = (s[i]*c[i, j] + s[j]*c[j, i]) / (s[i]**2 + s[j]**2)

        x = np.dot(np.dot(v_trans.T, y), v_trans)

        # Calculate the error
        error = self.single_sided_error(x)

        # Calculate the transformed input array
        array_transformed = np.dot(array_a, x)

        return x, array_transformed, error


# Reference
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.112.4378&rep=rep1&type=pdf
