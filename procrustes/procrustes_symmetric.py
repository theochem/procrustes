"""
"""


from procrustes.base import Procrustes
from procrustes.utils import hide_zero_padding, singular_value_decomposition
import numpy as np


class SymmetricProcrustes(Procrustes):
    """
    Symmetric Procrustes Class.
    """

    def __init__(self, array_a, array_b, translate=False, scale=False):
        """
        Initialize the class and transfer/scale the arrays followed by computing transformaion.

        Parameters
        ----------
        array_a : ndarray
            The 2d-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
        array_b : ndarray
            The 2d-array :math:`\mathbf{A}^0_{m \times n}` representing the reference.
        translate : bool
            If True, both arrays are translated to be centered at origin, default=False.
        scale : bool
            If True, both arrays are column normalized to unity, default=False.

        Notes
        -----
        The Procrustes analysis requires two 2d-arrays with the same number of rows, so the
        array with the smaller number of rows will automatically be padded with zero rows.
        """
        array_a = hide_zero_padding(array_a)
        array_b = hide_zero_padding(array_b)

        if array_a.shape[0] < array_a.shape[1]:
            raise ValueError('The unpadding array_a should cannot have more columns than rows.')
        if array_a.shape[0] != array_b.shape[0]:
            raise ValueError('Arguments array_a & array_b should have the same number of rows.')
        if array_a.shape[1] != array_b.shape[1]:
            raise ValueError('Arguments array_a & array_b should have the same number of columns.')
        if np.linalg.matrix_rank(array_b) >= array_a.shape[1]:
            raise ValueError('Rand of array_b should be less than number of columns of array_a.')

        super(SymmetricProcrustes, self).__init__(array_a, array_b, translate, scale)

        # compute transformation
        self.array_x = self.compute_transformation()

        # calculate the single-sided error
        self.error = self.single_sided_error(self.array_x)

    def compute_transformation(self):
        """
        Return optimum right hand sided symmetric transformation array.

        Returns
        -------
        ndarray
            The symmetric transformation array.
        """
        # compute SVD of A
        u, s, v_trans = singular_value_decomposition(self.array_a)

        # add zeros to the eigenvalue array so it has length n
        n = self.array_a.shape[1]
        if len(s) < self.array_a.shape[1]:
            s = np.concatenate((s, np.zeros(n - len(s))))

        c = np.dot(np.dot(u.T, self.array_b), v_trans.T)
        # Create the intermediate array Y and the optimum symmetric transformation array X
        y = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if s[i]**2 + s[j]**2 == 0:
                    y[i, j] = 0
                else:
                    y[i, j] = (s[i]*c[i, j] + s[j]*c[j, i]) / (s[i]**2 + s[j]**2)

        x = np.dot(np.dot(v_trans.T, y), v_trans)

        return x


# Reference
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.112.4378&rep=rep1&type=pdf
