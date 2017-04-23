"""
"""


from procrustes.base import Procrustes
from procrustes.utils import singular_value_decomposition
import numpy as np


class OrthogonalProcrustes(Procrustes):
    r"""
    Orthogonal Procrustes Class.

    Given a matrix :math:`A_{\m \times \n}` and a reference matrix :math:`A^0_{\m \times \n}`,
    find the unitary/orthogonla transformation of :math:`A_{\m \times \n}` that makes it as
    close as possible to :math:`A^0_{\m \times \n}`. I.e.,

    .. math::
       \underbrace{\text{min}}_{\left\{\mathbf{U} | \mathbf{U}^{-1} = {\mathbf{U}}^\dagger
                                \right\}}
          \|\mathbf{A}\mathbf{U} - \mathbf{A}^0\|_{F}^2
       &= \underbrace{\text{min}}_{\left\{\mathbf{U} | \mathbf{U}^{-1} = {\mathbf{U}}^\dagger
                                   \right\}}
          \text{Tr}\left[\left(\mathbf{A}\mathbf{U} - \mathbf{A}^0 \right)^\dagger
                         \left(\mathbf{A}\mathbf{U} - \mathbf{A}^0 \right)\right] \\
       &= \underbrace{\text{max}}_{\left\{\mathbf{U} | \mathbf{U}^{-1} = {\mathbf{U}}^\dagger
                                   \right\}}
          \text{Tr}\left[\mathbf{U}^\dagger {\mathbf{A}}^\dagger \mathbf{A}^0 \right]

    The solution is obtained by taking the singular value decomposition (SVD) of the
    product of the matrices,

    .. math::
       \mathbf{A}^\dagger \mathbf{A}^0 &= \tilde{\mathbf{U}} \tilde{\mathbf{\Sigma}}
                                          \tilde{\mathbf{V}}^{\dagger} \\
       \mathbf{U}_{\text{optimum}} &= \tilde{\mathbf{U}} \tilde{\mathbf{V}}^{\dagger}

    These singular values are always listed in decreasing order, with the smallest singular
    value in the bottom-right-hand corner of :math:`\tilde{\mathbf{\Sigma}}`.
    """

    def __init__(self, array_a, array_b, translate=False, scale=False):
        """
        """
        super(OrthogonalProcrustes, self).__init__(array_a, array_b, translate, scale)

        # compute transformation
        self.array_u = self.compute_transformation()

        # calculate the single-sided error
        self.error = self.single_sided_error(self.array_u)

    def compute_transformation(self):
        r"""
        Return the optimal orthogonal transformation array :math:`\mathbf{U}`

        Returns
        -------
        ndarray
            The optimum orthogonal transformation array.
        """
        # calculate SVD of A.T * A0
        product = np.dot(self.array_a.T, self.array_b)
        u, s, v_trans = singular_value_decomposition(product)
        # compute optimum orthogonal transformation
        u_optimum = np.dot(u, v_trans)
        return u_optimum
