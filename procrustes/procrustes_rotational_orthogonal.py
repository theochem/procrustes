"""
"""

from procrustes.base import Procrustes
from procrustes.utils import singular_value_decomposition
import numpy as np


class RotationalOrthogonalProcrustes(Procrustes):
    r"""
    Rotational Orthogonal Procrustes Class.

    Given a matrix :math:`A_{m \times n}` and a reference matrix :math:`A^0_{m \times n}`,
    find the unitary/orthogonla transformation of :math:`A_{m \times n}` that makes it as
    close as possible to :math:`A^0_{m times n}`. I.e.,

    .. math::
       \underbrace{\text{min}}_{\left\{\mathbf{U} \left| {\mathbf{U}^{-1} = {\mathbf{U}}^\dagger
                                \atop \left| \mathbf{U} \right| = 1} \right. \right\}}
          \|\mathbf{A}\mathbf{U} - \mathbf{A}^0\|_{F}^2
       &= \underbrace{\text{min}}_{\left\{\mathbf{U} \left| {\mathbf{U}^{-1} = {\mathbf{U}}^\dagger
                                   \atop \left| \mathbf{U} \right| = 1} \right. \right\}}
          \text{Tr}\left[\left(\mathbf{A}\mathbf{U} - \mathbf{A}^0 \right)^\dagger
                         \left(\mathbf{A}\mathbf{U} - \mathbf{A}^0 \right)\right] \\
       &= \underbrace{\text{max}}_{\left\{\mathbf{U} \left| {\mathbf{U}^{-1} = {\mathbf{U}}^\dagger
                                   \atop \left| \mathbf{U} \right| = 1} \right. \right\}}
          \text{Tr}\left[\mathbf{U}^\dagger {\mathbf{A}}^\dagger \mathbf{A}^0 \right]

    The solution of obtained by taking the singular value decomposition (SVD) of the product of
    the matrices,

    .. math::
       \mathbf{A}^\dagger \mathbf{A}^0 &= \tilde{\mathbf{U}} \tilde{\mathbf{\Sigma}}
                                          \tilde{\mathbf{V}}^{\dagger} \\
       \mathbf{U}_{\text{optimum}} &= \tilde{\mathbf{U}} \tilde{\mathbf{S}}
                                      \tilde{\mathbf{V}}^{\dagger}

    Where :math:`\tilde{\mathbf{S}}_{n \times m}` is almost an identity matrix,

    .. math::
       \tilde{\mathbf{S}}_{m \times n} \equiv
       \begin{bmatrix}
           1  &  0  &  \cdots  &  0   &  0 \\
           0  &  1  &  \ddots  & \vdots &0 \\
           0  & \ddots &\ddots & 0 &\vdots \\
           \vdots&0 & 0        & 1     &0 \\
           0 & 0 & 0 \cdots &0 &\operatorname{sgn}
                                \left(\left|\mathbf{U}\mathbf{V}^\dagger\right|\right)
       \end{bmatrix}

    I.e. the smallest singular value is replaced by

    .. math::
       \operatorname{sgn} \left(\left|\tilde{\mathbf{U}} \tilde{\mathbf{V}}^\dagger\right|\right) =
       \begin{cases}
        +1 \qquad \left|\tilde{\mathbf{U}} \tilde{\mathbf{V}}^\dagger\right| \geq 0 \\
        -1 \qquad \left|\tilde{\mathbf{U}} \tilde{\mathbf{V}}^\dagger\right| < 0
       \end{cases}
    """

    def __init__(self, array_a, array_b, translate=False, scale=False):
        r"""
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
        super(RotationalOrthogonalProcrustes, self).__init__(array_a, array_b, translate, scale)

        # compute transformation
        self.array_u = self.compute_transformation()

        # calculate the single-sided error
        self.error = self.single_sided_error(self.array_u)

    def compute_transformation(self):
        """
        Return the optimal rotational-orthogonal transformation array :math:`\mathbf{U}`.

        Returns
        -------
        ndarray
            The optimum rotational-orthogonal transformation array.
        """
        # compute SVD of A.T * A
        product = np.dot(self.array_a.T, self.array_b)
        u, s, v_trans = singular_value_decomposition(product)

        # construct matrix S which is an identity matrix where the smallest
        # singular value is replaced with sgn(|U*V^t|).
        array_s = np.eye(self.array_a.shape[1])
        array_s[-1, -1] = np.sign(np.linalg.det(np.dot(u, v_trans)))

        # compute optimum rotation matrix
        u_opt = np.dot(np.dot(u, array_s), v_trans)

        return u_opt
