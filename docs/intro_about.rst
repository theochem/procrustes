..
    : The Procrustes library provides a set of functions for transforming
    : a matrix to make it as similar as possible to a target matrix.
    :
    : Copyright (C) 2017-2021 The QC-Devs Community
    :
    : This file is part of Procrustes.
    :
    : Procrustes is free software; you can redistribute it and/or
    : modify it under the terms of the GNU General Public License
    : as published by the Free Software Foundation; either version 3
    : of the License, or (at your option) any later version.
    :
    : Procrustes is distributed in the hope that it will be useful,
    : but WITHOUT ANY WARRANTY; without even the implied warranty of
    : MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    : GNU General Public License for more details.
    :
    : You should have received a copy of the GNU General Public License
    : along with this program; if not, see <http://www.gnu.org/licenses/>
    :
    : --


About Procrustes
================

`Procrustes <https://github.com/theochem/procrustes>`_ is a free and open source Python library for
(generalized) Procrustes Problems. Procrustes problems arise when one wishes to find one or two
transformations (which can be permutations, rotations, unitary, or symmetric) that make one matrix,
:math:`\mathbf{A}` resemble a second "target" matrix :math:`\mathbf{B}` as closely as possible:

    .. math::
        \underbrace{\min}_{\mathbf{S}, \mathbf{T}} \| \mathbf{S}\mathbf{A}\mathbf{T} -
        \mathbf{B}\|_{F}^2

where :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` is the input matrix,
:math:`\mathbf{B} \in \mathbb{R}^{m \times n}` is the reference (target) matrix, and
:math:`\| \cdot \|_{F}` denotes the Frobenius norm defined as,

.. math::

    \begin{split}
        \\\| \mathbf{A} \|_{F}
        & = \sqrt{\sum^m_{i=1} \sum^n_{j=1} |a_{ij}|^2} \\
        & = \sqrt{ \text{Tr} (\mathbf{A}^{\dagger} \mathbf{A})} \\
        & = \sqrt{ \sum^{\min \{m, n \}}_{i=1} \rho^2_i (\mathbf{A})}
    \end{split}

Here :math:`a_{ij}`, :math:`\text{Tr}(\mathbf{A})`, and :math:`\rho^2_i (\mathbf{A})` denotes the
element :math:`ij`, trace, and singular values of the matrix of :math:`\mathbf{A}`, respectively.

Different Procrustes problems use different choices for the transformation matrices
:math:`\mathbf{S}` and :math:`\mathbf{T}` which are commonly taken to be orthogonal/unitary
matrices, rotation matrices, symmetric matrices, or permutation matrices. When :math:`\mathbf{S}`
is an identity matrix, it is called a one-sided Procrustes problem :cite:`GPA_Gower`, and when it is
equal to :math:`\mathbf{T}`, it becomes two-sided Procrustes problem with one transformation.

Procrustes problems arise when aligning molecules and other objects, when evaluating optimal basis
transformations, when determining optimal mappings between sets, and in many other contexts. This
package includes options to translate, scale, and zero-pad matrices, so that matrices with different
centers/scaling/sizes can be considered.
