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

`Procrustes <https://github.com/theochem/procrustes>`_ is a free and open source Python library for (generalized) Procrustes Problems. Procrustes problems arise when one wishes to find one or two transformations (which can be permutations, rotations, unitary, or symmetric) that make one matrix, :math:`\mathbf{A}` resemble a second "target" matrix :math:`\mathbf{B}` as closely as possible:

    .. math::
        \underbrace{\text{min}}_{\mathbf{T}_1 , \mathbf{T}_2 } \|\mathbf{T}_1 \mathbf{A} \mathbf{T}_2 - \mathbf{B}\|_{F}^2

Procrustes problems arise when aligning molecules and other objects, when evaluating optimal basis transformations, when determining optimal mappings between sets, and in many other contexts. This package includes options to translate, scale, and zero-pad matrices, so that matrices with different centers/scaling/sizes can be considered.

