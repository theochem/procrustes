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


+--------------------------------------------------------------------------------------------------------------------+------------------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Procrustes Type                                                                                                    | :math:`\mathbf{S}`           | :math:`\mathbf{T}`   | Constraints                                                                                                                                                                                                                                                     |
+--------------------------------------------------------------------------------------------------------------------+------------------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :ref:`Generic <generic>`  :cite:`GPA_Gower`                                                                        | :math:`\mathbf{I}`           | :math:`\mathbf{T}`   | None                                                                                                                                                                                                                                                            |
+--------------------------------------------------------------------------------------------------------------------+------------------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :ref:`Orthogonal <orthogonal>` :cite:`opp:1,opp:2,book1:gower2004procrustes`                                       | :math:`\mathbf{I}`           | :math:`\mathbf{Q}`   | :math:`{\mathbf{Q}^{-1} = {\mathbf{Q}}^\dagger}`                                                                                                                                                                                                                |
+--------------------------------------------------------------------------------------------------------------------+------------------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :ref:`Rotational <rotational>` :cite:`book1:gower2004procrustes,rot:brokken1983orthogonal,farrell1966least`        | :math:`\mathbf{I}`           | :math:`\mathbf{R}`   | :math:`\begin{cases} \mathbf{R}^{-1} = {\mathbf{R}}^\dagger \\ \left | \mathbf{R} \right | = 1 \\ \end{cases}`                                                                                                                                                  |
+--------------------------------------------------------------------------------------------------------------------+------------------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :ref:`Symmetric <symmetric>` :cite:`symmetric:1,escalante1998dykstra,peng2008m`                                    | :math:`\mathbf{I}`           | :math:`\mathbf{X}`   | :math:`\mathbf{X} = \mathbf{X}^\dagger`                                                                                                                                                                                                                         |
+--------------------------------------------------------------------------------------------------------------------+------------------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :ref:`Permutation <permutation>` :cite:`zadeh2013molecular`                                                        | :math:`\mathbf{I}`           | :math:`\mathbf{P}`   | :math:`\begin{cases} [\mathbf{P}]_{ij} \in \{0, 1\} \\ \sum_{i=1}^n [\mathbf{P}]_{ij} = \sum_{j=1}^n [\mathbf{P}]_{ij} = 1 \\ \end{cases}`                                                                                                                      |
+--------------------------------------------------------------------------------------------------------------------+------------------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :ref:`Two-sided Orthogonal <orthogonal>` :cite:`two_sided_ortho:1`                                                 | :math:`\mathbf{Q}_1^\dagger` | :math:`\mathbf{Q}_2` | :math:`\begin{cases} \mathbf{Q}_1^{-1} = \mathbf{Q}_1^\dagger \\ \mathbf{Q}_2^{-1} = \mathbf{Q}_2^\dagger \\ \end{cases}`                                                                                                                                       |
+--------------------------------------------------------------------------------------------------------------------+------------------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :ref:`Two-sided Orthogonal with One Transformation <orthogonal>` :cite:`two_sided_ortho_1trans:umeyama`            | :math:`\mathbf{Q}^{\dagger}` | :math:`\mathbf{Q}`   | :math:`\mathbf{Q}^{-1} = \mathbf{Q}^\dagger`                                                                                                                                                                                                                    |
+--------------------------------------------------------------------------------------------------------------------+------------------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :ref:`Two-sided Permutation <permutation>` :cite:`two_sided_permutation:1`                                         |:math:`\mathbf{P}_1^{\dagger}`| :math:`\mathbf{P}_2` | :math:`\begin{cases} [\mathbf{P}_1]_{ij} \in \{0, 1\} \\ [\mathbf{P}_2]_{ij} \in \{0, 1\} \\ \sum_{i=1}^n [\mathbf{P}_1]_{ij} = \sum_{j=1}^n [\mathbf{P}_1]_{ij} = 1 \\ \sum_{i=1}^n [\mathbf{P}_2]_{ij} = \sum_{j=1}^n [\mathbf{P}_2]_{ij} = 1 \\ \end{cases}` |
+--------------------------------------------------------------------------------------------------------------------+------------------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :ref:`Two-sided Permutation with One Transformation <permutation>` :cite:`opp:1,ding2008nonnegative`               | :math:`\mathbf{P}^{\dagger}` | :math:`\mathbf{P}`   | :math:`\begin{cases} [\mathbf{P}]_{ij} \in \{0, 1\} \\ \sum_{i=1}^n [\mathbf{P}]_{ij} = \sum_{j=1}^n [\mathbf{P}]_{ij} = 1 \\ \end{cases}`                                                                                                                      |
+--------------------------------------------------------------------------------------------------------------------+------------------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

In addition to these Procrustes methods, summarized in the table above, the
:ref:`generalized Procrustes analysis (GPA) <generalized>`
:cite:`stegmann2002brief,GPA_Gower,gower1975generalized,ten1977orthogonal,borg2005modern` and
softassign algorithm :cite:`kosowsky1994invisible,gold1996softassign,rangarajan1997convergence`
are also implemented in our package. The GPA algorithm seeks the optimal transformation matrices
:math:`\mathbf{T}` to superpose the given objects (usually more than 2) with minimum distance,

.. math::

    \begin{equation}
      \min \sum_{i<j}^{j} {\left\| \mathbf{A}_i \mathbf{T}_i - \mathbf{A}_j \mathbf{T}_j \right\|}^2
    \end{equation}

where :math:`\mathbf{A}_i` and :math:`\mathbf{A}_j` are the configurations and :math:`\mathbf{T}_i`
and :math:`\mathbf{T}_j` denotes the transformation matrices for :math:`\mathbf{A}_i` and
:math:`\mathbf{A}_j` respectively. When only two objects are given, the problem shrinks to generic
Procrustes.

The :ref:`softassign <softassign>` algorithm was first proposed to deal with quadratic
assignment problem
:cite:`kosowsky1994invisible` inspired by statistical physics algorithms and has subsequently been
developed theoretically
:cite:`gold1996softassign,rangarajan1997convergence` and extended to many other applications
:cite:`wang2018application,gold1996softassign,gold1996softmax,tian2012convergence,sheikhbahaee2017photometric`.
Because the two-sided permutation Procrustes problem is a special
quadratic assignment problem it can be used here. The objective function is to minimize
:math:`E_{qap} (\mathbf{M}, \mu, \nu)`, :cite:`gold1996softassign,yuille1994statistical`,
which is defined as follows,

.. math::

    \begin{equation}
        \begin{split}
            E_{qap}(\mathbf{M}, \mu, \nu) =
            & -\frac{1}{2}\sum_{aibj}\mathbf{C}_{ai;bj}\mathbf{M}_{ai}\mathbf{M}_{bj} \\
            & + \sum_{a} \mu_{a} \left( \sum_{i} \mathbf{M}_{ai} -1 \right ) + \sum_{i} \nu_{i}
            \left( \sum_{a} \mathbf{M}_{ai} -1 \right) \\
            & - \frac{\gamma}{2} \sum_{ai} \mathbf{M}^2_{ai} + \frac{1}{\beta} \sum_{ai} \mathbf{M}_{ai}
            \log{\mathbf{M}_{ai}}
        \end{split}
    \end{equation}


Procrustes problems arise when aligning molecules and other objects, when evaluating optimal basis
transformations, when determining optimal mappings between sets, and in many other contexts. This
package includes options to translate, scale, and zero-pad matrices, so that matrices with different
centers/scaling/sizes can be considered.
