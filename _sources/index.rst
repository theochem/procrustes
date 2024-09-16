..
    : The Procrustes library provides a set of functions for transforming
    : a matrix to make it as similar as possible to a target matrix.
    :
    : Copyright (C) 2017-2022 The QC-Devs Community
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

.. Procrustes documentation master file, created by
   sphinx-quickstart on Wed Apr 11 19:35:53 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Procrustes's Documentation!
======================================

`Procrustes <https://github.com/theochem/procrustes>`_ is a free, open-source, and cross-platform
Python library for (generalized) Procrustes problems with the goal of finding the optimal
transformation(s) that makes two matrices as close as possible to each other.
Please use the following citation in any publication using Procrustes library:

    **"Procrustes: A Python Library to Find Transformations that Maximize the Similarity Between
    Matrices"**, F. Meng, M. Richer, A. Tehrani, J. La, T. D. Kim, P. W. Ayers, F. Heidar-Zadeh,
    `Computer Physics Communications, 276(108334), 2022. <https://doi.org/10.1016/j.cpc.2022.108334>`__.

The Procrustes source code is hosted on `GitHub <https://github.com/theochem/procrustes>`_ and is
released under the
`GNU General Public License v3.0 <https://github.com/theochem/procrustes/blob/master/LICENSE>`_.
We welcome any contributions to the Procrustes library in accordance with our Code of Conduct;
please see our `Contributing Guidelines <https://qcdevs.org/guidelines/QCDevsCodeOfConduct/>`_.
Please report any issues you encounter while using Procrustes library on
`GitHub Issues <https://github.com/theochem/procrustes/issues>`_.
For further information and inquiries please contact us at `qcdevs@gmail.com <qcdevs@gmail.com>`_.


Description of Procrustes Methods
=================================

Procrustes problems arise when one wishes to find one or two transformations,
:math:`\mathbf{T} \in \mathbb{R}^{n \times n}` and :math:`\mathbf{S} \in \mathbb{R}^{m \times m}`,
that make matrix
:math:`\mathbf{A} \in \mathbb{R}^{m \times n}` (input matrix) resemble matrix
:math:`\mathbf{B} \in \mathbb{R}^{m \times n}` (target or reference matrix) as closely as possible:

.. math::
   \underbrace{\min}_{\mathbf{S}, \mathbf{T}} \|\mathbf{S}\mathbf{A}\mathbf{T} - \mathbf{B}\|_{F}^2

where, the :math:`\| \cdot \|_{F}` denotes the Frobenius norm defined as,

.. math::
   \| \mathbf{A} \|_{F} = \sqrt{\sum^m_{i=1} \sum^n_{j=1} |a_{ij}|^2}
                        = \sqrt{ \text{Tr} (\mathbf{A}^{\dagger} \mathbf{A})}

Here :math:`a_{ij}` and :math:`\text{Tr}(\mathbf{A})` denote the :math:`ij`-th element and trace
of matrix :math:`\mathbf{A}`, respectively. When :math:`\mathbf{S}`
is an identity matrix, this is called a **one-sided Procrustes problem**, and when it is
equal to :math:`\mathbf{T}`, this becomes **two-sided Procrustes problem with one transformation**,
otherwise, it is called **two-sided Procrustes problem**. Different Procrustes problems use
different choices for the transformation matrices :math:`\mathbf{S}` and :math:`\mathbf{T}` which
are commonly taken to be orthogonal/unitary matrices, rotation matrices, symmetric matrices, or
permutation matrices. The table below summarizes various Procrustes methods supported:

.. include:: table_procrustes.inc

In addition to these Procrustes methods, summarized in the table above, the
:ref:`generalized Procrustes analysis (GPA) <generalized>`
:cite:`stegmann2002brief,GPA_Gower,gower1975generalized,ten1977orthogonal,borg2005modern` and
softassign algorithm :cite:`kosowsky1994invisible,gold1996softassign,rangarajan1997convergence`
are also implemented in our package. The GPA algorithm seeks the optimal transformation matrices
:math:`\mathbf{T}` to superpose the given objects (usually more than 2) with minimum distance,

.. math::
    \begin{equation}
      \min \sum_{i<j}^{j} {\left\| \mathbf{A}_i \mathbf{T}_i - \mathbf{A}_j \mathbf{T}_j \right\|_F}^2
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
Because the two-sided permutation Procrustes problem is a special case of the
quadratic assignment problem, it can be used here. The objective function is to minimize
:math:`E_{qap} (\mathbf{M}, \mu, \nu)`, :cite:`gold1996softassign,yuille1994statistical`,
which is defined as follows,

.. math::

    \begin{aligned}
        E_{q a p}(\mathbf{M}, \mu, \nu) = &-\frac{1}{2} \sum_{a i b j} \mathbf{C}_{a i ; b j}
        \mathbf{M}_{a i} \mathbf{M}_{b j} \\
        & + \sum_{a} \mu_{a}\left(\sum_{i} \mathbf{M}_{a i} - 1 \right) + \sum_{i} \nu_{i} \left(
        \sum_{a} \mathbf{M}_{a i} - 1 \right) \\
        & - \frac{\gamma}{2} \sum_{a i} \mathbf{M}_{a i}^{2} + \frac{1}{\beta} \sum_{a i}
        \mathbf{M}_{a i} \log \mathbf{M}_{a i}
    \end{aligned}


Procrustes problems arise when aligning molecules and other objects, when evaluating optimal basis
transformations, when determining optimal mappings between sets, and in many other contexts. This
package includes the options to translate, scale, and zero-pad matrices, so that matrices with
different centers/scaling/sizes can be considered.


.. toctree::
   :maxdepth: 1
   :caption: User Documentation

   usr_doc_installization
   notebooks/Quick_Start
   usr_doc_tutorials
   usr_doc_zref

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API Documentation

   api/api_index.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

