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


Quick Start
###########

The code block below gives an example of the orthogonal Procrustes problem for random matrices
:math:`\mathbf{A} \in \mathbb{R}^{m \times n}` and :math:`\mathbf{B} \in \mathbb{R}^{m \times n}`.
Here, matrix :math:`\mathbf{B} \in \mathbb{R}^{m \times n}` is constructed by shifting an orthogonal
transformation of matrix :math:`\mathbf{A} \in \mathbb{R}^{m \times n}`, so the matrices can be
perfectly matched and the error is zero. As is the case with all Procrustes flavours, the user
can specify whether the matrices should be translated (so that both are centered at origin)
and/or scaled (so that both are normalized to unity with respect to the Frobenius norm).
In addition, the other optional arguments (not appearing in the code-block below) specify whether
the zero columns (on the right-hand side) and rows (at the bottom) should be removed prior to
transformation.

As a python library, Procrustes can be imported and used in python codes. People may get a better
understanding of how to use the package by the implemented examples.

Procrustes can be used as an computation library, either ending up with a script or a python
package. Here is a snippet for script.

.. code-block:: python
   :linenos:

   import numpy as np
   from scipy.stats import ortho_group
   from procrustes import orthogonal

   # random input 10x7 matrix A
   a = np.random.rand(10, 7)

   # random orthogonal 7x7 matrix T
   m = ortho_group.rvs(7)

   # target matrix B (which is a shifted AxT)
   b = np.dot(a, m) + np.random.rand(1, 7)

   # orthogonal Procrustes analysis with translation
   result = orthogonal(a, b, scale=True, translate=True)

   # display Procrustes results
   print(result.error)    # error (expected to be zero)
   print(result.t)    # transformation matrix (same as T)
   print(result.new_b)
   print(result.new_a)


Getting Help
============

You may refer to the API documentation for technical details. You can also try this

.. code-block:: python

   # function name followed by a question mark
   help(permutation)

You will get a detailed information about the parameters and returns.
