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


Atom-Atom Mapping
=================

Given two molecular structures, it is important to be able to identify atoms that are chemically
similar. This a commonly used in 3D QSAR pharmacore analysis, substructure searching, metabolic
pathway identification, and chemical machine learning.

The code block below shows how easily the Procrustes library can be used to map atoms of
*but-1-en-3-yne* (A) and *3,3-dimethylpent-1-en-4-yne* (B) as depicted in **Fig. (i)**.
Based on our chemical intuition, we can tell that the triple and double bonds of both molecules
"match" one another; however, simple (geometric) molecular alignment based on three-dimensional
coordinates does not identify that. The pivotal step is defining a representation that contains
bonding information, and then using permutation Procrustes to match atoms between the two chemical
structures.

Inspired by graph theory, we represented each molecule with an "adjacency" matrix where the
diagonal elements are the atomic numbers and the off-diagonal elements are the bond orders
(matrix :math:`\mathbf{A} \in \mathbb{R}^{4 \times 4}` and
:math:`\mathbf{B} \in \mathbb{R}^{7 \times 7}`
in **Fig. (ii)** ). The two-sided permutation Procrustes (:class:`procrustes.permutation_2sided`)
with one-transformation can be used to find the optimal matching of the two matrices.

.. figure:: notebooks/notebook_data/atom_atom_mapping/atom_atom_mapping.png
   :align: center
   :figwidth: 100%
   :figclass: align-center

   Atom-atom Mapping with Two-sided Permutation Procrustes

It is important to note that the permutation Procrustes requires the two matrices to be of the
same size, so the smaller matrix :math:`\mathbf{A}` is padded with zero rows and columns to have
same shape as matrix :math:`\mathbf{B}`. After obtaining the optimal permutation matrix
:math:`\mathbf{P}`, the transformed matrix :math:`\mathbf{P^{\top}AP}` should be compared to
matrix :math:`\mathbf{B}` for identifying the matching atoms; the zero rows/columns correspond to
atoms in :math:`\mathbf{B}` for which there are no corresponding atoms in :math:`\mathbf{A}`. The
mapping between atoms can be also directly deduced from matrix :math:`\mathbf{P}`,

.. math::
    \min_{\mathbf{P}} {\left\lVert \mathbf{P}^{\top} \mathbf{A} \mathbf{P} - \mathbf{B}
        \right\rVert}_{F}^2

.. code-block:: python
    :linenos:

    import numpy as np

    from procrustes import permutation_2sided

    # Define molecule A representing "but-1-en-3-yne"
    A = np.array([[6, 3, 0, 0],
                  [3, 6, 1, 0],
                  [0, 1, 6, 2],
                  [0, 0, 2, 6]])

    # Define molecule B representing "3,3‐dimethylpent‐1‐en‐4‐yne"
    B = np.array([[6, 3, 0, 0, 0, 0, 0],
                  [3, 6, 1, 0, 0, 0, 0],
                  [0, 1, 6, 1, 0, 1, 1],
                  [0, 0, 1, 6, 2, 0, 0],
                  [0, 0, 0, 2, 6, 0, 0],
                  [0, 0, 1, 0, 0, 6, 0],
                  [0, 0, 1, 0, 0, 0, 6]])

    # two-sided permutation Procrustes
    result = permutation_2sided(A, B, single=True, pad=True)

    # Compute the transformed molecule A
    P = result.t
    new_A = np.dot(P.T, np.dot(result.new_a, P)).astype(int)
    print("Transformed A: \n", new_A)    # compare to B

The computed result is shown in the **Fig. (iii)**, generating ideal matching of the double and
triple carbon-carbon bonds. The new matrix representation of :math:`\mathbf{A}` suggests that atom
3 is empty since the third row and third column of :math:`\mathbf{A}` are zero (matrix elements
in blue). That is, a virtual atom 3 was added to molecule :math:`\mathbf{A}` to align with atom 3
in molecule :math:`\mathbf{B}`. Similarly, atoms 6 and 7 in molecule :math:`\mathbf{B}` (matrix
elements in blue) do not have meaningful matches in :math:`\mathbf{A}`, and are mapped to two
virtual atoms, atom 6 and 7 in molecule :math:`\mathbf{A}`. This example is inspired by
:cite:`zadeh2013molecular`.

.. image:: https://img.shields.io/badge/binder-atom%20atom%20mapping-579ACA.svg?logo=data:image/png;base64,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
    :align: center
    :target: https://mybinder.org/v2/gh/theochem/procrustes/master?filepath=docs%2Fnotebooks%2F/Atom_Atom_Mapping.ipynb
