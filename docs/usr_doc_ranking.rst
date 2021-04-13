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


Ranking by Reordering
=====================

The problem of ranking a set of objects is ubiquitous not only in everyday life, but also for
many scientific problems such as information retrieval, recommender systems, natural language
processing, and drug discovery. In this tutorial, we will rank footable teams based on the game
scores.

.. table:: Team by Team Game Score Data
   :align: center

   ======= ======= ======= ======= ======= =======
   Team     Duke    Miami    UNC     UVA     VT
   ======= ======= ======= ======= ======= =======
   Duke       0       0       0       0        0
   Miami     45       0      18       8       20
   UNC        3       0       0       2        0
   UVA       31       0       0       0        0
   VT        45       0       27     38        0
   ======= ======= ======= ======= ======= =======

The code block below shows how easily the *Procrustes* library can be used to rank five
American collegiate football teams, where each team plays one game against every other team,
using their score-differentials as summarized in Table 1 (The data taken from A. N. Langville, C.
D. Meyer, *Ranking by Reordering Methods*, Princeton University Press, 2012, Ch. 8, pp. 97–112.
:cite:`langville2012s`).

Here, each team is given a zero score for a game they lost (e.g., Duke lost to every other team)
and the score difference is calculated for games won (e.g., Miami beat Duke by 45 points and UNC
by 18 points). These results are also summarized in the square score-differential matrix
:math:`\mathbf{A}` in **Fig. (i)**. Two-sided permutation Procrustes can be used to rank these
teams, but one needs to define a proper target matrix. Traditionally, the rank-differential matrix
has been used for this purpose and is defined for :math:`n` teams as,

.. math::
   \begin{equation}
       \mathbf{R}_{n \times n} =
       \begin{bmatrix}
           0 & 1 & 2 & \cdots & n-1 \\
             & 0 & 1 & \cdots & n-2 \\
             &   &\ddots &\ddots & \vdots \\
             &   &   & \ddots & 1 \\
             &   &   &        & 0
       \end{bmatrix}
   \end{equation}

The rank-differential matrix :math:`\mathbf{R} \in \mathbb{R}^{n \times n}` is an upper-triangular
matrix and its :math:`ij`-th element specifies the difference in ranking between team :math:`i` and
team :math:`j`. This a sensible target for the score-differential matrix. Now,
the two-sided permutation Procrustes method can be used to find the permutation matrix that
maximizes the similarity between the score-differential matrix, :math:`\mathbf{A}`, and the
rank-differential matrix based on the definition of rank-differential matrix,
:math:`\mathbf{B}` (**Fig. (ii)**)

.. math::
   \begin{equation}
      \min_{\mathbf{P}} {\left\lVert \mathbf{P}^{\top} \mathbf{A} \mathbf{P} - \mathbf{B}
         \right\rVert}_{F}^2
   \end{equation}

This results to :math:`[5,2,4,3,1]` as the final rankings of the teams (**Fig. (iii)**).

.. figure:: notebooks/notebook_data/ranking_reordering/ranking.png
   :align: center
   :figwidth: 100%
   :figclass: align-center

   Ranking by reordering with two-sided permutation with one-transformation

In order to compute the *ranking vector*, we need the *fundamental rank-differential matrix*
:math:`\hat{R}_{n \times n}`. So we build a function

.. code-block:: python
   :linenos:

   import numpy as np

   from procrustes import permutation_2sided

   # input score-differential matrix
   A = np.array([[ 0, 0, 0 ,  0,  0 ],    # Duke
                 [45, 0, 18,  8,  20],    # Miami
                 [ 3, 0, 0 ,  2,  0 ],    # UNC
                 [31, 0, 0 ,  0,  0 ],    # UVA
                 [45, 0, 27, 38,  0 ]])   # VT

   # make rank-differential matrix
   n = A.shape[0]
   B = np.zeros((n, n))
   for index in range(n):
       B[index, index:] = range(0, n - index)

   # rank teams using two-sided Procrustes
   result = permutation_2sided(A, B, single=True,
                               mode='normal1', tol=10.e-6)

   # compute teams' ranks
   _, ranks = np.where(result.t == 1)
   ranks += 1
   print("Ranks = ", ranks)     # displays [5, 2, 4, 3, 1]

Why we need to add all the rank values by 1? Because Python's list index starts with 0, but we
often index starting from 1 for physical objects.

.. image:: https://img.shields.io/badge/binder-ranking%20by%20reordering-579ACA.svg?logo=data:image/png;base64,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
   :align: center
   :target: https://mybinder.org/v2/gh/theochem/procrustes/master?filepath=docs%2Fnotebooks%2F/Ranking_by_Reordering.ipynb
