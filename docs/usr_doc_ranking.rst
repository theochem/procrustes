..
    : Procrustes is a collection of interpretive chemical tools for
    : analyzing outputs of the quantum chemistry calculations.
    :
    : Copyright (C) 2017-2018 The Procrustes Development Team
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


Ranking by Reordering Method
============================

In most football matches, the double round-robin method is often used, where
each team will play against all other teams at home and then play against them
in other team place. In this part, we will use :class:`procrustes.permutation_2sided` to compute the
rank by reordering method.

Here is an example. The data is taken from :cite:`langville2012s` and the following table
shows the pair-wise relationship for 5 football teams.

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

We introduce the concept of ranking vector, which can be cast as a permutation of the integer 1 to n
that ranks all the teams. For example, the
:math:`{rank\_vec}^{\top} = [1,3,4,5,2]` for team_A, team_B, team_C, team_D, team_E respectively. the
:math:`rank\_vec` assigns team_A with rank position 1, team_E with rank position 2 and so on. The
ranking vector with length :math:`n` can result in a :math:`n \times n` *rank-differential matrix*
which is a symmetric reordering of the *fundamental rank-differential matrix*
:math:`\hat{R}_{n \times n}`.

.. math::
    \begin{bmatrix}
      0 & 1 & 2 & \cdots & n-1 \\
        & 0 & 1 & \cdots & n-2 \\
        &   &\ddots &\ddots & \vdots \\
        &   &   & \ddots & 1 \\
        &   &   &        & 0
    \end{bmatrix}

:math:`\hat{R}` is built from the *fundamental ranking vector* :math:`\hat{r}` where the :math:`n`
items appear in a ascending pattern.

.. math::
    \begin{bmatrix}
      1 \\
      2 \\
      3 \\
      \vdots \\
      n
    \end{bmatrix}

We can formulate for 5 football team gaming score into a matrix :math:`D`,

.. math::
    D =
    \begin{bmatrix}
        0    &   0    &   0   &    0    &    0 \\
       45    &   0    &  18   &    8    &   20 \\
        3    &   0    &   0   &    2    &    0 \\
       31    &   0    &   0   &    0    &    0 \\
       45    &   0    &   27  &   38    &    0 \\
    \end{bmatrix}&

Now the problem becomes finding a optimal permutation matrix :math:`Q` that minimizes
:math:`\left\lVert Q^{\top} D Q - \hat{R} \right\rVert` and more detailed information can be found
in reference.

In order to compute the *ranking vector*, we need the *fundamental rank-differential matrix*
:math:`\hat{R}_{n \times n}`. So we build a function

.. code-block:: python
   :linenos:

   # import required libraries
   import numpy as np
   from procrustes import permutation_2sided

   def _rank_differential(D):
    r""" Compute the rank differential based on the shape of input data."""

    N = np.shape(D)[0]
    R_hat = np.zeros((N, N))
    # Compute the upper triangle part of R_hat
    a = []
    for each in range(N):
        # print(each)
        a.extend(range(0, N-each))
    # Get the R_hat
    R_hat[np.triu_indices_from(R_hat, 0)] = a
    return R_hat

Now we can use the function to compute the *fundamental rank-differential matrix*
:math:`\hat{R}_{n \times n}` by using :math:`D` followed by two sided permutation Procrustes
computation.

.. code-block:: python
   :linenos:

   def ranking(D, perm_mode='normal1'):
       r""" Compute the ranking vector."""

       #_check_input(D)

       R_hat = _rank_differential(D)
       _, _, Q, e_opt = permutation_2sided(D, R_hat,
                                           remove_zero_col=False,
                                           remove_zero_row=False,
                                           mode=perm_mode)
       # Compute the rank
       _, rank = np.where(Q == 1)
       rank += 1

       return rank

Here the result *rank* should added by 1 because python's index starts from zero, which means the
rank we first computed was :math:`rank^{\top} = [4, 1, 3, 2, 0]`. Of note, sometimes, one needs to
check the input data :math:`D` is squared or not. Here we provide a simple function.

.. code-block:: python
   :linenos:

   def _check_input(D):
       r"""Check if the input is squared."""
       m, n = np.shape(D)
       if not m == n:
           raise ValueError("Input matrix should be squared one.")

All the codes have been wrapped in a single python executable file which locates in the
**Example/ranking** folder.
