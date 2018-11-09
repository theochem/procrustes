# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2018 The Procrustes Development Team
#
# This file is part of Procrustes.
#
# Procrustes is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Procrustes is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --

r"""Rotational Procrustes example: chirality check."""

import numpy as np
from procrustes import *


def chiral_check(A_data, B_data):
    r"""Check if a organic compound is chiral.

    Parameters
    ----------
    A_data : string
        The data file that contains 3D coordinates of the first organic compound A.
    B_data : string
        The data file that contains 3D coordinates of the second organic compound B.
    Returns
    -------
    A : ndarray
        3D coordinates of the first organic compound A.
    B : ndarray
        3D coordinates of the first organic compound B.
    """

    # get the data
    A = np.loadtxt(A_data)
    B = np.loadtxt(B_data)

    reflection = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # create the reflection of compound A over the yz plane
    A_ref = np.dot(A, reflection)
    # Compute the rotational procrustes
    _, _, U_rot, e_rot = rotational(A, B,
                                    translate=True,
                                    scale=False,
                                    remove_zero_col=False,
                                    remove_zero_row=False)
    # Compute the error: reflection + rotation
    _, _, U__ref_rot, e_ref_rot = rotational(A_ref, B,
                                             translate=True,
                                             scale=False,
                                             remove_zero_col=False,
                                             remove_zero_row=False)

    if e_rot/e_ref_rot > 10:
    	print("These two compounds are enantiomers \
    		and there is at least one chiral center in each of them.")
    else:
    	print("These two compounds are not enantiomers \
    		and there is no chiral center in any of them.")

if __name__ == "__main__":
	chiral_check(A_data, B_data)



#In [5]: s = np.loadtxt("S.dat")
#   ...: r = np.loadtxt("R.dat")
#   ...:
#   ...:
#
#In [6]: s
#Out[6]:
#array([[ 0.3979, -0.5423, -0.4377],
#       [ 0.0726,  0.2672,  0.2157],
#       [ 1.1494,  0.1898,  1.8873],
#       [-1.2253,  0.096 ,  0.4617],
#       [ 0.3097,  1.8081, -0.6451]])
#
#In [7]: r
#Out[7]:
#array([[-0.6703,  1.1981,  1.0828],
#       [-0.1215,  0.5187,  0.4306],
#       [ 1.3718, -0.2786,  1.4765],
#       [-0.9723, -0.4134,  0.0043],
#       [ 0.4794,  1.4486, -0.9643]])
#
#In [8]: reflection = np.array([[-1,0,0], [0,1,0], [0,0,1]])
#   ...: s_ref = np.dot(s, reflection)
#   ...:
#   ...:
#
#In [9]: s_ref
#Out[9]:
#array([[-0.3979, -0.5423, -0.4377],
#       [-0.0726,  0.2672,  0.2157],
#       [-1.1494,  0.1898,  1.8873],
#       [ 1.2253,  0.096 ,  0.4617],
#       [-0.3097,  1.8081, -0.6451]])
#
#In [10]: _, _, U_rot, e_rot = rotational(s, r, translate=True, scale=False)
#
#In [11]: _, _, U__ref_rot, e_ref_rot = rotational(s_ref, r, translate=True, scale=False)
#
#In [12]: U_rot
#Out[12]:
#array([[ 0.89579755, -0.36416615, -0.25481318],
#       [ 0.34518644,  0.93117688, -0.11728566],
#       [ 0.27998762,  0.01710615,  0.95985119]])
#
#In [13]: e_rot
#Out[13]: 7.304696784953152
#
#In [14]: U__ref_rot
#Out[14]:
#array([[-0.67391318, -0.55353026, -0.48933146],
#       [ 0.55727312,  0.05400299, -0.82857127],
#       [ 0.48506464, -0.83107636,  0.27207421]])
#
#In [15]: e_ref_rot
#Out[15]: 1.241332446867831e-08#