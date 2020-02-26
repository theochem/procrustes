# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2020 The Procrustes Development Team
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


r"""Rotation Procrustes example: protein backbone alignment."""


import numpy as np
from procrustes import *
from check_chirality import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# no reflection
s = np.loadtxt("S.dat")
r = np.loadtxt("R.dat")
# only rotation
_, _, U_rot, e_rot = rotational(s, r, translate=True, scale=False)
s_rot = np.dot(s, U_rot)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(s_rot[:,0], s_rot[:,1], s_rot[:,2], zdir='z', s=55, c='blue', label='S with rotation')
ax.scatter(r[:,0], r[:,1], r[:,2], zdir='z', s=55, c='red', label='R')

ax.set_xlabel('X', fontsize=16)
ax.set_ylabel('Y', fontsize=16)
ax.set_zlabel('Z', fontsize=16)

#ax.set_title(rmsd, fontsize=24)
ax.set_title('Error=7.304696784953152', fontsize=24)
ax.legend(fontsize=20)
plt.show()


# with reflection
reflection = np.array([[-1,0,0], [0,1,0], [0,0,1]])
s_ref = np.dot(s, reflection)
_, _, U_ref_rot, e_ref_rot = rotational(s_ref, r, translate=True, scale=False)
s_ref_rot = np.dot(s_ref, U_ref_rot)
# plot the coordinates
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(s_ref_rot[:,0], s_ref_rot[:,1], s_ref_rot[:,2], zdir='z', s=55,
    c='blue', label='S with rotation and reflection')
ax.scatter(r[:,0], r[:,1], r[:,2], zdir='z', s=55,
    c='red', label='R')

ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('Y', fontsize=20)
ax.set_zlabel('Z', fontsize=20)

#ax.set_title(rmsd, fontsize=24)
ax.set_title('RMSD=0.23003871056618516', fontsize=24)
ax.legend(fontsize=20)

plt.show()






