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

r"""Rotation Procrustes example: protein backbone alignment."""
import numpy as np
from protein_align import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# before align
A = _get_coordinates('2hhb.pdb', '2hhb', 'A')
C = _get_coordinates('2hhb.pdb', '2hhb', 'C')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(A[:,0], A[:,1], A[:,2], zdir='z', s=55, c='blue', label='chain_A')
ax.scatter(C[:,0], C[:,1], C[:,2], zdir='z', s=55, c='red', label='chain_C')

ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('Y', fontsize=20)
ax.set_zlabel('Z', fontsize=20)

rmsd=_compute_rmsd(A, C)

#ax.set_title(rmsd, fontsize=24)
ax.set_title('RMSD=39.468519767018776', fontsize=24)
ax.legend(fontsize=20)

plt.show()

# after align
new_A, new_C, rot_array, rmsd = align(
    file_name_A='2hhb.pdb', pdb_id_A='2hhb', chain_id_A='A',
    file_name_B='2hhb.pdb', pdb_id_B='2hhb', chain_id_B='C')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(new_A[:,0], new_A[:,1], new_A[:,2], zdir='z', s=55,
    c='blue', label='chain_A_new')
ax.scatter(new_C[:,0], new_C[:,1], new_C[:,2], zdir='z', s=55,
    c='red', label='chain_C_new')

ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('Y', fontsize=20)
ax.set_zlabel('Z', fontsize=20)

#ax.set_title(rmsd, fontsize=24)
ax.set_title('RMSD=1.2413324468611958e-08', fontsize=24)
ax.legend(fontsize=20)

plt.show()
