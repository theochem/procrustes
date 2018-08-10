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


from __future__ import absolute_import, division, print_function
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from procrustes import rotational


__all__ = [
    "align",
    "_get_coordinates",
    "_compute_rmsd"
]


def align(file_name_A, pdb_id_A, chain_id_A,
          file_name_B, pdb_id_B, chain_id_B):
    r"""
    """

    # Get inputs coordinate matrices
    A = _get_coordinates(file_name_A, pdb_id_A, chain_id_A)
    B = _get_coordinates(file_name_B, pdb_id_B, chain_id_B)
    # Kabsch algorithm/ Procrustes rotation to
    # align protein structure
    # new_A is just the translated coordinate
    new_A, new_B, array_rot, _, = rotational(A, B,
                                             remove_zero_col=False,
                                             remove_zero_row=False,
                                             translate=True)
    # now new_A is the array after rotation
    new_A = np.dot(new_A, array_rot)
    # Compute the rmsd values
    rmsd = _compute_rmsd(new_A, new_B)

    return new_A, new_B, array_rot, rmsd


def _get_coordinates(file_name, pdb_id, chain_id):
    r"""
    Build alpha carbon coordinates matrix from PDB file.

    Parameters
    ----------
    file_name : string
        PDB file name.
    pdb_id : string
        PDB ID.
    chain_id : string
        Chain ID. Possible inputs can be any of 'A', 'B', 'C', et al
        if it exists in the protein.
    """

    # permissive parser
    p = PDBParser(PERMISSIVE=1)
    structure = p.get_structure(pdb_id, file_name)
    # get X-ray crystal structure
    matrix = []
    chain = structure[0][chain_id]

    for residue in chain:
        for atom in residue:
            # Using residue['CA'] results in error
            if atom.get_id() == 'CA':
                matrix += list(atom.get_vector())
    matrix = np.asarray(matrix).reshape(-1, 3)

    return matrix


def _compute_rmsd(A, B):
    r"""
    Calculate root mean square deviation (rmsd).
    """

    # Check if A and B are with the same dimension
    if A.shape != B.shape:
        raise ValueError("INput matrices must be with the same shape\
                         for rmsd calculations.")
    D = len(A[0, :])
    N = len(A[:, 0])

    # Compute rmsd
    rmsd = 0.0
    for a, b in zip(A, B):
        rmsd += sum([(a[i] - b[i])**2.0 for i in range(D)])
    return np.sqrt(rmsd/N)

if __name__ == "__main__":
    align()

