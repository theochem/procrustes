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

r"""Rotational Procrustes example: chirality check."""

import numpy as np
from iodata import load_one

from procrustes import rotational
from rdkit import Chem


def chiral_check(A_coords, B_coords):
    r"""Check if a organic compound is chiral.

    Parameters
    ----------
    A_coords : string
        Atomic coordinates of the first organic compound A.
    B_coords : string
        Atomic coordinates of the first organic compound B.
    Returns
    -------
    A : ndarray
        3D coordinates of the first organic compound A.
    B : ndarray
        3D coordinates of the first organic compound B.
    """

    reflection = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # create the reflection of compound A over the yz plane
    A_ref = np.dot(A_coords, reflection)
    # Compute the rotational procrustes
    res = rotational(A_coords, B_coords, translate=True, scale=False, unpad_col=False,
                     unpad_row=False)
    # Compute the error: reflection + rotation
    res_ref = rotational(A_ref, B_coords, translate=True, scale=False, unpad_col=False,
                         unpad_row=False)

    if res["e_opt"] / res_ref["e_opt"] > 10:
        print("These two compounds are enantiomers "
              "and there is at least one chiral center in each of them.")
    else:
        print("These two compounds are not enantiomers "
              "and there is no chiral center in any of them.")


def atom_coordinates_rdkit(sdf_name):
    r"""Load atomic coordinates from a sdf file with RDKit.

    Parameters
    ----------
    sdf_name : string
        SDF file name.

    Returns
    -------
    coords : ndarray
        3D atomic coordinates.
    """
    mol = Chem.SDMolSupplier(sdf_name)[0]
    conf = mol.GetConformer()
    return conf.GetPositions()


def extract_coordinates(sdf_name):
    r"""Extract atomic coordinates from a sdf file.

    Parameters
    ----------
    sdf_name : string
        SDF file name.

    Returns
    -------
    coords : ndarray
        3D atomic coordinates.
    """
    coordinates = []
    with open(sdf_name, "r") as mol_fname:
        for line in mol_fname:
            line = line.strip()
            if line.endswith("V2000") or line.endswith("V3000"):
                break
        for line in mol_fname:
            line_seg = line.strip().split()
            if len(line_seg) == 10:
                coordinates.append(line_seg[:3])
            else:
                break
    coordinates = np.array(coordinates).astype(np.float)
    return coordinates


def atom_coordinates_iodata(sdf_name):
    r"""Load atomic coordinates from a sdf file with iodata.

    Parameters
    ----------
    sdf_name : string
        SDF file name.

    Returns
    -------
    coords : ndarray
        3D atomic coordinates.
    """
    mol = load_one(sdf_name)
    coords = mol.atcoords
    return coords


if __name__ == "__main__":
    # use rdkit to load atomic coordinates
    # A_data = atom_coordinates_rdkit("R.sdf")
    # B_data = atom_coordinates_rdkit("S.sdf")

    # use iodata to load atomic coordinates
    A_data = atom_coordinates_iodata("R.sdf")
    B_data = atom_coordinates_iodata("S.sdf")

    # use numpy to load atomic coordinates
    # A_data = extract_coordinates("R.sdf")
    # B_data = extract_coordinates("S.sdf")

    chiral_check(A_data, B_data)
