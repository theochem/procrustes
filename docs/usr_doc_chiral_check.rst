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


Chirality Check
===============

In chemistry, a molecule is chiral if it cannot be superimposed onto its mirror image by any
combination of translation and rotation. These non-supposable mirror images are called
enantiomers which share identical chemical and physical properties, but have distinct chemical
reactivity and optical rotation properties.

.. figure:: notebooks/notebook_data/chirality_checking/chirality_checking.png
   :align: center
   :figwidth: 100%
   :figclass: align-center

   Enantiomers prediction of CHFClBr with rotational-orthogonal Procrustes by comparing atomic coordinates.

This example shows how easily the `Procrustes` library can be used to check whether two geometries
of the CHFClBr molecule are enantiomers using
`IOData <https://github.com/theochem/iodata>`_ library to obtain their
three-dimensional coordinates from XYZ files (**Fig. (i)**). This is done by testing whether their
coordinates can be matched through translation and rotation (i.e., rotational Procrustes);
the obtained Procrustes error of 26.09 Å reveals that these two structures are not identical.
However, it is confirmed that the two coordinates are enantiomers because they can be matched
through translation, rotation, and reflection (i.e., orthogonal Procrustes) gives a Procrustes
error of :math:`4.43 \times 10^{-8} Å`; thus, reflection is essential to match the structures.

.. code-block:: python
    :linenos:

    # load the libraries
    import numpy as np

    from iodata import load_one
    from procrustes import orthogonal, rotational

    # load CHClFBr enantiomers' coordinates from XYZ files
    a = load_one("notebook_data/chirality_checking/enantiomer1.xyz").atcoords
    b = load_one("notebook_data/chirality_checking/enantiomer2.xyz").atcoords

    # rotational Procrustes on a & b coordinates
    result_rot = rotational(a, b, translate=True, scale=False)
    print("Error =", result_rot.error)

    # rotational Procrustes on a & b coordinates
    result_rot = rotational(a, b, translate=True, scale=False)
    print("Error =", result_rot.error)

Now we define a function to plot the coordinates and then plot the coordinates only with rotation,

.. code-block:: python
    :linenos:

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    The error for only ration operation is 7.30 while it becomes 1.24e-08 after a reflection operation.
    def plot_atom_coordinates(coords1, coords2,Therefore, this example showed how we can use rotational procrustes to check chirality in organic
                              figsize=(12, 10),compounds.
                              fontsize_label=14,

    # rotated coordinates
    a_rot = np.dot(a, result_rot.t)

    # plot coordinates with only rotation
    plot_atom_coordinates(a_rot, b,
                          figsize=(10, 8),
                          fontsize_label=14,
                          fontsize_title=16,
                          fontsize_legend=16,
                          label1="enantiomer1 with rotation",
                          label2="enantiomer2",
                          title="Error={:0.2f}".format(result_rot.error),
                          figfile=None)

Then we check the case with both rotation and reflection,

.. code-block:: python
    :linenos:

    # orthogonal Procrustes on a & b coordinates
    result_ortho = orthogonal(a, b, translate=True, scale=False)
    print("Error =", result_ortho.error)

    # rotated and refelction coordinates
    a_ortho = np.dot(a, result_ortho.t)

    # plot coordinates with only rotation
    plot_atom_coordinates(a_ortho, b,
                          figsize=(10, 8),
                          fontsize_label=14,
                          fontsize_title=16,
                          fontsize_legend=16,
                          label1="enantiomer1 with orthoation and reflection",
                          label2="enantiomer2",
                          title="Error={:0.2f}".format(result_ortho.error),
                          figfile=None)
