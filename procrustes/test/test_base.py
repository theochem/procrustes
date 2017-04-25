# -*- coding: utf-8 -*-
# Procrustes is a collection of interpretive chemical tools for
# analyzing outputs of the quantum chemistry calculations.
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
"""
"""

import numpy as np
from procrustes import Procrustes


def test_procrustes_base_array():
    # check procrustes arrays
    array1 = np.array([[1, 2], [3, 4]])
    array2 = np.array([[5, 6]])
    procrust = Procrustes(array1, array2)
    assert procrust.array_a.shape == (2, 2)
    assert procrust.array_b.shape == (2, 2)
    assert (abs(procrust.array_a - array1) < 1.e-10).all()
    assert (abs(procrust.array_b - np.array([[5, 6], [0, 0]])) < 1.e-10).all()

    # check procrustes arrays
    array1 = np.array([[4, 7, 2], [1, 3, 5]])
    array2 = np.array([[5], [2]])
    procrust = Procrustes(array1, array2)
    assert procrust.array_a.shape == (2, 3)
    assert procrust.array_b.shape == (2, 1)
    assert (abs(procrust.array_a - array1) < 1.e-10).all()
    assert (abs(procrust.array_b - array2) < 1.e-10).all()
