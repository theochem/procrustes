# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2022 The QC-Devs Community
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
"""Testings for PSDP module."""

import numpy as np

# Test 1
# n = 4
# F = np.eye(n)
# G = np.diag(np.array([1, 2, 3, 4]))

# Test 2
n = 3
F = np.array([[1, 0, 0], [0, -2, 3], [0, 2, 4]])
G = np.array([[1, 6, 0], [4, 3, 0], [0, 0, -0.5]])

# Test 3
# n = 3
# F = np.array([[15, 1, 15 - 3, 2 + 5], [10, 5, 6, 3], [-3, 3, -3, -2 + 4]])
# G = np.array([[5, 1, 6, -1], [3, 2, 0, 2], [2, 4, 3, -3]])
# E = np.array(
#     [
#         [0.15858205, 0.05306816, 0.23642328],
#         [0.4749315, 0.6781462, 0.28026754],
#         [0.99497707, 0.00176484, 0.80068158],
#     ]
# )
