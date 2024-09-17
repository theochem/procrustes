# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2024 The QC-Devs Community
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
"""Common function used for testing."""


import numpy as np
from scipy.optimize import minimize


def _vector_to_matrix(vec, nsize):
    r"""Given a vector, change it to a matrix."""
    mat = np.zeros((nsize, nsize))
    mat[np.triu_indices(nsize)] = vec
    mat = mat + mat.T - np.diag(np.diag(mat))
    return mat


def _objective_func(vec, array_a, array_b, nsize):
    """Frobenius norm of AX-B for symmetric matrix X."""
    mat = _vector_to_matrix(vec, nsize)
    diff = array_a.dot(mat) - array_b
    return np.trace(diff.T.dot(diff))


def minimize_one_transformation(array_a, array_b, ncol):
    """Find X matrix by minimizing Frobenius norm of AX-B."""
    guess = np.random.random(int(ncol * (ncol + 1) / 2.0))
    results = minimize(
        _objective_func,
        guess,
        args=(array_a, array_b, ncol),
        method="slsqp",
        options={"eps": 1e-8, "ftol": 1e-11, "maxiter": 1000},
    )
    return _vector_to_matrix(results["x"], ncol), results["fun"]
