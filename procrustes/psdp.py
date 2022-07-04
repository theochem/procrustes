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
"""Positive semidefinite Procrustes Module."""

import numpy as np
import scipy

"""
Let n be the number of dimensions we are concerned with in this problem.
Initially, E = I_n and rank[G] = n. 
"""
from test.test_psdp import n, F, G


"""
Basic constructors of the algorithm.
"""


def T(arr):
    return np.transpose(arr)


def tr(arr):
    return np.trace(arr)


Q = F @ T(G) + G @ T(F)


def L(arr):
    return (T(arr) @ arr @ G @ T(G)) + (G @ T(G) @ T(arr) @ arr) - Q


def f(arr):
    return (1 / 2) * tr(
        T(F) @ F + T(arr) @ arr @ T(arr) @ arr @ G @ T(G) - T(arr) @ arr @ Q
    )


def permutation_matrix(E):
    # Find P such that, v(E') = Pv(E)
    k = 0
    x, y = E.shape
    P = np.zeros((x**2, x**2))

    for i in range(x**2):
        if i % x == 0:
            j = k
            k += 1
            P[i, j] = 1
        else:
            j += x
            P[i, j] = 1
    return P


def get_identity(E):
    arr1 = np.kron(E @ T(E), G @ T(G))
    arr2 = E @ G @ T(G) @ T(E)
    I = np.eye(arr1.shape[0] / arr2.shape[0])
    return I


def get_X(Z, s):
    X = Z[: n * (n - s), : n * (n - s)]
    return X


def D(E, LE):
    # D is constructed using two parts namely,
    # D1 and D2. D = [D1 D2]'

    # D2 is constructed using the following formula:
    s = np.linalg.matrix_rank(E)
    A = T(LE)
    ns = scipy.linalg.null_space(A)
    ns = ns * np.sign(ns[0, 0])
    v = ns[0]
    D2 = T(v) @ v

    # D1 is constructed using the following formula:
    P = permutation_matrix(E)
    I = get_identity(E)
    Z = (
        np.kron(E @ G @ T(G), T(E)) @ P
        + np.kron(E, G @ T(G) @ T(E)) @ P
        + np.kron(E @ G @ T(G) @ T(E), I)
        + np.kron(E @ T(E), G @ T(G))
    )
    X = get_X(Z, s)
    I = np.eye(n * (n - s) / LE.shape[0])
    flattened_D1 = np.linalg.inv(X + np.kron(I, LE)) @ np.kron(I, LE) @ E.flatten()
    D1 = flattened_D1.reshape(s, n)
    D = np.concatenate((D1, D2), axis=0)
    return D


def make_positive(LE):
    Lambda, U = np.linalg.eig(LE)
    Lambda_pos = [max(0, i) for i in Lambda]
    inv_U = np.linalg.inv(U)
    return np.dot(U, np.dot(np.diag(Lambda_pos), inv_U))


def find_minima(E, DE):
    def func(w):
        return f(E - w * DE)

    # Optimize the function using scipy.optimize.minimize
    # with respect to w > 0.
    w_min = scipy.optimize.minimize(func, 1, bounds=((0, None)))
    return w_min.x[0]


"""
Main algorithm:
1. E_i is chosen or E_0 is defined for the first iteration
2. Compute L(E_i)
3. If L(E_i) \geq 0 then we stop and E_i is the answer
4. Compute D_i
5. Minimize f(E_i - w_i D_i)
6. E_{i + 1} = E_i - w_i_min D_i
7. i = i + 1, start from 2 again
"""


def woodgate_algorithm():
    i = 0
    E = np.eye(N=n)
    while True:
        # Initialize once rather than calling multiple times.
        LE = L(E)
        if L(E) >= 0:
            print(f"Iteration: {i}, E = {E}")
            print(f"Required P = {T(E) @ E}")
            break
        LE_pos = make_positive(LE)
        DE = D(E, LE_pos)
        w = find_minima(E, DE)
        E = E - w * DE
        i += 1
    return T(E) @ E
