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
r"""Testings for PSDP (positive semi-definite Procrustes) module."""

import numpy as np
from numpy.testing import assert_almost_equal
from procrustes.psdp import psdp_woodgate, psdp_peng


def test_psdp_identity(n=np.random.randint(50, 100)):
    r"""Test PSDP with identity matrix."""
    a = np.eye(n)
    b = np.eye(n)
    res = psdp_woodgate(a=a, b=b)
    s, error = res["s"], res["error"]
    assert_almost_equal(s, np.eye(n))
    assert_almost_equal(error, 0.0)
    res = psdp_peng(a=a, b=b)
    s, error = res["s"], res["error"]
    assert_almost_equal(s, np.eye(n))
    assert_almost_equal(error, 0.0)


def test_psdp_diagonal():
    r"""Test PSDP with diagonal matrix."""
    a = np.diag([1, 2, 3, 4])
    b = np.eye(4)
    res = psdp_woodgate(a=a, b=b)
    s, error = res["s"], res["error"]
    actual_result = np.diag([0.99999, 0.5, 0.33333, 0.25])
    assert_almost_equal(s, actual_result, decimal=5)
    assert_almost_equal(error, 0.0, decimal=3)
    res = psdp_peng(a=a, b=b)
    s, error = res["s"], res["error"]
    assert_almost_equal(s, actual_result, decimal=5)
    assert_almost_equal(error, 0.0, decimal=3)


def test_psdp_generic_square():
    r"""Test PSDP with 2 generic square matrices."""
    a = np.array([[1, 6, 0], [4, 3, 0], [0, 0, -0.5]])
    b = np.array([[1, 0, 0], [0, -2, 3], [0, 2, 4]])
    res = psdp_woodgate(a=a, b=b)
    s, error = res["s"], res["error"]
    actual_result = np.array(
        [
            [0.22351489, -0.11059539, 0.24342428],
            [-0.11059539, 0.05472271, -0.12044658],
            [0.24342428, -0.12044658, 0.26510708],
        ]
    )
    assert_almost_equal(s, actual_result)
    assert_almost_equal(error, 31.371190566800497, decimal=3)
    res = psdp_peng(a=a, b=b)
    s, error = res["s"], res["error"]
    actual_result = np.array(
        [
            [0.1440107, -0.05853613, 0.00939016],
            [-0.05853613, 0.02379322, -0.00381683],
            [0.00939016, -0.00381683, 0.00061228],
        ]
    )
    assert_almost_equal(s, actual_result)
    assert_almost_equal(error, 33.43617791613022, decimal=3)


def test_psdp_generic_non_square():
    r"""Test PSDP with 2 generic non-square matrices."""
    a = np.array([[5, 1, 6, -1], [3, 2, 0, 2], [2, 4, 3, -3]])
    b = np.array([[15, 1, 15 - 3, 2 + 5], [10, 5, 6, 3], [-3, 3, -3, -2 + 4]])
    res = psdp_woodgate(a=a, b=b)
    s, error = res["s"], res["error"]
    actual_result = np.array(
        [
            [2.57997197, 1.11007896, -1.08770156],
            [1.11007896, 1.68429863, 0.12829214],
            [-1.08770156, 0.12829214, 0.75328052],
        ]
    )
    assert_almost_equal(s, actual_result, decimal=5)
    assert_almost_equal(error, 32.200295757989856, decimal=3)
    res = psdp_peng(a=a, b=b)
    s, error = res["s"], res["error"]
    actual_result = np.array(
        [
            [2.58773004, 1.10512076, -1.07911235],
            [1.10512076, 1.65881535, 0.14189328],
            [-1.07911235, 0.14189328, 0.75610083],
        ]
    )
    assert_almost_equal(s, actual_result, decimal=5)
    assert_almost_equal(error, 32.21671819685297)


def test_psdp_non_full_rank():
    r"""Test PSDP when the to be transformed matrix doesn't have full rank."""
    a = np.array(
        [
            [0.3452, -0.9897, 0.8082, -0.1739, -1.4692, -0.2531, 1.0339],
            [0.2472, -1.4457, -0.6672, -0.5790, 1.2516, -0.8184, -0.4790],
            [-1.3567, -0.9348, 0.7573, 1.7002, -0.9627, -0.5655, 2.5222],
            [1.6639, 0.6111, -0.1858, 0.0485, 0.1136, 0.1372, -0.0149],
            [-0.1400, -0.3303, -0.2965, 0.0218, 0.0565, -0.1907, -0.2544],
            [-1.2662, 0.1905, 0.3302, -0.4041, 1.1479, -1.4716, 0.0857],
        ]
    )
    b = np.array(
        [
            [6.3043, -6.5364, 1.2659, -2.7625, -2.9861, -5.4362, 2.7422],
            [-0.5694, -9.4371, -5.5455, -15.6041, 24.4958, -20.4567, -11.4576],
            [-0.1030, 2.3164, 3.0813, 8.1280, -10.6447, 6.6903, 7.9874],
            [8.1678, -4.5977, 0.0559, -2.6948, 1.1326, -6.5904, 2.0167],
            [6.3043, -6.5364, 1.2659, -2.7625, -2.9861, -5.4362, 2.7422],
            [-0.5694, -9.4371, -5.5455, -15.6041, 24.4958, -20.4567, -11.4576],
        ]
    )
    res = psdp_woodgate(a=a, b=b)
    s, error = res["s"], res["error"]
    actual_result = np.array(
        [
            [5.41919155, 1.62689999, -0.31097211, 3.87095011, 5.4023016, 1.6426171],
            [1.62689999, 9.65181183, -3.52330026, 2.48010767, 1.64006384, 9.61898593],
            [-0.31097211, -3.52330026, 2.71903863, 0.02576251, -0.30656676, -3.5354399],
            [3.87095011, 2.48010767, 0.02576251, 5.97846041, 3.85902148, 2.48086299],
            [5.4023016, 1.64006384, -0.30656676, 3.85902148, 5.40096355, 1.62207837],
            [1.6426171, 9.61898593, -3.5354399, 2.48086299, 1.62207837, 9.65931602],
        ]
    )
    assert_almost_equal(s, actual_result, decimal=2)
    assert_almost_equal(error, 0.0025848018603061226)
    res = psdp_peng(a=a, b=b)
    s, error = res["s"], res["error"]
    actual_result = np.array(
        [
            [5.40904359, 1.63342796, -0.30678904, 3.87229001, 5.40837406, 1.63361756],
            [1.63342796, 9.63675505, -3.53016762, 2.47911674, 1.63339076, 9.63661932],
            [
                -0.30678904,
                -3.53016762,
                2.71130391,
                0.02463814,
                -0.30688059,
                -3.53026462,
            ],
            [3.87229001, 2.47911674, 0.02463814, 5.96975888, 3.87182355, 2.47929281],
            [5.40837406, 1.63339076, -0.30688059, 3.87182355, 5.40770783, 1.6335514],
            [1.63361756, 9.63661932, -3.53026462, 2.47929281, 1.6335514, 9.63674522],
        ]
    )
    assert_almost_equal(s, actual_result, decimal=2)
    assert_almost_equal(error, 2.9011440718759514e-06)
