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
"""Utils module for Procrustes."""

import numpy as np
from numpy.testing import assert_almost_equal, assert_raises

from procrustes.utils import _hide_zero_padding, _scale_array, _translate_array, _zero_padding


def test_zero_padding_rows():
    r"""Test _zero_padding with random array padding rows."""
    array1 = np.array([[1, 2], [3, 4]])
    array2 = np.array([[5, 6]])

    # match the number of rows of the 1st array
    padded2, padded1 = _zero_padding(array2, array1, pad_mode="row")
    assert padded1.shape == (2, 2)
    assert padded2.shape == (2, 2)
    assert (abs(padded1 - array1) < 1.0e-10).all()
    assert (abs(padded2 - np.array([[5, 6], [0, 0]])) < 1.0e-10).all()

    # match the number of rows of the 1st array
    array3 = np.arange(8).reshape(2, 4)
    array4 = np.arange(8).reshape(4, 2)
    padded3, padded4 = _zero_padding(array3, array4, pad_mode="row")
    assert padded3.shape == (4, 4)
    assert padded4.shape == (4, 2)
    assert (abs(array4 - padded4) < 1.0e-10).all()
    expected = list(range(8))
    expected.extend([0] * 8)
    expected = np.array(expected).reshape(4, 4)
    assert (abs(expected - padded3) < 1.0e-10).all()

    # padding the padded_arrays should not change anything
    padded5, padded6 = _zero_padding(padded3, padded4, pad_mode="row")
    assert padded3.shape == (4, 4)
    assert padded4.shape == (4, 2)
    assert padded5.shape == (4, 4)
    assert padded6.shape == (4, 2)
    assert (abs(padded5 - padded3) < 1.0e-10).all()
    assert (abs(padded6 - padded4) < 1.0e-10).all()


def test_zero_padding_columns():
    r"""Test _zero_padding with random array padding columns."""
    array1 = np.array([[4, 7, 2], [1, 3, 5]])
    array2 = np.array([[5], [2]])

    # match the number of columns of the 1st array
    padded2, padded1 = _zero_padding(array2, array1, pad_mode="col")
    assert padded1.shape == (2, 3)
    assert padded2.shape == (2, 3)
    assert (abs(padded1 - array1) < 1.0e-10).all()
    assert (abs(padded2 - np.array([[5, 0, 0], [2, 0, 0]])) < 1.0e-10).all()

    # match the number of columns of the 1st array
    array3 = np.arange(8).reshape(8, 1)
    array4 = np.arange(8).reshape(2, 4)
    padded3, padded4 = _zero_padding(array3, array4, pad_mode="col")
    assert padded3.shape == (8, 4)
    assert padded4.shape == (2, 4)
    assert (abs(array4 - padded4) < 1.0e-10).all()
    expected = list(range(8))
    expected.extend([0] * 24)
    expected = np.array(expected).reshape(4, 8).T
    assert (abs(expected - padded3) < 1.0e-10).all()

    # padding the padded_arrays should not change anything
    padded5, padded6 = _zero_padding(padded3, padded4, pad_mode="col")
    assert padded3.shape == (8, 4)
    assert padded4.shape == (2, 4)
    assert padded5.shape == (8, 4)
    assert padded6.shape == (2, 4)
    assert (abs(padded5 - padded3) < 1.0e-10).all()
    assert (abs(padded6 - padded4) < 1.0e-10).all()


def test_zero_padding_rows_columns():
    r"""Test _zero_padding with random array padding rows and columns."""
    array1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    array2 = np.array([[1, 2.5], [9, 5], [4, 8.5]])

    padded2, padded1 = _zero_padding(array2, array1, pad_mode="row-col")
    array2_test = np.array([[1, 2.5, 0], [9, 5, 0], [4, 8.5, 0], [0, 0, 0]])
    assert padded1.shape == (4, 3)
    assert padded2.shape == (4, 3)
    assert (abs(padded1 - array1) < 1.0e-10).all()
    assert (abs(padded2 - array2_test) < 1.0e-10).all()

    # Test in the scenario they have the same shape but fat rectangular.
    array1 = np.array([[60, 85, 86, 1.0], [85, 151, 153, 2.0], [86, 153, 158, 10.0]])
    padded2, padded1 = _zero_padding(array1, array1, pad_mode="row-col")
    assert np.all(np.abs(array1 - padded2)) < 1e-5
    assert np.all(np.abs(array1 - padded1)) < 1e-5
    assert padded1.shape == padded2.shape
    assert padded1.shape == (3, 4)

    # Test in the scenario they have the same shape but tall rectangular.
    array1 = np.random.random((2, 10))
    array2 = np.random.random((2, 10))
    padded2, padded1 = _zero_padding(array1, array2, pad_mode="row-col")
    assert np.all(np.abs(array1 - padded2)) < 1e-5
    assert np.all(np.abs(array2 - padded1)) < 1e-5
    assert padded1.shape == padded2.shape
    assert padded1.shape == (2, 10)


def test_zero_padding_square():
    r"""Test _zero_padding with squared array."""
    # Try two equivalent (but different sized) symmetric arrays
    array1 = np.array([[60, 85, 86], [85, 151, 153], [86, 153, 158]])
    array2 = np.array(
        [[60, 85, 86, 0, 0], [85, 151, 153, 0, 0], [86, 153, 158, 0, 0], [0, 0, 0, 0, 0]]
    )
    square1, square2 = _zero_padding(array1, array2, pad_mode="square")
    assert square1.shape == square2.shape
    assert square1.shape[0] == square1.shape[1]

    # Test in the scenario they have the same shape but rectangular.
    array1 = np.array([[60, 85, 86, 1.0], [85, 151, 153, 2.0], [86, 153, 158, 10.0]])
    array2 = np.array([[60, 85, 86, 1.0], [85, 151, 153, 2.0], [86, 153, 158, 10.0]])
    square1, square2 = _zero_padding(array1, array2, pad_mode="square")
    assert square1.shape == square2.shape
    assert square1.shape[0] == square1.shape[1]
    assert square1.shape[0] == 4

    # Performing the analysis on equally sized square arrays should return the same input arrays
    sym_part = np.array([[1, 7, 8, 4], [6, 4, 8, 1]])
    array1 = np.dot(sym_part, sym_part.T)
    array2 = array1
    assert array1.shape == array2.shape
    square1, square2 = _zero_padding(array1, array2, pad_mode="square")
    assert square1.shape == square2.shape
    assert square1.shape[0] == square1.shape[1]
    assert (abs(array2 - array1) < 1.0e-10).all()


def test_hide_zero_padding_flat():
    r"""Test _hide_zero_padding with flat array."""
    array0 = np.array([0, 1, 5, 8, 0, 1])
    # check array with no padding
    np.testing.assert_almost_equal(_hide_zero_padding(array0), array0, decimal=6)
    array1 = np.array([0, 1, 5, 8, 0, 1, 0])
    np.testing.assert_almost_equal(_hide_zero_padding(array1), array0, decimal=6)
    array2 = np.array([0, 1, 5, 8, 0, 1, 0, 0, 0, 0])
    np.testing.assert_almost_equal(_hide_zero_padding(array2), array0, decimal=6)


def test_hide_zero_padding_rectangular():
    r"""Test _hide_zero_padding by array with redundant row of zeros."""
    array0 = np.array([[1, 6, 0, 7, 8], [5, 7, 0, 22, 7]])
    # check array with no padding
    np.testing.assert_almost_equal(_hide_zero_padding(array0), array0, decimal=6)
    # check row-padded arrays
    array1 = np.array([[1, 6, 0, 7, 8], [5, 7, 0, 22, 7], [0, 0, 0, 0, 0]])
    np.testing.assert_almost_equal(_hide_zero_padding(array1), array0, decimal=6)
    array2 = np.array([[1, 6, 0, 7, 8], [5, 7, 0, 22, 7], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    np.testing.assert_almost_equal(_hide_zero_padding(array2), array0, decimal=6)
    # check column-padded arrays
    array3 = np.array([[1, 6, 0, 7, 8, 0], [5, 7, 0, 22, 7, 0]])
    np.testing.assert_almost_equal(_hide_zero_padding(array3), array0, decimal=6)
    array4 = np.array([[1, 6, 0, 7, 8, 0, 0, 0], [5, 7, 0, 22, 7, 0, 0, 0]])
    np.testing.assert_almost_equal(_hide_zero_padding(array4), array0, decimal=6)
    # check row- and column-padded arrays
    array5 = np.array(
        [[1, 6, 0, 7, 8, 0, 0, 0], [5, 7, 0, 22, 7, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
    )
    np.testing.assert_almost_equal(_hide_zero_padding(array5), array0, decimal=6)


def test_hide_zero_padding_square():
    r"""Test _hide_zero_padding with squared array."""
    array0 = np.array([[0, 0.5, 1.0], [0, 3.1, 4.6], [0, 7.2, 9.2]])
    # check array with no padding
    np.testing.assert_almost_equal(_hide_zero_padding(array0), array0, decimal=6)
    # check row-padded arrays
    array1 = np.array([[0, 0.5, 1.0], [0, 3.1, 4.6], [0, 7.2, 9.2], [0.0, 0.0, 0.0]])
    np.testing.assert_almost_equal(_hide_zero_padding(array1), array0, decimal=6)
    # check column-padded arrays
    array2 = np.array([[0, 0.5, 1.0, 0], [0, 3.1, 4.6, 0], [0, 7.2, 9.2, 0]])
    np.testing.assert_almost_equal(_hide_zero_padding(array2), array0, decimal=6)
    array3 = np.array([[0, 0.5, 1.0, 0, 0], [0, 3.1, 4.6, 0, 0], [0, 7.2, 9.2, 0, 0]])
    np.testing.assert_almost_equal(_hide_zero_padding(array3), array0, decimal=6)
    # check row- and column-padded arrays
    array4 = np.array(
        [
            [0, 0.5, 1.0, 0, 0],
            [0, 3.1, 4.6, 0, 0],
            [0, 7.2, 9.2, 0, 0],
            [0, 0.0, 0.0, 0, 0],
            [0, 0.0, 0.0, 0, 0],
        ]
    )
    np.testing.assert_almost_equal(_hide_zero_padding(array4), array0, decimal=6)
    # check with close to zero values.
    array4 = np.array(
        [
            [0, 0.5, 1.0, 1e-6, 1e-8],
            [0, 3.1, 4.6, 1e-6, 1e-8],
            [0, 7.2, 9.2, 1e-5, 1e-9],
            [0, 0.0, 0.0, 0.0, 1e-8],
            [0, 0.0, 0.0, 0.0, 0],
        ]
    )
    actual = np.array([[0, 0.5, 1.0, 1e-6], [0.0, 3.1, 4.6, 1e-6], [0.0, 7.2, 9.2, 1e-5]])
    np.testing.assert_almost_equal(_hide_zero_padding(array4), actual, decimal=6)


def test_translate_array():
    r"""Test _translate_array with random array."""
    array_translated = np.array([[2, 4, 6, 10], [1, 3, 7, 0], [3, 6, 9, 4]])
    # Find the means over each dimension
    column_means_translated = np.zeros(4)
    for i in range(4):
        column_means_translated[i] = np.mean(array_translated[:, i])
    # Confirm that these means are not all zero
    assert (abs(column_means_translated) > 1.0e-8).all()
    # Compute the origin-centred array
    origin_centred_array, _ = _translate_array(array_translated)
    # Confirm that the column means of the origin-centred array are all zero
    column_means_centred = np.ones(4)
    for i in range(4):
        column_means_centred[i] = np.mean(origin_centred_array[:, i])
    assert (abs(column_means_centred) < 1.0e-10).all()

    # translating a centered array does not do anything
    centred_sphere = 25.25 * np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    )
    predicted, _ = _translate_array(centred_sphere)
    expected = centred_sphere
    assert (abs(predicted - expected) < 1.0e-8).all()

    # centering a translated unit sphere dose not do anything
    shift = np.array([[1, 4, 5], [1, 4, 5], [1, 4, 5], [1, 4, 5], [1, 4, 5], [1, 4, 5]])
    translated_sphere = (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]) + shift
    )
    predicted, _ = _translate_array(translated_sphere)
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    assert (abs(predicted - expected) < 1.0e-8).all()
    # If an arbitrary array is centroid translated, the analysis applied to the original array
    # and the translated array should give identical results
    # Define an arbitrary array
    array_a = np.array([[1, 5, 7], [8, 4, 6]])
    # Define an arbitrary translation
    translate = np.array([[5, 8, 9], [5, 8, 9]])
    # Define the translated original array
    array_translated = array_a + translate
    # Begin translation analysis
    centroid_a_to_b, _ = _translate_array(array_a, array_translated)
    assert (abs(centroid_a_to_b - array_translated) < 1.0e-10).all()


def test_translate_weight():
    r"""Test _translate_array with weighted array."""
    rng = np.random.RandomState(789)
    arr = rng.randint(0, 10, (4, 3))
    weight = np.arange(1, 5)
    # center the data points to mass of center, the way in the notes
    arr_weighted = np.dot(np.diag(weight), arr)
    col_sum = np.dot(np.ones((arr_weighted.shape[0], arr_weighted.shape[0])), arr_weighted)
    # center the data points to mass of center
    arr_centered = arr - col_sum / weight.sum()
    array_a_centered, _ = _translate_array(array_a=arr, array_b=None, weight=weight)
    assert_almost_equal(arr_centered, array_a_centered)


def test_translate_invalid():
    """Test _translate_array with invalid input."""
    rng = np.random.RandomState(789)
    arr = rng.randint(0, 10, (4, 3))
    weight1 = np.arange(1, 5).reshape(-1, 1)
    # array_a, array_b=None, weight=None
    assert_raises(ValueError, _translate_array, arr, None, weight1)
    weight2 = np.array([-1, 1, 3, 4])
    assert_raises(ValueError, _translate_array, arr, None, weight2)


def test_scale_array():
    r"""Test _scale_array with random array."""
    # Rescale arbitrary array
    array_a = np.array([[6, 2, 1], [5, 2, 9], [8, 6, 4]])
    # Confirm Frobenius normaliation has transformed the array to lie on the unit sphere in
    # the R^(mxn) vector space. We must centre the array about the origin before proceeding
    array_a, _ = _translate_array(array_a)
    # Confirm proper centering
    column_means_centred = np.zeros(3)
    for i in range(3):
        column_means_centred[i] = np.mean(array_a[:, i])
    assert (abs(column_means_centred) < 1.0e-10).all()
    # Proceed with Frobenius normalization
    scaled_array, _ = _scale_array(array_a)
    # Confirm array has unit norm
    assert abs(np.sqrt((scaled_array ** 2.0).sum()) - 1.0) < 1.0e-10
    # This test verifies that when _scale_array is applied to two scaled unit spheres,
    # the Frobenius norm of each new sphere is unity.
    # Rescale spheres to unitary scale
    # Define arbitrarily scaled unit spheres
    unit_sphere = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    sphere_1 = 230.15 * unit_sphere
    sphere_2 = 0.06 * unit_sphere
    # Proceed with scaling procedure
    scaled1, _ = _scale_array(sphere_1)
    scaled2, _ = _scale_array(sphere_2)
    # Confirm each scaled array has unit Frobenius norm
    assert abs(np.sqrt((scaled1 ** 2.0).sum()) - 1.0) < 1.0e-10
    assert abs(np.sqrt((scaled2 ** 2.0).sum()) - 1.0) < 1.0e-10
    # If an arbitrary array is scaled, the scaling analysis should be able to recreate the scaled
    # array from the original
    # applied to the original array and the scaled array should give identical results.
    # Define an arbitrary array
    array_a = np.array([[1, 5, 7], [8, 4, 6]])
    # Define an arbitrary scaling factor
    scale = 6.3
    # Define the scaled original array
    array_scaled = scale * array_a
    # Begin scaling analysis
    scaled_a, _ = _scale_array(array_a, array_scaled)
    assert (abs(scaled_a - array_scaled) < 1.0e-10).all()

    # Define an arbitrary array
    array_a = np.array([[6.0, 12.0, 16.0, 7.0], [4.0, 16.0, 17.0, 33.0], [5.0, 17.0, 12.0, 16.0]])
    # Define the scaled original array
    array_scale = 123.45 * array_a
    # Verify the validity of the translate_scale analysis
    # Proceed with analysis, matching array_trans_scale to array
    predicted, _ = _scale_array(array_scale, array_a)
    # array_trans_scale should be identical to array after the above analysis
    expected = array_a
    assert (abs(predicted - expected) < 1.0e-10).all()
