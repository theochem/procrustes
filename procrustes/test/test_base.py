"""
"""

import numpy as np
from procrustes import Procrustes


def test_zero_padding_rows():
    array1 = np.array([[1, 2], [3, 4]])
    array2 = np.array([[5, 6]])
    procrust = Procrustes(array1, array2)
    # check array shape
    assert procrust.array_a.shape == (2, 2)
    assert procrust.array_b.shape == (2, 2)
    # check arrays
    assert (abs(procrust.array_a - array1) < 1.e-10).all()
    assert (abs(procrust.array_b - np.array([[5, 6], [0, 0]])) < 1.e-10).all()

    # match the number of rows of the 1st array
    padded2, padded1 = procrust.zero_padding(array2, array1, row=True)
    assert padded1.shape == (2, 2)
    assert padded2.shape == (2, 2)
    assert (abs(padded1 - array1) < 1.e-10).all()
    assert (abs(padded2 - np.array([[5, 6], [0, 0]])) < 1.e-10).all()

    # match the number of rows of the 1st array
    array3 = np.arange(8).reshape(2, 4)
    array4 = np.arange(8).reshape(4, 2)
    padded3, padded4 = procrust.zero_padding(array3, array4, row=True)
    assert padded3.shape == (4, 4)
    assert padded4.shape == (4, 2)
    assert (abs(array4 - padded4) < 1.e-10).all()
    expected = range(8)
    expected.extend([0] * 8)
    expected = np.array(expected).reshape(4, 4)
    assert (abs(expected - padded3) < 1.e-10).all()

    # padding the padded_arrays should not change anything
    padded5, padded6 = procrust.zero_padding(padded3, padded4, row=True)
    assert padded3.shape == (4, 4)
    assert padded4.shape == (4, 2)
    assert padded5.shape == (4, 4)
    assert padded6.shape == (4, 2)
    assert (abs(padded5 - padded3) < 1.e-10).all()
    assert (abs(padded6 - padded4) < 1.e-10).all()


def test_zero_padding_columns():
    array1 = np.array([[4, 7, 2], [1, 3, 5]])
    array2 = np.array([[5], [2]])
    procrust = Procrustes(array1, array2)
    assert procrust.array_a.shape == (2, 3)
    assert procrust.array_b.shape == (2, 1)
    assert (abs(procrust.array_a - array1) < 1.e-10).all()
    assert (abs(procrust.array_b - array2) < 1.e-10).all()

    # match the number of columns of the 1st array
    padded2, padded1 = procrust.zero_padding(array2, array1, column=True)
    assert padded1.shape == (2, 3)
    assert padded2.shape == (2, 3)
    assert (abs(padded1 - array1) < 1.e-10).all()
    assert (abs(padded2 - np.array([[5, 0, 0], [2, 0, 0]])) < 1.e-10).all()

    # match the number of columns of the 1st array
    array3 = np.arange(8).reshape(8, 1)
    array4 = np.arange(8).reshape(2, 4)
    padded3, padded4 = procrust.zero_padding(array3, array4, row=False, column=True)
    assert padded3.shape == (8, 4)
    assert padded4.shape == (2, 4)
    assert (abs(array4 - padded4) < 1.e-10).all()
    expected = range(8)
    expected.extend([0] * 24)
    expected = np.array(expected).reshape(4, 8).T
    assert (abs(expected - padded3) < 1.e-10).all()

    # padding the padded_arrays should not change anything
    padded5, padded6 = procrust.zero_padding(padded3, padded4, row=False, column=True)
    assert padded3.shape == (8, 4)
    assert padded4.shape == (2, 4)
    assert padded5.shape == (8, 4)
    assert padded6.shape == (2, 4)
    assert (abs(padded5 - padded3) < 1.e-10).all()
    assert (abs(padded6 - padded4) < 1.e-10).all()


def test_zero_padding_square():
    # Try two equivalent (but different sized) symmetric arrays
    array1 = np.array([[60, 85, 86], [85, 151, 153], [86, 153, 158]])
    array2 = np.array([[60, 85, 86, 0, 0], [85, 151, 153, 0, 0],
                       [86, 153, 158, 0, 0], [0, 0, 0, 0, 0]])
    assert(array1.shape != array2.shape)
    procrust = Procrustes(array1, array2)
    square1, square2 = procrust.zero_padding(
        array1, array2, row=False, column=False, square=True)
    assert(square1.shape == square2.shape)
    assert(square1.shape[0] == square1.shape[1])

    # Performing the analysis on equally sized square arrays should return the same input arrays
    sym_part = np.array([[1, 7, 8, 4], [6, 4, 8, 1]])
    array1 = np.dot(sym_part, sym_part.T)
    array2 = array1
    assert(array1.shape == array2.shape)
    procrust = Procrustes(array1, array2)
    square1, square2 = procrust.zero_padding(array1, array2, row=False, column=False, square=True)
    assert(square1.shape == square2.shape)
    assert(square1.shape[0] == square1.shape[1])
    assert(abs(array2 - array1) < 1.e-10).all()


def test_hide_zero_padding_array():
    array0 = np.array([[1, 6, 7, 8], [5, 7, 22, 7]])
    # Create (arbitrary) pads to add onto the permuted input array, array_permuted
    m, n = array0.shape
    arb_pad_col = 27
    arb_pad_row = 13
    pad_vertical = np.zeros((m, arb_pad_col))
    pad_horizontal = np.zeros((arb_pad_row, n + arb_pad_col))
    array1 = np.concatenate((array0, pad_vertical), axis=1)
    array1 = np.concatenate((array1, pad_horizontal), axis=0)
    # Assert array has been zero padded
    assert(array0.shape != array1.shape)
    # Confirm that after hide_zero_padding_array has been applied, the arrays are of equal size and
    # are identical
    procrust = Procrustes(array0, array1)
    hide_array0 = procrust.hide_zero_padding_array(array0)
    hide_array1 = procrust.hide_zero_padding_array(array1)
    assert(hide_array0.shape == hide_array1.shape)
    assert(abs(hide_array0 - hide_array1) < 1.e-10).all()

    # Define an arbitrary array
    array0 = np.array([[124.25, 625.15, 725.64, 158.51], [536.15, 367.63, 322.62, 257.61],
                       [361.63, 361.63, 672.15, 631.63]])
    # Create (arbitrary) pads to add onto the permuted input array, array_permuted
    m, n = array0.shape
    arb_pad_col = 14
    arb_pad_row = 19
    pad_vertical = np.zeros((m, arb_pad_col))
    pad_horizontal = np.zeros((arb_pad_row, n + arb_pad_col))
    array1 = np.concatenate((array0, pad_vertical), axis=1)
    array1 = np.concatenate((array1, pad_horizontal), axis=0)
    # Assert array has been zero padded
    assert(array0.shape != array1.shape)
    # Confirm that after hide_zero_padding_array has been applied, the arrays are of equal size and
    # are identical
    procrust = Procrustes(array0, array1)
    hide_array0 = procrust.hide_zero_padding_array(array0)
    hide_array1 = procrust.hide_zero_padding_array(array1)
    assert(hide_array0.shape == hide_array1.shape)
    assert(abs(hide_array0 - hide_array1) < 1.e-10).all()


def test_translate_array():
    array_translated = np.array([[2, 4, 6, 10], [1, 3, 7, 0], [3, 6, 9, 4]])
    # Instantiate Procrustes class with arbitrary second argument
    procrust = Procrustes(array_translated, array_translated.T)
    # Find the means over each dimension
    column_means_translated = np.zeros(4)
    for i in range(4):
        column_means_translated[i] = np.mean(array_translated[:, i])
    # Confirm that these means are not all zero
    assert (abs(column_means_translated) > 1.e-8).all()
    # Compute the origin-centred array
    origin_centred_array, unused = procrust.translate_array(array_translated)
    # Confirm that the column means of the origin-centred array are all zero
    column_means_centred = np.ones(4)
    for i in range(4):
        column_means_centred[i] = np.mean(origin_centred_array[:, i])
    assert (abs(column_means_centred) < 1.e-10).all()

    # translating a centered array does not do anything
    centred_sphere = 25.25 * \
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    procrust = Procrustes(centred_sphere, centred_sphere.T)
    predicted, unused = procrust.translate_array(centred_sphere)
    expected = centred_sphere
    assert(abs(predicted - expected) < 1.e-8).all()

    # centering a translated unit sphere dose not do anything
    shift = np.array([[1, 4, 5], [1, 4, 5], [1, 4, 5], [1, 4, 5], [1, 4, 5], [1, 4, 5]])
    translated_sphere = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])\
        + shift
    procrust = Procrustes(translated_sphere, translated_sphere.T)
    predicted, unused = procrust.translate_array(translated_sphere)
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    assert(abs(predicted - expected) < 1.e-8).all()
    # If an arbitrary array is centroid translated, the analysis applied to the original array
    # and the translated array should give identical results
    # Define an arbitrary array
    array_a = np.array([[1, 5, 7], [8, 4, 6]])
    # Define an arbitrary translation
    translate = np.array([[5, 8, 9], [5, 8, 9]])
    # Define the translated original array
    array_translated = array_a + translate
    # Begin translation analysis
    procrust = Procrustes(array_a, array_translated)
    centroid_a_to_b, unused = procrust.translate_array(array_a, array_translated)
    assert(abs(centroid_a_to_b - array_translated) < 1.e-10).all()


def test_scale_array():
    # Rescale arbitrary array
    array = np.array([[6, 2, 1], [5, 2, 9], [8, 6, 4]])
    # Create (arbitrary second argument) Procrustes instance
    procrust = Procrustes(array, array.T)
    # Confirm Frobenius normaliation has transformed the array to lie on the unit sphere in
    # the R^(mxn) vector space. We must centre the array about the origin before proceeding
    array, unused = procrust.translate_array(array)
    # Confirm proper centering
    column_means_centred = np.zeros(3)
    for i in range(3):
        column_means_centred[i] = np.mean(array[:, i])
    assert (abs(column_means_centred) < 1.e-10).all()
    # Proceed with Frobenius normalization
    scaled_array, unused = procrust.scale_array(array)
    # Confirm array has unit norm
    assert(abs(np.sqrt((scaled_array**2.).sum()) - 1.) < 1.e-10)
    # This test verifies that when scale_array is applied to two scaled unit spheres,
    # the Frobenius norm of each new sphere is unity.
    # Rescale spheres to unitary scale
    # Define arbitrarily scaled unit spheres
    unit_sphere = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    sphere_1 = 230.15 * unit_sphere
    sphere_2 = .06 * unit_sphere
    # Proceed with scaling procedure
    procrust = Procrustes(sphere_1, sphere_2)
    scaled1, unused = procrust.scale_array(sphere_1)
    scaled2, unused = procrust.scale_array(sphere_2)
    # Confirm each scaled array has unit Frobenius norm
    assert(abs(np.sqrt((scaled1**2.).sum()) - 1.) < 1.e-10)
    assert(abs(np.sqrt((scaled2**2.).sum()) - 1.) < 1.e-10)
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
    procrust = Procrustes(array_a, array_scaled)
    scaled_a, unused = procrust.scale_array(array_a, array_scaled)
    assert(abs(scaled_a - array_scaled) < 1.e-10).all()

    # Define an arbitrary array
    array = np.array([[6., 12., 16., 7.], [4., 16., 17., 33.], [5., 17., 12., 16.]])
    # Define the scaled original array
    array_scale = 123.45 * array
    # Verify the validity of the translate_scale analysis
    procrust = Procrustes(array, array_scale)
    # Proceed with analysis, matching array_trans_scale to array
    predicted, unused = procrust.scale_array(array_scale, array)
    # array_trans_scale should be identical to array after the above analysis
    expected = array
    assert(abs(predicted - expected) < 1.e-10).all()


def test_translate_scale_array():
    # Define an arbitrary array
    array = np.array([[5, 3, 2, 5], [7, 5, 4, 3]])
    # Proceed with the translate_scale process.
    procrust = Procrustes(array, array.T)
    array_trans_scale_1, unused, _ = procrust.translate_scale_array(array)
    # Perform the process again using the already translated and scaled array as input
    array_trans_scale_2, unused, _ = procrust.translate_scale_array(array_trans_scale_1)
    assert(abs(array_trans_scale_1 - array_trans_scale_2) < 1.e-10).all()

    # Define an arbitrary array
    array = np.array([[1., 4., 6., 7.], [4., 6., 7., 3.], [5., 7., 3., 1.]])
    # Define an arbitrary centroid shift
    shift = np.array([[1., 4., 6., 9.], [1., 4., 6., 9.], [1., 4., 6., 9.]])
    # Define the translated and scaled original array
    array_trans_scale = 14.76 * array + shift
    # Verify the validity of the translate_scale analysis
    procrust = Procrustes(array, array_trans_scale)
    # Returns an object with an origin centred centroid unit Frobenius norm
    predicted, unused, _ = procrust.translate_scale_array(array_trans_scale)
    # Returns the same object, origin centred and unit Frobenius norm
    expected, _, _ = procrust.translate_scale_array(array)
    assert(abs(predicted - expected) < 1.e-10).all()

    # Define an arbitrary array
    array = np.array([[6., 12., 16., 7.], [4., 16., 17., 33.], [5., 17., 12., 16.]])
    # Define an arbitrary centroid shift
    shift = np.array([[3., 7., 9., 1.], [3., 7., 9., 1.], [3., 7., 9., 1.]])
    # Define the translated and scaled original array
    array_trans_scale = 123.45 * array + shift
    # Verify the validity of the translate_scale analysis
    procrust = Procrustes(array, array_trans_scale)
    # Proceed with analysis, matching array_trans_scale to array
    predicted, trans_vec, scale_vec = procrust.translate_scale_array(array_trans_scale, array)
    # array_trans_scale should be identical to array after the above analysis
    expected = array
    assert(abs(predicted - expected) < 1.e-10).all()


def test_eigenvalue_decomposition():
    array = np.array([[-1. / 2, 3. / 2], [3. / 2, -1. / 2]])
    procrust = Procrustes(array, array.T)
    assert(procrust.is_diagonalizable(array) is True)
    s_predicted, u_predicted = procrust.eigenvalue_decomposition(array)
    s_expected = np.array([1, -2])
    assert(abs(np.dot(u_predicted, u_predicted.T) - np.eye(2)) < 1.e-8).all()
    # The eigenvalue decomposition must return the original array
    predicted = np.dot(u_predicted, np.dot(np.diag(s_predicted), u_predicted.T))
    assert (abs(predicted - array) < 1.e-8).all()
    assert(abs(s_predicted - s_expected) < 1.e-8).all()
    # check that product of u, s, and u.T obtained from eigenvalue_decomposition gives original array
    array = np.array([[3, 1], [1, 3]])
    procrust = Procrustes(array, array.T)
    assert(procrust.is_diagonalizable(array) is True)
    s_predicted, u_predicted = procrust.eigenvalue_decomposition(array)
    s_expected = np.array([4, 2])
    assert(abs(np.dot(u_predicted, u_predicted.T) - np.eye(2)) < 1.e-8).all()
    # The eigenvalue decomposition must return the original array
    predicted = np.dot(u_predicted, np.dot(np.diag(s_predicted), u_predicted.T))
    assert (abs(predicted - array) < 1.e-8).all()
    assert(abs(s_predicted - s_expected) < 1.e-8).all()


def test_centroid():
    # Define an arbitrary array
    array = np.array([[6., 12., 16., 7.], [4., 16., 17., 33.], [5., 17., 12., 16.]])
    procrust = Procrustes(array, array.T)
    array_centred, translate = procrust.translate_array(array)
    assert(abs(procrust.centroid(array_centred)) < 1.e-10).all()

    # Define an arbitrary array
    array = np.array([[6325.26, 1232.46, 1356.75, 7351.64], [4351.36, 1246.63, 1247.63, 3243.64]])
    procrust = Procrustes(array, array.T)
    array_centred, translate = procrust.translate_array(array)
    assert(abs(procrust.centroid(array_centred)) < 1.e-10).all()
    # Even if the array is zero-padded, the correct translation about the origin is obtained.
    # Define an arbitrary, zero-padded array
    array = np.array([[1.5294e-4, 1.242e-5, 1.624e-3, 7.35e-4], [4.534e-5, 1.652e-5, 1.725e-5, 3.314e-4],
                      [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    procrust = Procrustes(array, array.T)
    array_centred, translate = procrust.translate_array(array)
    assert(abs(procrust.centroid(array_centred)) < 1.e-10).all()


def test_frobenius_norm():
    # Define an arbitrary, zero-padded array
    array = np.array([[1.5294e-4, 1.242e-5, 1.624e-3, 7.35e-4], [4.534e-5, 1.652e-5, 1.725e-5, 3.314e-4],
                      [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    procrust = Procrustes(array, array.T)
    array_scaled, scaling = procrust.scale_array(array)
    assert(abs(procrust.frobenius_norm(array_scaled) - 1.) < 1.e-10).all()

    # Define an arbitrary, zero-padded array
    array = np.array([[6325.26, 1232.46, 1356.75, 7351.64, 0, 0], [4351.36, 1246.63, 1247.63, 3243.64, 0, 0],
                      [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    procrust = Procrustes(array, array.T)
    array_scaled, scaling = procrust.scale_array(array)
    assert(abs(procrust.frobenius_norm(array_scaled) - 1.) < 1.e-10).all()

    # Define an arbitrary
    array = np.array([[6., 12., 16., 7.], [4., 16., 17., 33.], [5., 17., 12., 16.]])
    procrust = Procrustes(array, array.T)
    array_scaled, scaling = procrust.scale_array(array)
    assert(abs(procrust.frobenius_norm(array_scaled) - 1.) < 1.e-10).all()
