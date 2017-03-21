__author__ = 'Jonny'


from procrustes import *
import numpy as np



def test_zero_padding_rows():
    array1 = np.array([[1, 2], [3, 4]])
    array2 = np.array([[5, 6]])
    procrust = Procrustes(array1, array2)
    # match the number of rows of the 2nd array (automatically down when initiating the class)
    assert array1.shape == (2, 2)
    assert array2.shape == (1, 2)
    assert procrust.array_a.shape == (2, 2)
    assert procrust.array_b.shape == (2, 2)
    expected = np.array([[5, 6], [0, 0]])
    assert (abs(procrust.array_a - array1) < 1.e-10).all()
    assert (abs(procrust.array_b - expected) < 1.e-10).all()

    # match the number of rows of the 1st array
    padded_array2, padded_array1 = procrust.zero_padding(array2, array1, row=True)
    assert array1.shape == (2, 2)
    assert array2.shape == (1, 2)
    assert padded_array1.shape == (2, 2)
    assert padded_array2.shape == (2, 2)
    assert (abs(padded_array1 - array1) < 1.e-10).all()
    assert (abs(padded_array2 - expected) < 1.e-10).all()

    # match the number of rows of the 1st array
    array3 = np.arange(8).reshape(2, 4)
    array4 = np.arange(8).reshape(4, 2)
    padded_array3, padded_array4 = procrust.zero_padding(array3, array4, row=True)
    assert array3.shape == (2, 4)
    assert array4.shape == (4, 2)
    assert padded_array3.shape == (4, 4)
    assert padded_array4.shape == (4, 2)
    assert (abs(array4 - padded_array4) < 1.e-10).all()
    expected = range(8)
    expected.extend([0]*8)
    expected = np.array(expected).reshape(4, 4)
    assert (abs(expected - padded_array3) < 1.e-10).all()

    # padding the padded_arrays should not change anything
    padded_array5, padded_array6 = procrust.zero_padding(padded_array3, padded_array4, row=True)
    assert padded_array3.shape == (4, 4)
    assert padded_array4.shape == (4, 2)
    assert padded_array5.shape == (4, 4)
    assert padded_array6.shape == (4, 2)
    assert (abs(padded_array5 - padded_array3) < 1.e-10).all()
    assert (abs(padded_array6 - padded_array4) < 1.e-10).all()


def test_zero_padding_columns():
    array1 = np.array([[4, 7, 2], [1, 3, 5]])
    array2 = np.array([[5], [2]])
    procrust = Procrustes(array1, array2)
    assert array1.shape == (2, 3)
    assert array2.shape == (2, 1)
    assert procrust.array_a.shape == (2, 3)
    assert procrust.array_b.shape == (2, 1)
    expected = np.array([[5, 0, 0], [2, 0, 0]])
    assert (abs(procrust.array_a - array1) < 1.e-10).all()
    assert (abs(procrust.array_b - array2) < 1.e-10).all()

    # match the number of columns of the 1st array
    padded_array2, padded_array1 = procrust.zero_padding(array2, array1, column=True)
    assert array1.shape == (2, 3)
    assert array2.shape == (2, 1)
    assert padded_array1.shape == (2, 3)
    assert padded_array2.shape == (2, 3)
    assert (abs(padded_array1 - array1) < 1.e-10).all()
    assert (abs(padded_array2 - expected) < 1.e-10).all()

    # match the number of columns of the 1st array
    array3 = np.arange(8).reshape(8, 1)
    array4 = np.arange(8).reshape(2, 4)
    padded_array3, padded_array4 = procrust.zero_padding(array3, array4, row=False, column=True)
    assert array3.shape == (8, 1)
    assert array4.shape == (2, 4)
    assert padded_array3.shape == (8, 4)
    assert padded_array4.shape == (2, 4)
    assert (abs(array4 - padded_array4) < 1.e-10).all()
    expected = range(8)
    expected.extend([0]*24)
    expected = np.array(expected).reshape(4, 8).T
    assert (abs(expected - padded_array3) < 1.e-10).all()

    # padding the padded_arrays should not change anything
    padded_array5, padded_array6 = procrust.zero_padding(padded_array3, padded_array4, row=False, column=True)
    assert padded_array3.shape == (8, 4)
    assert padded_array4.shape == (2, 4)
    assert padded_array5.shape == (8, 4)
    assert padded_array6.shape == (2, 4)
    assert (abs(padded_array5 - padded_array3) < 1.e-10).all()
    assert (abs(padded_array6 - padded_array4) < 1.e-10).all()


def test_zero_padding_square():
    # Try two equivalent (but different sized) symmetric arrays
    sym_array1 = np.array([[60,  85,  86], [85, 151, 153], [86, 153, 158]])
    sym_array2 = np.array([[60,  85,  86, 0, 0], [85, 151, 153, 0, 0], [86, 153, 158, 0, 0], [0, 0, 0, 0, 0]])
    assert(sym_array1.shape != sym_array2.shape)
    procrust = Procrustes(sym_array1, sym_array2)
    square_1, square_2 = procrust.zero_padding(sym_array1, sym_array2, row=False, column=False, square=True)
    assert(square_1.shape == square_2.shape)
    assert(square_1.shape[0] == square_1.shape[1])

    # Performing the analysis on equally sized square arrays should return the same input arrays
    sym_part = np.array([[1, 7, 8, 4], [6, 4, 8, 1]])
    sym_array1 = np.dot(sym_part, sym_part.T)
    sym_array2 = sym_array1
    assert(sym_array1.shape == sym_array2.shape)
    procrust = Procrustes(sym_array1, sym_array2)
    square_1, square_2 = procrust.zero_padding(sym_array1, sym_array2, row=False, column=False, square=True)
    assert(square_1.shape == square_2.shape)
    assert(square_1.shape[0] == square_1.shape[1])
    assert(abs(sym_array2 - sym_array1) < 1.e-10).all()


#-------------------------------------------

def test_hide_zero_padding_array():
    # Define an arbitrary array
    array0 = np.array([[1, 6, 7, 8], [5, 7, 22, 7]])
    # Create (arbitrary) pads to add onto the permuted input array, array_permuted
    m, n = array0.shape
    arb_pad_col = 27
    arb_pad_row = 13
    pad_vertical = np.zeros((m, arb_pad_col))
    pad_horizontal = np.zeros((arb_pad_row, n+arb_pad_col))
    array1 = np.concatenate((array0, pad_vertical), axis=1)
    array1 = np.concatenate((array1, pad_horizontal), axis=0)
    # Assert array has been zero padded
    assert(array0.shape != array1.shape)
    # Confirm that after hide_zero_padding has been applied, the arrays are of equal size and
    # are identical
    procrust = Procrustes(array0, array1)
    hide_array0 = procrust.hide_zero_padding_array(array0)
    hide_array1 = procrust.hide_zero_padding_array(array1)
    assert(hide_array0.shape == hide_array1.shape)
    assert(abs(hide_array0 - hide_array1) < 1.e-10).all()


# ----------------------------------------------------------


def test_translate_array():

    # Define an arbitrary nd array
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
    origin_centred_array = procrust.translate_array(array_translated)
    # Confirm that the column means of the origin-centred array are all zero
    column_means_centred = np.ones(4)
    for i in range(4):
        column_means_centred[i] = np.mean(origin_centred_array[:, i])
    assert (abs(column_means_centred) < 1.e-10).all()

    # Translating an already centred object should return the original object
    centred_sphere = 25.25 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    procrust = Procrustes(centred_sphere, centred_sphere.T)
    predicted = procrust.translate_array(centred_sphere)
    expected = centred_sphere
    assert(abs(predicted - expected) < 1.e-8).all()

    # Centering a translated unit sphere should return the unit sphere
    translate_array = np.array([[1, 4, 5], [1, 4, 5], [1, 4, 5], [1, 4, 5], [1, 4, 5], [1, 4, 5]])
    translated_sphere = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])\
        + translate_array
    procrust = Procrustes(translated_sphere, translated_sphere.T)
    predicted = procrust.translate_array(translated_sphere)
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    assert(abs(predicted - expected) < 1.e-8).all()


def test_scale_array():

    # Rescale arbitrary array
    array = np.array([[6, 2, 1], [5, 2, 9], [8, 6, 4]])
    # Create (arbitrary second argument) Procrustes instance
    procrust = Procrustes(array, array.T)
    # Confirm Frobenius normaliation has transformed the array to lie on the unit sphere in
    # the R^(mxn) vector space. We must centre the array about the origin before proceding
    array = procrust.translate_array(array)
    # Confirm proper centering
    column_means_centred = np.zeros(3)
    for i in range(3):
        column_means_centred[i] = np.mean(array[:, i])
    assert (abs(column_means_centred) < 1.e-10).all()
    # Proceed with Frobenius normalization
    scaled_array = procrust.scale_array(array)
    # Confirm array has unit norm
    assert(abs(np.sqrt((scaled_array**2.).sum()) - 1.) < 1.e-10)

    # Rescale spheres to unitary scale
    # Define arbitrarily scaled unit spheres
    unit_sphere = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    sphere_1 = 230.15 * unit_sphere
    sphere_2 = .06 * unit_sphere
    # Proceed with scaling procedure
    procrust = Procrustes(sphere_1, sphere_2)
    scaled1 = procrust.scale_array(sphere_1)
    scaled2 = procrust.scale_array(sphere_2)
    # Confirm each scaled array has unit Frobenius norm
    assert(abs(np.sqrt((scaled1**2.).sum()) - 1.) < 1.e-10)
    assert(abs(np.sqrt((scaled2**2.).sum()) - 1.) < 1.e-10)

# -------------------------------------------------------------


def test_translate_scale_array():
    # Performing translate_scale on an array multiple times always returns the same solution
    # Define an arbitrary array
    array = np.array([[5, 3, 2, 5], [7, 5, 4, 3]])
    # Proceed with the translate_scale process. Arbitrary parameters into Procrustes
    procrust = Procrustes(array, array.T)
    array_trans_scale_1 = procrust.translate_scale_array(array)
    # Perform the process again using the already translated and scaled array as input
    array_trans_scale_2 = procrust.translate_scale_array(array_trans_scale_1)
    assert(abs(array_trans_scale_1 - array_trans_scale_2) < 1.e-10).all()

    # Applying translate scale to an array which is translated and scaled returns
    # the original array
    # Define an arbitrary array
    array = np.array([[1., 4., 6., 7.], [4., 6., 7., 3.], [5., 7., 3., 1.]])
    # Define an arbitrary shift
    shift = np.array([[1., 4., 6., 9.], [1., 4., 6., 9.], [1., 4., 6., 9.]])
    # Define the translated and scaled original array
    array_trans_scale = 14.76 * array + shift
    # Verify the validity of translate_scale
    procrust = Procrustes(array, array_trans_scale)
    predicted = procrust.translate_scale_array(array_trans_scale)
    expected = procrust.translate_scale_array(array)
    assert(abs(predicted - expected) < 1.e-10).all()

# ----------------------------------------------------


def test_singular_value_decomposition():
    pass
    """ This is a numpy function, please refer to their
    documents for their testing procedure
    """

#------------------------------------------------------


def test_eigenvalue_decomposition():
    # If eigenvalue decomposition is doable (dimension of eigenspace = dimension of array)
    # function should return the appropriate decomposition
    array = np.array([[-1./2, 3./2], [3./2, -1./2]])
    procrust = Procrustes(array, array.T)
    assert(procrust.is_diagonalizable(array) == True)
    s_predicted, u_predicted = procrust.eigenvalue_decomposition(array)
    s_expected = np.array([1, -2])
    assert(abs(np.dot(u_predicted, u_predicted.T) - np.eye(2)) < 1.e-8).all()
    # The eigenvalue decomposition must return the original array
    predicted = np.dot(u_predicted, np.dot(np.diag(s_predicted), u_predicted.T))
    assert (abs(predicted - array) < 1.e-8).all()
    assert(abs(s_predicted - s_expected) < 1.e-8).all()

    array = np.array([[3, 1], [1, 3]])
    procrust = Procrustes(array, array.T)
    assert(procrust.is_diagonalizable(array) == True)
    s_predicted, u_predicted = procrust.eigenvalue_decomposition(array)
    s_expected = np.array([4, 2])
    assert(abs(np.dot(u_predicted, u_predicted.T) - np.eye(2)) < 1.e-8).all()
    # The eigenvalue decomposition must return the original array
    predicted = np.dot(u_predicted, np.dot(np.diag(s_predicted), u_predicted.T))
    assert (abs(predicted - array) < 1.e-8).all()
    assert(abs(s_predicted - s_expected) < 1.e-8).all()











