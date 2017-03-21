__author__ = 'Jonny'


from procrustes import *
import numpy as np
from math import *
import unittest


def test_orthogonal():
    """
    This test verifies that orthogonal procrustes analysis
    works with equivalent input arrays of different sizes (i.e. zero padded arrays)
    """
    # Define an arbitrary array
    array = np.array([[1, 4], [7, 9]])
    # Define an arbitrary rotational transformation
    theta = pi/2
    rot_trans = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    assert(abs(np.linalg.det(rot_trans) - 1) < 1.e-8).all()
    assert(abs(np.dot(rot_trans, rot_trans.T) - np.eye(2)) < 1.e-8).all()
    # Define an arbitrary reflection transformation
    refl_trans = np.array([[1, 0], [0, -1]])
    assert(abs(np.linalg.det(refl_trans) + 1) < 1.e-8).all()
    assert(abs(np.dot(refl_trans, refl_trans.T) - np.eye(2)) < 1.e-8).all()
    # Define the orthogonal transformation (composition of rotation and reflection transformations)
    ortho_trans = np.dot(rot_trans, refl_trans)
    assert(abs(np.dot(ortho_trans, ortho_trans.T) - np.eye(2)) < 1.e-8).all()
    # Define the orthogonally transformed original array
    array_ortho_transf = np.dot(array, ortho_trans)
    # Create (arbitrary) pads to add onto the orthogonally transformed input array, array_ortho_transf
    m, n = array_ortho_transf.shape
    arb_pad_col = 3
    arb_pad_row = 7
    pad_vertical = np.zeros((m, arb_pad_col))
    pad_horizontal = np.zeros((arb_pad_row, n+arb_pad_col))
    array_ortho_transf = np.concatenate((array_ortho_transf, pad_vertical), axis=1)
    array_ortho_transf = np.concatenate((array_ortho_transf, pad_horizontal), axis=0)
    # Proceed with orthogonal procrustes analysis
    ortho = OrthogonalProcrustes(array, array_ortho_transf)
    u_optimum, array_transformed, error = ortho.calculate()
    # Assert that the analysis returns zero error and is correct
    assert error < 1.e-10

    """
    This test verifies that orthogonal procrustes analysis
    works when the input arrays are identical
    """
    # If the input arrays are equivalent, the transformed array should be equal to the inputs
    # Define arbitrary array
    array_a = np.array([[1, 5, 6, 7], [1, 2, 9, 4]])
    # Match arbitrary array (above) to itself
    array_b = np.array([[1, 5, 6, 7], [1, 2, 9, 4]])
    assert(abs(array_a - array_b) < 1.e-8).all()
    # Proceed with orthogonal procrustes analysis
    ortho = OrthogonalProcrustes(array_a, array_b)
    u_optimum, a_transformed, error = ortho.calculate()
    # The transformation should return zero error
    assert error < 1.e-10
    """
    This test verifies that orthogonal procrustes analysis is capable of matching an input array
    to itself after it undergoes translation, scaling, and orthogonal transformations
    """
    # Define an arbitrary array
    array_a = np.array([[-1, 0], [-1, 1], [1, 1], [1, 0]])
    # Scale, translate and shift the initial array
    shift = np.array([[3, 3], [3, 3], [3, 3], [3, 3]])
    array_b = 4 * array_a + shift
    # Rotate & reflect the initial array, after shifting and scaling
    theta = pi / 4
    rot_array = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    assert(np.dot(rot_array, rot_array.T) - np.eye(2) < 1.e-8).all()
    assert(np.linalg.det(rot_array) == 1)
    refl_array = np.array([[1, 0], [0, -1]])
    assert(np.dot(refl_array, refl_array.T) - np.eye(2) < 1.e-8).all()
    assert(abs(np.linalg.det(refl_array) + 1) < 1.e-8)
    ortho_trans = np.dot(rot_array, refl_array)
    # Define the translated, scaled, and orthogonally transformed array
    array_b = np.dot(array_b, ortho_trans)
    # Proceed with the orthgonal procrustes analysis
    ortho = OrthogonalProcrustes(array_a, array_b, translate_scale=True)
    u_optimum, a_transformed, error = ortho.calculate()
    # The transformation should return zero error
    assert error < 1.e-10

# -----------------------------------------------------------------------


def test_permutation():
    """
    This test verifies that orthogonal procrustes analysis
    works with equivalent input arrays of different sizes (i.e. zero padded arrays)
    """
    # Define arbitrary array
    array = np.array([[1, 5, 8, 4], [1, 5, 7, 2], [1, 6, 9, 3], [2, 7, 9, 4]])
    # Define an arbitrary permutation transformation
    perm_array = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]])
    assert(abs(np.linalg.det(perm_array)) - 1. < 1.e-8)
    assert(abs([x for x in perm_array.flatten().tolist() if x != 0] - np.ones(4)) < 1.e-8).all()
    # Define the permuted original array
    array_permuted = np.dot(array, perm_array)
    # Create (arbitrary) pads to add onto the permuted input array, array_permuted
    m, n = array_permuted.shape
    arb_pad_col = 6
    arb_pad_row = 3
    pad_vertical = np.zeros((m, arb_pad_col))
    pad_horizontal = np.zeros((arb_pad_row, n+arb_pad_col))
    array_permuted = np.concatenate((array_permuted, pad_vertical), axis=1)
    array_permuted = np.concatenate((array_permuted, pad_horizontal), axis=0)
    # Proceed with permutation procrustes analysis
    perm = PermutationProcrustes(array, array_permuted)
    perm_optimum, array_transformed_predicted, total_potential, error = perm.calculate()
    # Perm_optimum must be a permutation array
    assert(abs(np.linalg.det(perm_optimum)) - 1. < 1.e-8)
    assert(abs([x for x in perm_optimum.flatten().tolist() if x != 0] - np.ones(4)) < 1.e-8).all()
    # Assert that the analysis returns zero error
    assert error < 1.e-10
    """
    This test verifies that permutation procrustes analysis
    works when the input arrays are identical
    """
    # If the input arrays are equivalent, the permutation transformation must be the identity
    # Define an arbitrary array
    array_a = np.array([[3, 5, 4, 3], [1, 6, 5, 4], [1, 6, 4, 2]])
    # Match arbitrary array (above) to itself
    array_b = np.array([[3, 5, 4, 3], [1, 6, 5, 4], [1, 6, 4, 2]])
    assert(abs(array_a - array_b) < 1.e-8).all()
    # Proceed with procrustes analysis
    perm = PermutationProcrustes(array_a, array_b)
    perm_optimum, a_transformed, total_potential, error = perm.calculate()
    # Perm_optimum must be a permutation array
    assert(abs(np.linalg.det(perm_optimum) - 1.) < 1.e-8)
    assert(abs([x for x in perm_optimum.flatten().tolist() if x != 0] - np.ones(4)) < 1.e-8).all()
    # The expected permutation-transformation is the 4x4 identity array
    expected = np.eye(4)
    # The transformation should return zero error
    assert error < 1.e-8
    # The transformation must be the 4x4 identity
    assert(abs(perm_optimum - expected) < 1.e-8).all()
    """
    This test verifies that permutation procrustes analysis is capable of matching an input array
    to itself after it undergoes translation, scaling, and permutation transformations
    """
    # Define arbitrary array
    array_a = np.array([[1, 5, 8, 4], [1, 5, 7, 2], [1, 6, 9, 3], [2, 7, 9, 4]])
    # Define an arbitrary translation
    shift = np.array([[6, 1, 5, 3], [6, 1, 5, 3], [6, 1, 5, 3], [6, 1, 5, 3]])
    # Translate and scale the initial array
    array_b = 3.78 * array_a + shift
    # Define an arbitrary permutation transformation
    perm_array = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    assert(abs(np.linalg.det(perm_array) - 1.) < 1.e-8)
    assert(abs([x for x in perm_array.flatten().tolist() if x != 0] - np.ones(4)) < 1.e-8).all()
    # Define the translated, scaled, and permuted initial array
    array_b = np.dot(array_b, perm_array)
    # Proceed with permutation procrustes analysis
    perm = PermutationProcrustes(array_a, array_b, translate_scale=True)
    perm_optimum, array_transformed_predicted, total_potential, error = perm.calculate()
    # Perm_optimum must be a permutation array
    assert(abs(np.linalg.det(perm_optimum) - 1.) < 1.e-8)
    assert(abs([x for x in perm_optimum.flatten().tolist() if x != 0] - np.ones(4)) < 1.e-8).all()
    # The transformation should return zero error
    assert error < 1.e-10

# ---------------------------------------------------------------------


def test_rotational_orthogonal():
    """
    This test verifies that orthogonal procrustes analysis
    works with equivalent input arrays of different sizes (i.e. zero padded arrays)
    """
    # Define arbitrary array
    array = np.array([[1, 7], [9, 4]])
    # Define an arbitrary rotation transformation
    theta = pi/4
    rot_array = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    # Define the rotated original array
    array_rotated = np.dot(array, rot_array)
    assert(abs(np.dot(rot_array, rot_array.T) - np.eye(2)) < 1.e-8).all()
    assert(abs(np.linalg.det(rot_array) - 1) < 1.e-10)
    # Create (arbitrary) pads to add onto the rotated input array, array_rotated
    m, n = array_rotated.shape
    arb_pad_col = 17
    arb_pad_row = 21
    pad_vertical = np.zeros((m, arb_pad_col))
    pad_horizontal = np.zeros((arb_pad_row, n+arb_pad_col))
    array_rotated = np.concatenate((array_rotated, pad_vertical), axis=1)
    array_rotated = np.concatenate((array_rotated, pad_horizontal), axis=0)
    # Proceed with rotational-orthogonal procrustes analysis
    rot_ortho = RotationalOrthogonalProcrustes(array, array_rotated)
    r, array_transformed_predicted, error = rot_ortho.calculate()
    # Verify r is purely rotational and equivalent to rot_array
    assert(abs(np.dot(r, r.T) - np.eye(2)) < 1.e-8).all()
    assert(abs(np.linalg.det(r) - 1) < 1.e-10)
    # The transformation should return zero error
    assert error < 1.e-8
    """
    This test verifies that permutation procrustes analysis
    works when the input arrays are identical
    """
    # Define an arbitrary array
    array_a = np.array([[3, 6, 2, 1], [5, 6, 7, 6], [2, 1, 1, 1]])
    # Match arbitrary array (above) to itself
    array_b = np.array([[3, 6, 2, 1], [5, 6, 7, 6], [2, 1, 1, 1]])
    assert(abs(array_a - array_b) < 1.e-8).all()
    # Proceed with rotational-orthogonal procrustes analysis
    rot_ortho = RotationalOrthogonalProcrustes(array_a, array_b)
    r, a_transformed, error = rot_ortho.calculate()
    # r must be a purely rotational-orthogonal transformation
    assert(abs(np.dot(r, r.T) - np.eye(4)) < 1.e-8).all()
    assert(abs(np.linalg.det(r) - 1) < 1.e-10)
    # The transformation should return zero error
    assert error < 1.e-8
    """
    This test verifies that rotational-orthogonal procrustes analysis is capable of matching an input array
    to itself after it undergoes translation, scaling, and rotational transformations
    """

    """
    THIS TEST HAS AN ISSUE: PLEASE FIX IT
    """

    # Define arbitrary array
    array_a = np.array([[1., 7., 8.], [4., 6., 8.], [7., 9., 4.], [6., 8., 23.]])
    # Define an arbitrary translation
    shift = np.array([[3., 21., 21.], [3., 21., 21.], [3., 21., 21.], [3., 21., 21.]])
    # Translate and scale the initial array
    array_b = 477.412 * array_a + shift
    # Define an arbitrary rotation transformation
    theta = 44.3 * pi / 5.7
    rot_array = np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])
    assert(abs(np.linalg.det(rot_array) - 1.) < 1.e-8)
    assert(abs(np.dot(rot_array, rot_array.T) - np.eye(3)) < 1.e-8).all()
    # Define the translated, scaled, and rotated initial array
    array_rotated = np.dot(array_b, rot_array)
    # Proceed with rotational-orthogonal procrustes analysis
    rot_ortho = RotationalOrthogonalProcrustes(array_a, array_rotated, translate_scale=True)
    r, a_transformed, error = rot_ortho.calculate()
    # r must be a rotation array
    assert(abs(np.linalg.det(r) - 1.) < 1.e-8)
    assert(abs(np.dot(r, r.T) - np.eye(3)) < 1.e-8).all()
    # The transformation should return zero error
    assert error < 1.e-10

# ----------------------------------------------------


def test_symmetric():
    """
    This test verifies that orthogonal procrustes analysis
    works with equivalent input arrays of different sizes (i.e. zero padded arrays)
    """
    # Define arbitrary array
    array = np.array([[1, 2, 4, 5], [5, 7, 3, 3], [1, 5, 1, 9], [1, 5, 2, 7], [5, 7, 9, 0]])
    sym_part = np.array([[1, 7, 4, 9]])
    sym_array = np.dot(sym_part.T, sym_part)
    assert(sym_array == sym_array.T).all()
    # Compute the symetrically transformed original array
    array_symmetrically_transformed = np.dot(array, sym_array)
    # Create (arbitrary) pads to add onto the permuted input array, array_permuted
    m, n = array_symmetrically_transformed.shape
    arb_pad_col = 2
    arb_pad_row = 8
    pad_vertical = np.zeros((m, arb_pad_col))
    pad_horizontal = np.zeros((arb_pad_row, n+arb_pad_col))
    array_symmetrically_transformed = np.concatenate((array_symmetrically_transformed, pad_vertical), axis=1)
    array_symmetrically_transformed = np.concatenate((array_symmetrically_transformed, pad_horizontal), axis=0)
    # Proceed with symmetric procrustes analysis
    # calculate the symmetric array
    symm = SymmetricProcrustes(array, array_symmetrically_transformed)
    symmetric_transformation, a_transformed_predicted, error = symm.calculate()
    assert((symmetric_transformation - symmetric_transformation.T - np.zeros(symmetric_transformation.shape)) < 1.e-10).all()
    assert error < 1.e-8
    """
    This test verifies that permutation procrustes analysis
    returns an error when rank(array_b) >= rank(array_a). We will show the case for
    when the input arrays are identical
    """
    """
    # Define an symmetric arbitrary array
    sym_array1 = np.array([[60,  85,  86], [85, 151, 153], [86, 153, 158]])
    assert(sym_array1 == sym_array1.T).all()
    # Match arbitrary array (above) to itself
    sym_array2 = np.array([[60,  85,  86], [85, 151, 153], [86, 153, 158]])
    assert(sym_array2 == sym_array2.T).all()
    # Proceed with symmmetric procrustes analysis
    symm = SymmetricProcrustes(sym_array1, sym_array2)
    symmetric_transformation, a_transformed_predicted, error = symm.calculate()
    # The symmetric procrustes analysis should return an error since rank(array_b) = >= rank(array_a)
    """

    """
    This test verifies that symmetric procrustes analysis is capable of matching an input array
    to itself after it undergoes translation, scaling, and symmetric transformations
    """
    # Define an arbitrary array.
    array_a = np.array([[5, 2, 8], [2, 2, 3], [1, 5, 6], [7, 3, 2]])
    # Define an arbitrary translation
    shift = np.array([[9., 4., 3.], [9., 4., 3.], [9., 4., 3.], [9., 4., 3.]])
    # Translate and scale the initial array
    array_b = 614.5 * array_a + shift
    sym_part = np.array([[1, 4, 9]])
    sym_array = np.dot(sym_part.T, sym_part)
    assert(sym_array == sym_array.T).all()
    # Compute the symetrically transformed original array
    array_symmetrically_transformed = np.dot(array_b, sym_array)
    # Proceed with symmetric-procrustes analysis
    symm = SymmetricProcrustes(array_a, array_symmetrically_transformed, translate_scale=True)
    x, a_transformed, error = symm.calculate()
    assert((x - x.T - np.zeros(x.shape)) < 1.e-10).all()
    # The transformation should return zero error
    assert error < 1.e-8

# ----------------------------------------------------------


def test_two_sided_orthogonal():
    """
    This test verifies that orthogonal procrustes analysis
    works with equivalent input arrays of different sizes (i.e. zero padded arrays)
    """
    # Define arbitrary array
    array = np.array([[1., 4., 6.], [6., 1., 4.], [7., 8., 1.]])
    # Define two arbitrary rotation transformations
    theta1 = pi / 5.
    rot_array1 = np.array([[cos(theta1), -sin(theta1), 0.], [sin(theta1), cos(theta1), 0.], [0., 0., 1.]])
    assert(abs(np.dot(rot_array1, rot_array1.T) - np.eye(3)) < 1.e-8).all()
    assert(abs(np.linalg.det(rot_array1) - 1.) < 1.e-10)
    theta2 = 12. * pi / 9.
    rot_array2 = np.array([[cos(theta2), -sin(theta2), 0.], [sin(theta2), cos(theta2), 0.], [0., 0., 1.]])
    assert(abs(np.dot(rot_array2, rot_array2.T) - np.eye(3)) < 1.e-8).all()
    assert(abs(np.linalg.det(rot_array2) - 1) < 1.e-10)
    # Define two arbitrary reflection transformations
    refl_array1 = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    assert(abs(np.linalg.det(refl_array1) + 1) < 1.e-8)
    assert(abs(np.dot(refl_array1, refl_array1.T) - np.eye(3)) < 1.e-8).all()
    refl_array2 = 1./3. * np.array([[1., -2., -2.], [-2., 1., -2.], [-2., -2., 1.]])
    assert(abs(np.linalg.det(refl_array2) + 1) < 1.e-8)
    assert(abs(np.dot(refl_array2, refl_array2.T) - np.eye(3)) < 1.e-8).all()
    # Define two orthogonal transformations
    ortho_trans1 = np.dot(refl_array1, rot_array1)
    assert(abs(np.dot(ortho_trans1, ortho_trans1.T) - np.eye(3)) < 1.e-10).all()
    ortho_trans2 = np.dot(rot_array2, refl_array2)
    assert(abs(np.dot(ortho_trans2, ortho_trans2.T) - np.eye(3)) < 1.e-10).all()
    # Define the two-sided orthogonally transformed original array
    array_ortho_transformed = np.dot(np.dot(ortho_trans1.T, array), ortho_trans2)
    # Create (arbitrary) pads to add onto the two-sided orthogonally transformed input array, array_ortho_transformed
    m, n = array_ortho_transformed.shape
    arb_pad_col = 8
    arb_pad_row = 4
    pad_vertical = np.zeros((m, arb_pad_col))
    pad_horizontal = np.zeros((arb_pad_row, n+arb_pad_col))
    array_ortho_transformed = np.concatenate((array_ortho_transformed, pad_vertical), axis=1)
    array_ortho_transformed = np.concatenate((array_ortho_transformed, pad_horizontal), axis=0)
    # Proceed with two-sided procrustes analysis
    twosided_ortho = TwoSidedOrthogonalProcrustes(array, array_ortho_transformed)
    u1, u2, array_transformed, error = twosided_ortho.calculate()
    assert((np.dot(u1, u1.T) - np.eye(3)) < 1.e-10).all()
    assert((np.dot(u2, u2.T) - np.eye(3)) < 1.e-10).all()
    # The transformation should return zero error
    print error
    # assert error < 1.e-10

    """
    This test verifies that two-sided orthogonal procrustes analysis
    works when the input arrays are identical
    """
    # Define an arbitrary array
    array_a = np.array([[2, 5, 4, 1], [5, 3, 1, 2], [8, 9, 1, 0], [1, 5, 6, 7]])
    # Match arbitrary array (above) to itself
    array_b = np.array([[2, 5, 4, 1], [5, 3, 1, 2], [8, 9, 1, 0], [1, 5, 6, 7]])
    assert(abs(array_a - array_b) < 1.e-8).all()
    # Proceed with two-sided orthogonal procrustes analysis
    twosided_ortho = TwoSidedOrthogonalProcrustes(array_a, array_b)
    u1, u2, array_transformed, error = twosided_ortho.calculate()
    assert((np.dot(u1, u1.T) - np.eye(4)) < 1.e-10).all()
    assert((np.dot(u2, u2.T) - np.eye(4)) < 1.e-10).all()
    # The transformation should return zero error
    assert error < 1.e-8
    """
    This test verifies that the two-sided orthogonal procrustes analysis is capable of matching an input array
    to itself after it undergoes translation, scaling, and orthogonal transformations
    """
    # Define an arbitrary array.
    array_a = np.array([[1, 3, 5], [3, 5, 7], [8, 11, 15]])
    # Define an arbitrary translation
    shift = np.array([[16., 41., 33.], [16., 41., 33.], [16., 41., 33.]])
    # Translate and scale the initial array
    array_b = 23.5 * array_a + shift
    # Define an arbitrary rotation transformations
    theta = 1.8 * pi / 34.
    rot_array = np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])
    assert(abs(np.dot(rot_array1, rot_array1.T) - np.eye(3)) < 1.e-8).all()
    assert(abs(np.linalg.det(rot_array1) - 1) < 1.e-10)
    # Define an arbitrary reflection transformations
    refl_array = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    assert(abs(np.linalg.det(refl_array1) + 1) < 1.e-8)
    assert(abs(np.dot(refl_array1, refl_array1.T) - np.eye(3)) < 1.e-8).all()
    # Compute the two-sided orthogonally transformed original array
    array_twosided_ortho_transf = np.dot(np.dot(refl_array, array_b), rot_array)
    # Proceed with two-sided orthogonal procrustes analysis
    twosided_ortho = TwoSidedOrthogonalProcrustes(array_a, array_twosided_ortho_transf, translate_scale=True)
    u1, u2, a_transformed, error = twosided_ortho.calculate()
    assert((np.dot(u1, u1.T) - np.eye(3)) < 1.e-10).all()
    assert((np.dot(u2, u2.T) - np.eye(3)) < 1.e-10).all()
    # The transformation should return zero error
    assert error < 1.e-8

# -----------------------------------------------------------


def test_two_sided_orthogonal_single_transformation():
    """
    Require A and A0 to be symmetric...
    """
    # Define arbitrary symmetric array
    array = np.array([[5, 2, 1], [4, 6, 1], [1, 6, 3]])
    sym_array = np.dot(array, array.T)
    # Define an arbitrary rotation transformation
    theta = 16. * pi / 5.
    rot_array = np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])
    assert(abs(np.linalg.det(rot_array) - 1) < 1.e-8)
    assert((np.dot(rot_array, rot_array.T) - np.eye(3)) < 1.e-8).all()
    # Define an arbitrary reflection transformation
    refl_array = 1./3 * np.array([[1, -2, -2], [-2, 1, -2], [-2, -2, 1]])
    assert((np.dot(refl_array, refl_array.T) - np.eye(3)) < 1.e-8).all()
    assert(abs(np.linalg.det(refl_array) + 1) < 1.e-8)
    # Define the orthogonal transformation
    orth_transf = np.dot(rot_array, refl_array)
    assert(abs(np.dot(orth_transf, orth_transf.T) - np.eye(3)) < 1.e-8).all()
    # Define the two-sided orthogonal transformation of the original array
    array_singleortho_transf = np.dot(np.dot(orth_transf.T, sym_array), orth_transf)
    assert(abs(array_singleortho_transf - array_singleortho_transf.T) < 1.e-10).all()
    # Create (arbitrary) pads to add onto the two-sided single transformation
    # orthogonally transformed input array, array_singleortho_transf
    m, n = array_singleortho_transf.shape
    arb_pad_col = 7
    arb_pad_row = 4
    pad_vertical = np.zeros((m, arb_pad_col))
    pad_horizontal = np.zeros((arb_pad_row, n+arb_pad_col))
    array_singleortho_transf = np.concatenate((array_singleortho_transf, pad_vertical), axis=1)
    array_singleortho_transf = np.concatenate((array_singleortho_transf, pad_horizontal), axis=0)
    # Proceed with two-sided single transformation procrustes analysis
    twosided_single_ortho = TwoSidedOrthogonalSingleTransformationProcrustes(sym_array, array_singleortho_transf)
    u_approx, u_exact, array_transformed_approx, array_transformed_exact, error_approx, error_best\
        = twosided_single_ortho.calculate(return_u_approx=True, return_u_best=True)
    assert(abs(np.dot(u_approx, u_approx.T) - np.eye(3)) < 1.e-8).all()
    assert(abs(np.dot(u_exact, u_exact.T) - np.eye(3)) < 1.e-8).all()
    assert error_best < 1.e-10

    """
    This test verifies that two-sided single transformation orthogonal procrustes analysis
    works when the input arrays are identical
    """
    # Define an arbitrary symmetric array
    sym_part = np.array([[2, 5, 4, 1], [5, 3, 1, 2], [8, 9, 1, 0], [1, 5, 6, 7]])
    sym_array_a = np.dot(sym_array, sym_array.T)
    # Match arbitrary array (above) to itself
    sym_array_b = sym_array_a
    assert(abs(sym_array_a - sym_array_b) < 1.e-8).all()
    # Proceed with two-sided single transformation orthogonal procrustes analysis
    twosided_single_ortho = TwoSidedOrthogonalSingleTransformationProcrustes(sym_array_a, sym_array_b)
    u_approx, u_exact, array_transformed_approx, array_transformed_exact, error_approx, error_best\
        = twosided_single_ortho.calculate(return_u_approx=True, return_u_best=True)
    assert(abs(np.dot(u_approx, u_approx.T) - np.eye(3)) < 1.e-8).all()
    assert(abs(np.dot(u_exact, u_exact.T) - np.eye(3)) < 1.e-8).all()
    assert error_best < 1.e-10


    """
    This test verifies that the two-sided single transformation orthogonal procrustes analysis is capable of matching an
     input array to itself after it undergoes translation, scaling, and orthogonal transformations
    """
    # Define an arbitrary symmetric array.
    sym_part = np.array([[1., 6., 7.], [1., 5., 6.], [6., 9., 4.]])
    array_a = np.dot(sym_part, sym_part.T)
    assert(abs(array_a - array_a.T) < 1.e-10).all()
    # Define an arbitrary translation. The shift must preserve the symmetry of the array
    # (i.e. the shift must too be symmetric)
    sym_shift = np.array([[6.7, 6.7, 6.7], [6.7, 6.7, 6.7], [6.7, 6.7, 6.7]])
    assert(abs(sym_shift - sym_shift.T) < 1.e-10).all()
    # Translate and scale the initial array
    array_b = 6.9 * array_a + sym_shift
    # Define an arbitrary rotation transformations
    theta = 16.9 * pi / 26.3
    rot_array = np.array([[cos(theta), -sin(theta), 0.], [sin(theta), cos(theta), 0.], [0., 0., 1.]])
    assert(abs(np.dot(rot_array, rot_array.T) - np.eye(3)) < 1.e-8).all()
    assert(abs(np.linalg.det(rot_array) - 1.) < 1.e-10)
    # Define an arbitrary reflection transformations
    refl_array = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
    assert(abs(np.linalg.det(refl_array) + 1) < 1.e-8)
    assert(abs(np.dot(refl_array, refl_array.T) - np.eye(3)) < 1.e-8).all()
    # Define the single orthogonal transformation
    # single_ortho_transf = np.dot(rot_array, refl_array)
    single_ortho_transf = refl_array
    assert(abs(np.dot(single_ortho_transf, single_ortho_transf.T) - np.eye(3)) < 1.e-8).all()
    # Define the sinlge othogonal transformation of the original array
    array_singleortho_transf = np.dot(np.dot(single_ortho_transf.T, array_b), single_ortho_transf)
    assert(abs(array_singleortho_transf - array_singleortho_transf.T) < 1.e-10).all()
    # Proceed with two-sided single transformation orthogonal procrustes analysis
    twosided_single_ortho = TwoSidedOrthogonalSingleTransformationProcrustes(array_a, array_singleortho_transf, translate_scale=True)
    u_approx, u_exact, array_transformed_approx, array_transformed_exact, error_approx, error_best\
        = twosided_single_ortho.calculate(return_u_approx=True, return_u_best=True)
    assert(abs(np.dot(u_approx, u_approx.T) - np.eye(3)) < 1.e-8).all()
    assert(abs(np.dot(u_exact, u_exact.T) - np.eye(3)) < 1.e-8).all()
    print error_best


# ------------------------------------------------------------


def test_two_sided_permutation_single_transformation():
    """
    Require A and A0 to be symmetric...
    """
    """
    This test verifies that two-sided single transformation permutation procrustes analysis
    works with equivalent input arrays of different sizes (i.e. zero padded arrays)
    """
    # Define arbitrary array
    sym_part = np.array([[5., 2., 1.], [4., 6., 1.], [1., 6., 3.]])
    sym_array = np.dot(sym_part, sym_part.T)
    # Define an arbitrary permutation transformation
    perm_array = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
    assert(abs(np.linalg.det(perm_array)) - 1. < 1.e-8)
    assert(abs([x for x in perm_array.flatten().tolist() if x != 0] - np.ones(3)) < 1.e-8).all()
    # Define the permuted original array
    array_permuted = np.dot(np.dot(perm_array.T, sym_array), perm_array)
    # Create (arbitrary) pads to add onto the permuted input array, array_permuted
    m, n = array_permuted.shape
    arb_pad_col = 4
    arb_pad_row = 8
    pad_vertical = np.zeros((m, arb_pad_col))
    pad_horizontal = np.zeros((arb_pad_row, n+arb_pad_col))
    array_permuted = np.concatenate((array_permuted, pad_vertical), axis=1)
    array_permuted = np.concatenate((array_permuted, pad_horizontal), axis=0)
    # Proceed with permutation procrustes analysis
    twosided_single_perm = TwoSidedPermutationSingleTransformationProcrustes(sym_array, array_permuted)
    least_error_perm, least_error_array_transformed, min_error = twosided_single_perm.calculate()
    assert(abs(np.linalg.det(least_error_perm)) - 1. < 1.e-8)
    assert(abs([x for x in least_error_perm.flatten().tolist() if x != 0] - np.ones(3)) < 1.e-8).all()
    # Assert that the analysis returns zero error
    print min_error
    assert min_error < 1.e-10
    """
    This test verifies that permutation procrustes analysis
    works when the input arrays are identical
    """
    # If the input arrays are equivalent, the permutation transformation must be the identity
    # Define an arbitrary symmetric array
    sym_part = np.array([[12, 22, 63, 45, 7], [23, 54, 66, 63, 21]])
    array_a = np.dot(sym_part, sym_part.T)
    # Match arbitrary array (above) to itself
    array_b = array_a
    assert(abs(array_a - array_b) < 1.e-8).all()
    # Proceed with two sided single-transformation procrustes analysis
    twosided_single_perm = TwoSidedPermutationSingleTransformationProcrustes(array_a, array_b)
    least_error_perm, least_error_array_transformed, min_error = twosided_single_perm.calculate()
    # Perm_optimum must be a permutation array
    assert(abs(np.linalg.det(least_error_perm)) - 1. < 1.e-8)
    assert(abs([x for x in least_error_perm.flatten().tolist() if x != 0] - np.ones(2)) < 1.e-8).all()
    # The expected permutation-transformation is the 4x4 identity array
    expected = np.eye(2)
    # The transformation should return zero error
    assert min_error < 1.e-8
    # The transformation must be the 4x4 identity
    assert(abs(least_error_perm - expected) < 1.e-8).all()


    """
    This test verifies that permutation procrustes analysis is capable of matching an input array
    to itself after it undergoes translation, scaling, and permutation transformations
    """
    # Define arbitrary array
    sym_part = np.array([[5., 2., 1.], [4., 6., 1.], [1., 6., 3.]])
    sym_array = np.dot(sym_part, sym_part.T)
    # Define an arbitrary translation. The shift must preserve the symmetry of the array
    #  (i.e. the shift must too be symmetric)
    sym_shift = np.array([[3.14, 3.14, 3.14], [3.14, 3.14, 3.14], [3.14, 3.14, 3.14]])
    assert(abs(sym_shift - sym_shift.T) < 1.e-10).all()
    # Translate and scale the initial array
    array_b = 14.7 * sym_array + sym_shift
    # Define an arbitrary permutation transformation
    perm_array = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
    assert(abs(np.linalg.det(perm_array)) - 1. < 1.e-8)
    assert(abs([x for x in perm_array.flatten().tolist() if x != 0] - np.ones(3)) < 1.e-8).all()
    # Define the permuted original array
    array_permuted = np.dot(np.dot(perm_array.T, array_b), perm_array)
    # Proceed with permutation procrustes analysis
    twosided_single_perm = TwoSidedPermutationSingleTransformationProcrustes(sym_array, array_permuted, translate_scale=True)
    least_error_perm, least_error_array_transformed, min_error = twosided_single_perm.calculate()
    assert(abs(np.linalg.det(least_error_perm)) - 1. < 1.e-8)
    assert(abs([x for x in least_error_perm.flatten().tolist() if x != 0] - np.ones(3)) < 1.e-8).all()
    # Assert that the analysis returns zero error
    assert min_error < 1.e-10
