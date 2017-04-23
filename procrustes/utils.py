"""
Utility Module.
"""


import numpy as np


def zero_padding(array_a, array_b, mode):
    r"""
    Return arrays padded with rowm and/or columns of zero.

    Parameters
    ----------
    array_a : ndarray
        The 2D array :math:`\mathbf{A}_{n_1 \times m_1}`.
    array_b : ndarray
        The 2D array :math:`\mathbf{B}_{n_2 \times m_2}`.
    mode : str
        One of the following values specifying how to padd arrays:

        **'row'**
             The array with fewer rows is padded with zero rows so that both have the same
             number of rows.
        **'col'**
             The array with fewer columns is padded with zero columns so that both have the
             same number of columns.
        **'row-col'**
             The array with fewer rows is padded with zero rows, and the array with fewer
             columns is padded with zero columns, so that both have the same dimensions.
             This does not necessarily result in square arrays.
        'square'
             The arrays are padded with zero rows and zero columns so that they are both
             squared arrays. The dimension of square array is specified based on the highest
             dimension, i.e. :math:`\text{max}(n_1, m_1, n_2, m_2)`.

    Returns
    -------
    ndarray, ndarray
        Padded array_a and array_b arrays.
    """
    # sanity checks
    if not isinstance(array_a, np.ndarray) or not isinstance(array_b, np.ndarray):
        raise ValueError('Arguments array_a & array_b should be numpy arrays.')
    if array_a.ndim != 2 or array_b.ndim != 2:
        raise ValueError('Arguments array_a & array_b should be 2D arrays.')

    if array_a.shape == array_b.shape:
        # special case of square arrays, mode is set to None so that array_a & array_b are returned.
        mode = None

    if mode == 'square':
        # calculate desired dimension of square array
        (n1, m1), (n2, m2) = array_a.shape, array_b.shape
        dim = max(n1, n2, m1, m2)
        # padding rows to have both arrays have dim rows
        if n1 < dim:
            array_a = np.pad(array_a, [[0, dim - n1], [0, 0]], 'constant', constant_values=0)
        if n2 < dim:
            array_b = np.pad(array_b, [[0, dim - n2], [0, 0]], 'constant', constant_values=0)
        # padding columns to have both arrays have dim columns
        if m1 < dim:
            array_a = np.pad(array_a, [[0, 0], [0, dim - m1]], 'constant', constant_values=0)
        if m2 < dim:
            array_b = np.pad(array_b, [[0, 0], [0, dim - m2]], 'constant', constant_values=0)

    if mode == 'row' or mode == 'row-col':
        # padding rows to have both arrays have the same number of rows
        diff = array_a.shape[0] - array_b.shape[0]
        if diff < 0:
            array_a = np.pad(array_a, [[0, -diff], [0, 0]], 'constant', constant_values=0)
        else:
            array_b = np.pad(array_b, [[0, diff], [0, 0]], 'constant', constant_values=0)

    if mode == 'col' or mode == 'row-col':
        # padding columns to have both arrays have the same number of columns
        diff = array_a.shape[1] - array_b.shape[1]
        if diff < 0:
            array_a = np.pad(array_a, [[0, 0], [0, -diff]], 'constant', constant_values=0)
        else:
            array_b = np.pad(array_b, [[0, 0], [0, diff]], 'constant', constant_values=0)

    return array_a, array_b


def translate_array(array_a, array_b=None):
    """
    Return translated array_a and translation vector.

    Parameters
    ----------
    array_a : ndarray
        The 2D array to translate.
    array_b : ndarray
        The 2D array to translate array_a based on (default=None).

    Returns
    -------
    ndarray, ndarray
        If array_b is None, array_a is translated to origin using its centroid.
        If array_b is given, array_a is translated to centroid of array_b (the centroid of
        translated array_a will centroid with the centroid array_b).
     """
    # The mean is strongly affected by outliers and is not a robust estimator for central location
    # see https://docs.python.org/3.6/library/statistics.html?highlight=mean#statistics.mean
    centroid = compute_centroid(array_a)
    if array_b is not None:
        # translation vector to b centroid
        centroid -= compute_centroid(array_b)
    return array_a - centroid, -centroid


def scale_array(array_a, array_b=None):
    """
    Return scaled array_a and scaling vector.

    Parameters
    ----------
    array_a : ndarray
        The 2D array to scale
    array_b : ndarray
        The 2D array to scale array_a based on (default=None).

    Returns
    -------
    ndarray, ndarray
        If array_b is None, array_a is scaled to match norm of unit sphere using array_a's
        Frobenius norm.
        If array_b is given, array_a is scaled to match array_b's norm (the norm of array_a
        will be equal norm of array_b).
    """
    # scaling factor to match unit sphere
    scale = 1. / frobenius_norm(array_a)
    if array_b is not None:
        # scaling factor to match array_b norm
        scale *= frobenius_norm(array_b)
    return array_a * scale, scale


def singular_value_decomposition(array):
    """
    Return Singular Value Decomposition of an array.

    Decomposes an math:`m \times n` array math:`A` such that math:`A = U*S*V.T`

    Parameters
    -----------

    array: ndarray
    A 2D array who's singular value decomposition is to be calculated.

    Returns
    --------------
    u = a unitary matrix.
    s = diagonal matrix of singular values, sorting from greatest to least.
    v = a unitary matrix.
    """
    return np.linalg.svd(array)


def eigenvalue_decomposition(array, two_sided_single=False):
    """
    Computes the eigenvalue decomposition of array
    Decomposes array A such that A = U*S*U.T

    Parameters
    ------------
    array: ndarray
       A 2D array who's eigenvalue decomposition is to be calculated.

    two_sided_single : bool
        Set to True when dealing with two-sided single transformation procrustes problems,
        such as two_sided single transformation orthogonal / permutation. When true, array of
        eigenvectors is rearranged according to rows rather than columns, allowing the analysis
        to proceed.

    Returns
    ------------
    s = 1D array of the eigenvalues of array, sorted from greatest to least.
    v = 2D array of eigenvectors of array, sorted according to S.
    """
    # Test whether eigenvalue decomposition is possible on the input array
    if is_diagonalizable(array) is False:
        raise ValueError('The input array is not diagonalizable. The analysis cannot continue.')

    # Find eigenvalues and array of eigenvectors
    s, v = np.linalg.eigh(array)
    # Sort the eigenvalues from greatest to least
    idx = s.argsort()[::-1]

    s = s[idx]
    if two_sided_single:
        # For the two-sided single-transformation problems, we permute rows by idx
        v = v[idx]
    else:
        # For all other given problems, we permute columns by idx
        v = v[:, idx]

    return s, v


def hide_zero_padding(array, tol=1.e-8):
    """
    Removes any zero-padding that may be on the array.

    Parameters
    ------------
    array: An array that may or may not contain zero-padding, where all important information
    is contained in upper-left block of array.

    tol: Tolerance for which is sum(row/column) < tol, then the row/col will be removed.

    Returns
    ----------
    Returns the input array with any zero-padding removed.
    All zero padding is assumed to be such that all relevant information is contained
    within upper-left most array block
    """
    m, n = array.shape
    for i in range(m)[::-1]:
        # If the sum of array elements across given row is less than tol..
        if sum(np.absolute(array[i, :])) < tol:
            array = np.delete(array, i, 0)

    # If the sum of array elements down given col is less than tol..
    for j in range(n)[::-1]:
        if sum(np.absolute(array[:, j])) < tol:
            array = np.delete(array, j, 1)
    return array


def is_diagonalizable(array):
    """
    Check if the given array is diagonalizable or not.

    Parameters
    ------------
    array: A square array for which the diagonalizability is of interest

    Returns
    ---------
    Returns a boolean value dictating whether or not the input array is diagonalizable
    """
    array = hide_zero_padding(array)
    m, n = array.shape
    if m != n:
        raise ValueError('The input array must be square.')
    u, s, vt = np.linalg.svd(array)
    rank_u = np.linalg.matrix_rank(u)
    rank_array = np.linalg.matrix_rank(array)
    if rank_u != rank_array:
        # The eigenvectors cannot span the dimension of the vector space
        # The array cannot be diagonalizable
        return False
    else:
        # The eigenvectors span the dimension of the vector space and therefore
        # the array is diagonalizable
        return True


def compute_centroid(array):
    """
    """
    centroid = array.mean(0)
    return centroid


def frobenius_norm(array):
    """Return the Forbenius norm of array."""
    return np.sqrt((array ** 2.).sum())
