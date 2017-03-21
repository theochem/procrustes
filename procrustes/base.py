__author__ = 'Jonny'
# -*- coding: utf-8 -*-
"""Base class for Procrustes Analysis"""

import numpy as np


class Procrustes(object):
    """
    This class provides a base class for all types of Procrustes analysis.
    """
    def __init__(self, array_a, array_b, translate_scale=False, translate=False, scale=False):
        """
        Parameters
        ----------
        array_a : ndarray
            The 2D array
        array_b : ndarray
            The 2D array
        translate_scale : bool
            Set to True to translate and scale the input arrays; default=False
        translate : bool
            Set to True to translate the input arrays; default=False

        scale : bool
            Set to True to scale the input arrays; default=False
            Note: Scaling is meaningful, if preceded with translating. So, the code
            sets translate=True, if scale=True and translate=False are specified.
        """
        # Check type and dimension of arrays
        if not isinstance(array_a, np.ndarray) or not isinstance(array_b, np.ndarray):
            raise ValueError('The array_a and array_b should be of numpy.ndarray type.')

        if array_a.ndim != 2 or array_b.ndim != 2:
            raise ValueError('The array_a and array_b should be 2D arrays.')

        if (translate is False) and (scale is True):
            print 'Translate has been set to True because scaling without translating does not make sense!'
            translate_scale = True
            translate = False
            scale = False

        # Remove any zero-padding that may already be attached to arrays.
        # This is important if translating is to be done.
        # Hiding and zero padding
        array_a = self.hide_zero_padding_array(array_a)
        array_b = self.hide_zero_padding_array(array_b)

        # Translate and scale arrays
        if translate_scale:
            print 'Input arrays have been centred and scaled.'
            array_a = self.translate_scale_array(array_a)
            array_b = self.translate_scale_array(array_b)

        if translate:
            print 'Input arrays have been translated.'
            array_a = self.translate_array(array_a)
            array_b = self.translate_array(array_b)
        if scale:
            print 'Input arrays have been scaled.'
            array_a = self.scale_array(array_a)
            array_b = self.scale_array(array_b)

        # Match the number of rows of arrays
        print 'The input 2D arrays are {0} and {1}.\n'.format(array_a.shape, array_b.shape)
        if array_a.shape[0] != array_b.shape[0]:
            print 'The general Procrustes analysis requires two 2D arrays with the same number of rows,',
            print 'so the array with the smaller number of rows will automatically be padded with zero rows.'
            array_a, array_b = self.zero_padding(array_a, array_b, row=True, column=False)
            if array_a.shape[1] == array_b.shape[1]:
                print 'Tip: The 2D input arrays have the same number of columns, so'
                print 'the Procrustes analysis is doable (without zero padding) on the transposed matrices.'
            array_a, array_b = self.zero_padding(array_a, array_b, row=False, column=True)
        else:
            print 'The number of rows are the same, the analysis will proceed'
        # Update self.array_a/b
        self.array_a = array_a
        self.array_b = array_b

    def zero_padding(self, x1, x2, row=False, column=False, square=False):
        """
        Match the number of rows and/or columns of arrays x1 and x2, by
        padding zero rows and/or columns to the array with the smaller dimensions.

        Parameters
        ----------
        x1 : ndarray
            The 2D array
        x2 : ndarray
            The 2D array
        row : bool
            Set to True to match the number of rows by zero-padding; default=False.
        column : bool
            Set to True to match the number of columns by zero-padding; default=False.
        square: bool
            Set to True to zero pad the input arrays such that the inputs become square
            arrays of the same size
        Returns
        -------
        If row = True and Column = False:

             Returns the input arrays, x1 and x2, where the array with the fewer number
             of rows has been padded with zeros to match the number of rows of the other array

        if row = False and column = True

             Returns the input arrays, x1 and x2, where the array with the fewer number
             of columns has been padded with zeros to match the number of columns of the other array

        if row = True and column = True

             Returns the input arrays, x1 and x2, where the array with the fewer rows has been row-padded
             with zeros, and the array with the fewer number of columns has been column-padded with zeros
        """
        # CHECK: Make sure there is no in place operations on x1 and x2, otherwise the tests fail.
        # Check the inputs
        assert isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray)
        assert x1.ndim == 2 and x2.ndim == 2
        if square:
            if (x1.shape == x2.shape) and (x1.shape[0] == x1.shape[1]):
                print "The arrays are already square and of the same size."
            n_1, m_1 = x1.shape
            n_2, m_2 = x2.shape
            new_dimension = max(n_1, n_2, m_1, m_2)
            # Row pad
            if n_1 < new_dimension:
                pad = np.zeros((new_dimension - n_1, x1.shape[1]))
                x1 = np.concatenate((x1, pad), axis=0)
            if n_2 < new_dimension:
                pad = np.zeros((new_dimension - n_2, x2.shape[1]))
                x2 = np.concatenate((x2, pad), axis=0)
            # Column Pad
            if m_1 < new_dimension:
                pad = np.zeros((new_dimension, new_dimension - m_1))
                x1 = np.concatenate((x1, pad), axis=1)
            if m_2 < new_dimension:
                pad = np.zeros((new_dimension, new_dimension - m_2))
                x2 = np.concatenate((x2, pad), axis=1)

        if x1.shape == x2.shape:
            pass
        else:
            if row and column:
                if x1.shape[0] < x2.shape[0]:
                    # padding x1 with zero rows
                    pad = np.zeros((x2.shape[0] - x1.shape[0], x1.shape[1]))
                    x1 = np.concatenate((x1, pad), axis=0)
                elif x1.shape[0] > x2.shape[0]:
                    # padding x2 with zero rows
                    pad = np.zeros((x1.shape[0] - x2.shape[0], x2.shape[1]))
                    x2 = np.concatenate((x2, pad), axis=0)

                if x1.shape[1] < x2.shape[1]:
                    # padding x1 with zero columns
                    pad = np.zeros((x1.shape[0], x2.shape[1] - x1.shape[1]))
                    x1 = np.concatenate((x1, pad), axis=1)
                elif x1.shape[1] > x2.shape[1]:
                    # padding x2 with zero columns
                    pad = np.zeros((x2.shape[0], x1.shape[1] - x2.shape[1]))
                    x2 = np.concatenate((x2, pad), axis=1)

            elif row:
                if x1.shape[0] < x2.shape[0]:
                    # padding x1 with zero rows
                    pad = np.zeros((x2.shape[0] - x1.shape[0], x1.shape[1]))
                    x1 = np.concatenate((x1, pad), axis=0)
                elif x1.shape[0] > x2.shape[0]:
                    # padding x2 with zero rows
                    pad = np.zeros((x1.shape[0] - x2.shape[0], x2.shape[1]))
                    x2 = np.concatenate((x2, pad), axis=0)

            elif column:
                if x1.shape[1] < x2.shape[1]:
                    # padding x1 with zero columns
                    pad = np.zeros((x1.shape[0], x2.shape[1] - x1.shape[1]))
                    x1 = np.concatenate((x1, pad), axis=1)
                elif x1.shape[1] > x2.shape[1]:
                    # padding x2 with zero columns
                    pad = np.zeros((x2.shape[0], x1.shape[1] - x2.shape[1]))
                    x2 = np.concatenate((x2, pad), axis=1)

            else:
                raise ValueError('Either row or column arguments should be set to True for the padding to be meaningful.')

        return x1, x2

    def translate_array(self, array):
        """
         Remove translational component of array by translating the object
          such that its centroid is centred about the origin

        Parameters
        ----------
        array : ndarray
            A 2D array

        Returns
        ----------
        Returns the input array such that its centroid is centred about the origin
         """
        mean_array_cols = array.mean(0)
        centred_array = array - mean_array_cols
        return centred_array

    def scale_array(self, array):
        """
        Uniform scaling of the input array. Scales the array such that the standard deviation
        of each point (row) of the array is equal to unity after scaling

        Parameters
        ----------
        array : ndarray
            A 2D array

        Returns
        ----------
        Returns the input array such that the standard deviation of each point (row) of the input matrix
        is scaled to unity
         """
        # Centre array by normalizing by the Frobenius norm
        squared_frobenius_norm = (array**2.).sum()  # Calculate Frobenius norm
        frobenius_norm = np.sqrt(squared_frobenius_norm)
        # Scale array to lie on unit sphere
        array = array / frobenius_norm
        return array
    # ---------------------------------------------------

    def translate_scale_array(self, array):
        """
           Parameters
        ----------
        array : ndarray
            An (m x n) 2D array representing the array to be translated (to origin) and scaled to lie on
            the unit sphere in R^(n x m)

        Returns
        ------------
        Returns the original array centred to the origin and scaled via Frobenius normalization
        """
        array_translated = self.translate_array(array)
        array_translated_scaled = self.scale_array(array_translated)
        return array_translated_scaled



    # ---------------------------------------------------

    def single_sided_procrustes_error(self, array_a, array_b, t_array):
        """ Returns the error for all single-sided procrustes problems

        min { (([array_a]*[t_array] - [array_b]).T)*([array_a]*[t_array] - [array_b]) }
            : t_array satisfies some condition


        Parameters
        ----------

        array_a : ndarray
            A 2D array representing the array to be transformed (as close as possible to array_b)

        array_b : ndarray
            A 2D array representing the reference array

        t_array: ndarray
           A 2D array representing the 'optimum' transformation

        Returns
        ------------
        Returns the error given by the optimum choice of t_array

        """

        at = np.dot(array_a, t_array)
        error = np.trace(np.dot((at-array_b).T, at-array_b))
        return error

    def double_sided_procrustes_error(self, array_a, array_b, t_array1, t_array2):
        """
        Returns the error for all double-sided procrustes problems

        min { ([t_array1].T*[array_a]*[t_array2] - [array_b]).T) * ([t_array1].T*[array_a]*[t_array2] - [array_b])) }
            : t_array1, t_array2 satisfies some condition

        Parameters
        ----------

        array_a : ndarray
            A 2D array representing the array to be transformed (as close as possible to array_b)

        array_b : ndarray
            A 2D array representing the reference array

        t_array: ndarray
           A 2D array representing the 1st 'optimum' transformation
           must satisfy some criteria
        t_array2: ndarray
           A 2D array representing the 2nd 'optimum' transformation
           must satisfy some criteria

        Returns
        ------------
        Returns the error given by the optimum choice of t_array1 and t_array2
        """
        t_trans_a_t = np.dot(np.dot(t_array1.T, array_a), t_array2)
        error = np.trace(np.dot((t_trans_a_t - array_b).T, (t_trans_a_t-array_b)))
        return error


    def singular_value_decomposition(self, array):
        """
        Singular Value Decomposition of an array
        Decomposes an mxn array A such that A = U*S*V.T

        Parameters
        -----------

        array: ndarray
        A 2D array who's singular value deocmposition is to be calculated

        Returns
        --------------
        u = a unitary matrix
        s = diagonal matrix of singular values, sorting from greatest to least
        v = a unitary matrix
        """
        return np.linalg.svd(array)

    def eigenvalue_decomposition(self, array, two_sided_single=False):
        """
        Computes the eigenvalue decomposition of array
        Decomposes array A such that A = U*S*U.T

        Parameters
        ------------
        array: ndarray
           A 2D array who's eigenvalue decomposition is to be calculated

        two_sided_single : bool
            Set to True when dealing with two-sided single transformation procrustes problems,
            such as two_sided single transformation orthogonal / permutation. When true, array of
            eigenvectors is rearranged according to rows rather than columns, allowing the analysis
            to proceed.

        Returns
        ------------
        s = 1D array of eigenvalues of array, sorted from greatest to least
        v = 2D array of eigenvectors of array, sorted according to S
        """
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

    def hide_zero_padding_array(self, array, tol=1.e-8):
        """
        :param array: An array that may or may not contain zero-padding, where all important
        information is contained in upper-left block of array
        :param tol: Tolerance for which is sum(row/column) < tol, then the row/col will be removed
        :return:Returns the input array with any zero-padding removed.
        All zero padding is assumed to be such that all relevant information is contained
        within upper-left most array block
        """
        m, n = array.shape
        for i in range(m)[::-1]:
            if sum(np.absolute(array[i, :])) < tol:  # If the sum of array elements across given row is less than tol..
                array = np.delete(array, i, 0)
        for j in range(n)[::-1]:
            if sum(np.absolute(array[:, j])) < tol:  # If the sum of array elements down given col is less than tol..
                array = np.delete(array, j, 1)
        return array

    def is_diagonalizable(self, array):
        """
        :param array: A square array for which the diagonalizability is of interest
        :return: Returns a boolean value dictating whether or not the input array is diagonalizable
        """
        array = self.hide_zero_padding_array(array)
        m, n = array.shape
        if m != n:
            raise ValueError('The input array must be square.')
        u, s, vt = np.linalg.svd(array)
        rank_u = np.linalg.matrix_rank(array)
        if rank_u != n:
            # The eigenvectors cannot span the dimension of the vector space
            # The array cannot be diagonalizable
            return False
        else:
            # The eigenvectors span the dimension of the vector space and therefore the array is diagonalizable
            return True






