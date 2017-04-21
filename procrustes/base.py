"""
Base Procrustes Module.
"""


import numpy as np
from utils import zero_padding, hide_zero_padding, translate_scale_array


class Procrustes(object):
    """
    Base Procrustes Class.
    """

    def __init__(self, array_a, array_b, translate=False, scale=False):
        """
        Parameters
        ----------
        array_a : ndarray
            A 2D array
        array_b : ndarray
            A 2D array
        translate : bool
            Set to True to translate the input arrays; default=False
        scale : bool
            Set to True to scale the input arrays; default=False
            Note: Scaling is meaningful only if preceded with translation.
            So, the code sets translate=True if scale=True and translate=False.
        """
        translate_scale = False
        # Check type and dimension of arrays
        if not isinstance(array_a, np.ndarray) or not isinstance(array_b, np.ndarray):
            raise ValueError('The array_a and array_b should be of numpy.ndarray type.')

        if array_a.ndim != 2 or array_b.ndim != 2:
            raise ValueError('The array_a and array_b should be 2D arrays.')

        if (translate is False) and (scale is True) or (translate is True) and (scale is True):
            print 'Translate has been set to True by default since scaling without translating is useless in most ' \
                  'procrustes problems.'
            translate_scale = True
            translate = False
            scale = False

        # Remove any zero-padding that may already be attached to arrays.
        # This is important if translating is to be done.
        # Hiding zero padding
        array_a = hide_zero_padding(array_a)
        array_b = hide_zero_padding(array_b)

        # Initialize the translate_and_or_scale list to contain no entries
        self.translate_and_or_scale = []
        # Translate and scale arrays
        if translate_scale is True:
            print 'array_a has been translated (to array_b\'s centroid) and scaled (to array_b\'s scaling). '
            array_a, translate_a_to_b, scaling_factor = translate_scale_array(array_a, array_b)
            array_b = array_b
            # Translate vector and/or scaling factor
            self.translate_and_or_scale = [translate_a_to_b, scaling_factor]

        if translate is True:
            print 'Input arrays have been translated.'
            array_a, translate_a_to_origin = self.translate_array(array_a)
            array_b, translate_b_to_origin = self.translate_array(array_b)
            # Translate vector and/or scaling factor
            self.translate_and_or_scale = [translate_a_to_origin, translate_b_to_origin]
        if scale is True:
            print 'Input arrays have been scaled.'
            array_a, scale_a_to_unit_norm = self.scale_array(array_a)
            array_b, scale_b_to_unit_norm = self.scale_array(array_b)
            # Translate vector and/or scaling factor
            self.translate_and_or_scale = [scale_a_to_unit_norm, scale_b_to_unit_norm]

        # Zero pad arrays
        print 'The input 2D arrays are {0} and {1}.\n'.format(array_a.shape, array_b.shape)
        # expand the array with smaller number of rows to expand to the same row
        # number with the big one
        if array_a.shape[0] != array_b.shape[0]:
            print 'The general Procrustes analysis requires two 2D arrays with the same number of rows,',
            print 'so the array with the smaller number of rows will automatically be padded with zero rows.'
            array_a, array_b = zero_padding(array_a, array_b, mode='row')
            if array_a.shape[1] == array_b.shape[1]:
                print 'Tip: The 2D input arrays have the same number of columns, so'
                print 'the Procrustes analysis is doable (without zero padding) on the transposed matrices.'
            array_a, array_b = zero_padding(array_a, array_b, mode='col')
        # proceed once the number of rows are the same
        else:
            print 'The number of rows are the same, the analysis will proceed.'

        # Update self.array_a and self.array_b
        self.array_a = array_a
        self.array_b = array_b

    def single_sided_procrustes_error(self, array_a, array_b, t_array):
        r""" Returns the error for all single-sided procrustes problems

         .. math::

            min[({AU-A^0})^\dagger (AU-A^0)]

        where :math:`A` is a :math:`\text{m}\times\text{n }` array,
        :math:`\text{A}^0 \text{is the reference }\text{m}\times\text{n}` array and
        :math:`U` represents the optimum transformation array.

        Parameters
        ----------
        array_a : ndarray
            A 2D array representing the array to be transformed (as close as possible to array_b).

        array_b : ndarray
            A 2D array representing the reference array.

        t_array: ndarray
           A 2D array representing the 'optimum' transformation.

        Returns
        -------
        Returns the error given by the optimum choice of t_array.

        """
        at = np.dot(array_a, t_array)
        error = np.trace(np.dot((at - array_b).T, at - array_b))
        return error

    def double_sided_procrustes_error(self, array_a, array_b, t_array1, t_array2):
        """
        Returns the error for all double-sided procrustes problems

         .. math::

        min { ([t_array1].^\mathrm{T}*[array_a]*[t_array2] - [array_b]).^\mathrm{T}) * ([t_array1].^\mathrm{T}*[array_a]*[t_array2] - [array_b])) }
            : t_array1, t_array2 satisfies some condition.

        Parameters
        ----------

        array_a : ndarray
            A 2D array representing the array to be transformed (as close as possible to array_b).

        array_b : ndarray
            A 2D array representing the reference array.

        t_array: ndarray
           A 2D array representing the 1st 'optimum' transformation
           in the two-sided procrustes problems.
        t_array2: ndarray
           A 2D array representing the 2nd 'optimum' transformation
           in the two-sided procrustes problems.

        Returns
        ------------
        Returns the error given by the optimum choice of t_array1 and t_array2
        """
        t_trans_a_t = np.dot(np.dot(t_array1.T, array_a), t_array2)
        error = np.trace(np.dot((t_trans_a_t - array_b).T, (t_trans_a_t - array_b)))
        return error
