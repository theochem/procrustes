"""
"""

import numpy as np
from procrustes import Procrustes


def test_procrustes_base_array():
    # check procrustes arrays
    array1 = np.array([[1, 2], [3, 4]])
    array2 = np.array([[5, 6]])
    procrust = Procrustes(array1, array2)
    assert procrust.array_a.shape == (2, 2)
    assert procrust.array_b.shape == (2, 2)
    assert (abs(procrust.array_a - array1) < 1.e-10).all()
    assert (abs(procrust.array_b - np.array([[5, 6], [0, 0]])) < 1.e-10).all()

    # check procrustes arrays
    array1 = np.array([[4, 7, 2], [1, 3, 5]])
    array2 = np.array([[5], [2]])
    procrust = Procrustes(array1, array2)
    assert procrust.array_a.shape == (2, 3)
    assert procrust.array_b.shape == (2, 1)
    assert (abs(procrust.array_a - array1) < 1.e-10).all()
    assert (abs(procrust.array_b - array2) < 1.e-10).all()
