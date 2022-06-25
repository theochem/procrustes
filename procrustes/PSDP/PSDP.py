import numpy as np
import scipy

"""
Let n be the number of dimensions we are concerned with in this problem.
Initially, E = I_n and rank[G] = n. 
"""
from input import n, F, G


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


def D(E, LE):
    # TODO
    pass


def make_positive(LE):
    Lambda, U = np.linalg.eig(M)
    Lambda_pos = [max(0, i) for i in Lambda]
    inv_U = np.linalg.inv(U)
    return np.dot(U, np.dot(np.diag(Lambda_pos), inv_U))


def find_minima(E, DE):
    # TODO
    pass


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
        # Initialize once rather than calling multiple times
        LE = L(E)
        if L(E) >= 0:
            print(f"Iteration: {i}, E = {E}")
            print(f"Required P = {T(E)@E}")
            break
        LE_pos = make_positive(LE)
        DE = D(E, LE_pos)
        w = find_minima(E, DE)
        E = E - w * DE
        i += 1
