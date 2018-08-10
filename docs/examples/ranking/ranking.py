# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2018 The Procrustes Development Team
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


r"""Two Sided Permutation Procrustes Example: Ranking by Reordering Method."""


from __future__ import absolute_import, division, print_function
import numpy as np

from procrustes import permutation_2sided


__all__ = [
    "ranking",
    "_rank_differential",
    "_check_input"
]


def ranking(D, perm_mode='normal1'):
    r""" Compute the ranking vector."""
    _check_input(D)

    R_hat = _rank_differential(D)
    _, _, Q, e_opt = permutation_2sided(D, R_hat,
                                        remove_zero_col=False,
                                        remove_zero_row=False,
                                        mode=perm_mode)
    # Compute the rank
    _, rank = np.where(Q == 1)
    rank += 1

    return rank


def _rank_differential(D):
    r""" Compute the rank differential based on the shape of input data.
    """
    N = np.shape(D)[0]
    R_hat = np.zeros((N, N))
    # Compute the upper triangle part of R_hat
    a = []
    for each in range(N):
        # print(each)
        a.extend(range(0, N-each))
    # Get the R_hat
    R_hat[np.triu_indices_from(R_hat, 0)] = a
    return R_hat


def _check_input(D):
    r"""Check if the input is squared."""
    m, n = np.shape(D)
    if not m == n:
        raise ValueError("Input matrix should be squared one.")


if __name__ == "__main__":
    ranking()
