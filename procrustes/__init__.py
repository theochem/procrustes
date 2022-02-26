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
"""Package for Various Procrustes Algorithms."""


try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0.post0"


from procrustes.utils import *
from procrustes.kopt import *
from procrustes.orthogonal import *
from procrustes.permutation import *
from procrustes.rotational import *
from procrustes.softassign import *
from procrustes.symmetric import *
from procrustes.generic import *
from procrustes.generalized import *
