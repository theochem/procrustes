# -*- coding: utf-8 -*-
# Procrustes is a collection of interpretive chemical tools for
# analyzing outputs of the quantum chemistry calculations.
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
from distutils.core import setup

setup(
    name='procrustes',
    version='0.0',
    packages=['procrustes', 'procrustes.hungarian', 'procrustes.procrustes', 'procrustes.procrustes.test'],
    url='',
    #test_suite='nose.collector',
    license='MIT',
    author='Jonathan La, Farnaz Zadeh',
    author_email='',
    description='A package for basic procrustes problems'
)