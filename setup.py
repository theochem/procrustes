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
"""Setup and Install Script."""


from distutils.core import setup


setup(
    name="procrustes",
    version="0.0",
    description="Procrustes Package",
    url="http://github.com/QuantumElephant/procrustes",
    license="GNU (Version 3)",
    author="Ayers Group",
    author_email="ayers@mcmaster.ca",
    package_dir={"procrustes": "procrustes"},
    packages=["procrustes"],
    # test_suite="nose.collector",
    requires=["numpy", "scipy", "sphinx"],
)
