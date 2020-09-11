# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2020 The Procrustes Development Team
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


import io
from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with io.open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="procrustes",
    version="0.0.1-alpha",
    description="Procrustes Package",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="http://github.com/theochem/procrustes",
    license="GNU (Version 3)",
    author="Ayers Group",
    author_email="ayers@mcmaster.ca",
    package_dir={"procrustes": "procrustes"},
    packages=["procrustes"],
    install_requires=["numpy>=1.18.5", "scipy>=1.5.0", "pytest>=5.4.3", "sphinx>=2.3.0"],
)
