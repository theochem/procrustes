# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2021 The QC-Devs Community
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


import os

from setuptools import setup


def get_version_info():
    """Read __version__ and DEV_CLASSIFIER from version.py, using exec, not import."""
    fn_version = os.path.join("procrustes", "_version.py")
    if os.path.isfile(fn_version):
        myglobals = {}
        with open(fn_version, "r") as f:
            exec(f.read(), myglobals)  # pylint: disable=exec-used
        return myglobals["__version__"], myglobals["DEV_CLASSIFIER"]
    return "0.0.0.post0", "Development Status :: 2 - Pre-Alpha"


def get_readme():
    """Load README.rst for display on PyPI."""
    with open('README.md') as fhandle:
        return fhandle.read()


VERSION, DEV_CLASSIFIER = get_version_info()

setup(
    name="qc-procrustes",
    version=VERSION,
    description="Procrustes Package",
    long_description=get_readme(),
    long_description_content_type='text/markdown',
    url="http://github.com/theochem/procrustes",
    license="GNU (Version 3)",
    author="QC-Devs Community",
    author_email="qcdevs@gmail.com",
    package_dir={"procrustes": "procrustes"},
    packages=["procrustes", "procrustes.test"],
    include_package_data=True,
    classifiers=[
        DEV_CLASSIFIER,
        'Environment :: Console',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Intended Audience :: Science/Research',
    ],
    install_requires=["numpy>=1.18.5", "scipy>=1.5.0", "pytest>=5.4.3", "sphinx>=2.3.0"],
)
