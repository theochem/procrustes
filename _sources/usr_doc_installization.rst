..
    : The Procrustes library provides a set of functions for transforming
    : a matrix to make it as similar as possible to a target matrix.
    :
    : Copyright (C) 2017-2021 The QC-Devs Community
    :
    : This file is part of Procrustes.
    :
    : Procrustes is free software; you can redistribute it and/or
    : modify it under the terms of the GNU General Public License
    : as published by the Free Software Foundation; either version 3
    : of the License, or (at your option) any later version.
    :
    : Procrustes is distributed in the hope that it will be useful,
    : but WITHOUT ANY WARRANTY; without even the implied warranty of
    : MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    : GNU General Public License for more details.
    :
    : You should have received a copy of the GNU General Public License
    : along with this program; if not, see <http://www.gnu.org/licenses/>
    :
    : --


.. _usr_installation:

Installation
############

Downloading Code
================

The latest code can be obtained through theochem (https://github.com/theochem/procrustes) in Github,

.. code-block:: bash

   git clone git@github.com:theochem/procrustes.git

.. _usr_py_depend:

Dependencies
============

The following dependencies will be necessary for Procrustes to build properly,

* Python >= 3.6: `https://www.python.org/ <http://www.python.org/>`_
* SciPy >= 1.5.0: `https://www.scipy.org/ <http://www.scipy.org/>`_
* NumPy >= 1.18.5: `https://www.numpy.org/ <http://www.numpy.org/>`_
* Pip >= 19.0: `https://pip.pypa.io/ <https://pip.pypa.io/>`_
* PyTest >= 5.4.3: `https://docs.pytest.org/ <https://docs.pytest.org/>`_
* PyTest-Cov >= 2.8.0: `https://pypi.org/project/pytest-cov/ <https://pypi.org/project/pytest-cov/>`_
* Sphinx >= 2.3.0, if one wishes to build the documentation locally:
  `https://www.sphinx-doc.org/ <https://www.sphinx-doc.org/>`_

Installation
============

The stable release of the package can be easily installed through the *pip* and
*conda* package management systems, which install the dependencies automatically, if not
available. To use *pip*, simply run the following command:

.. code-block:: bash

    pip install qc-procrustes

To use *conda*, one can either install the package through Anaconda Navigator or run the following
command in a desired *conda* environment:

.. code-block:: bash

    conda install -c theochem qc-procrustes


Alternatively, the *Procrustes* source code can be download from GitHub (either the stable version
or the development version) and then installed from source. For example, one can download the latest
source code using *git* by:

.. code-block:: bash

    # download source code
    git clone git@github.com:theochem/procrustes.git
    cd procrustes

From the parent directory, the dependencies can either be installed using *pip* by:

.. code-block:: bash

    # install dependencies using pip
    pip install -r requirements.txt


or, through *conda* by:

.. code-block:: bash

    # create and activate myenv environment
    # Procruste works with Python 3.6, 3.7, and 3.8
    conda create -n myenv python=3.6
    conda activate myenv
    # install dependencies using conda
    conda install --yes --file requirements.txt


Finally, the *Procrustes* package can be installed (from source) by:

.. code-block:: bash

    # install Procrustes from source
    pip install .

.. _usr_testing:

Testing
=======

To make sure that the package is installed properly, the *Procrustes* tests should be executed using
*pytest* from the parent directory:

.. code-block:: bash

    # testing without coverage report
    pytest -v .


In addition, to generate a coverage report alongside testing, one can use:

.. code-block:: bash

    # testing with coverage report
    pytest --cov-config=.coveragerc --cov=procrustes procrustes/test

