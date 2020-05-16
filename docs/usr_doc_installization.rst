..
    : Procrustes is a collection of interpretive chemical tools for
    : analyzing outputs of the quantum chemistry calculations.
    :
    : Copyright (C) 2017-2020 The Procrustes Development Team
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

The latest code can be obtained through Github (private at present),

.. code-block:: bash

   git clone git@github.com:theochem/procrustes.git

.. _usr_py_depend:

Dependencies
============

The following dependencies will be necessary for Procrustes to build properly,

* Python >= 2.7, or Python >= 3.6: http://www.python.org/
* PIP >= 19.0: https://pip.pypa.io/
* SciPy >= 1.0.0: http://www.scipy.org/
* NumPy >= 1.14: http://www.numpy.org/
* PyTest >= 5.3.0: https://docs.pytest.org/
* PyTest-Cov >= 2.8.0: https://pypi.org/project/pytest-cov/

Python Dependencies
~~~~~~~~~~~~~~~~~~~

There are several different ways of installing python dependencies. A better and preferred
practice is to creat a virtual environment by using

#. `venv`, https://docs.python.org/3/library/venv.html
#. `conda`,  https://docs.conda.io/en/latest/
#. `virtualenv`, https://virtualenv.pypa.io/en/latest/

* **venv**

This is a standard builtin library since Python 3.3.

    .. code-block:: bash

        # mode details at
        # https://docs.python.org/3/library/venv.html
        python -m venv myenv
        # on Windows
        myenv\Scripts\activate.bat
        # on Unix or MacOS
        source myenv/bin/activate
        # install the dependencies
        pip install -r requirements.txt

* **conda**

    .. code-block:: bash

        codna create -n myenv python=3.6
        # activate the virtual environment
        conda activate myenv
        # install the dependencies
        conda install --yes --file requirements.txt

* **virtualenv**

You need to install it with `pip install virtualenv`.

    .. code-block:: bash

        # create the virtual environment
        virtualenv myenv
        # on Windows
        myenv\Scripts\activate
        # on Unix or MacOS
        source myenv/bin/activate
        # install the dependencies
        pip install -r requirements.txt


And then install the package accordingly.

The other option is to install the dependencies in the operating system as follows.

* **Ubuntu Linux 18.04**

  .. code-block:: bash

    sudo apt-get install python-dev python-pip python-numpy python-scipy python-pytest python-pytest-cov

* **Ubuntu Linux 15.04 & 14.04**

  .. code-block:: bash

     sudo apt-get install python-dev python-pip python-numpy python-scipy python-pytest
     pip install --user --upgrade numpy scipy pytest pytest-cov

* **Fedora 30 and later**

  .. code-block:: bash

    sudo dnf install python3-dev python3-pip python3-numpy python3-scipy python3-pytest python3-pytest-cov

* **Mac OS (using MacPorts)**

  .. code-block:: bash

    # the command line works for python36
    # change to py37, py38, py39 if you want
    sudo port install python36; sudo port select --set python python36
    sudo port install py36-pytest; sudo port select --set pytest py36-pytest
    sudo port install py36-pytest-cov; sudo port select --set pytest-cov py36-pytest-cov
    sudo port install py36-numpy py36-scipy
    sudo port install py36-pip; sudo port select --set pip pip36

Installation
============

To install Procrustes:

.. code-block:: bash

    # download the package
    git clone git@github.com:theochem/procrustes.git
    # navigate to the package folder
    cd procrustes
    # installation
   ./setup.py install --user


Or by using the python package manager pip:

.. code-block:: bash

   pip install -e ./ --user

If one wants to remove the package, people can just run:

.. code-block:: bash

   pip uninstall procrustes

.. todo::
    #. Add Anaconda installation support
    #. Add MacPorts installation support
    #. Add pip command line installation support

.. _usr_testing:

Testing
=======

To make sure the package is working properly, it is recommended to run tests:

.. code-block:: bash

   pytest --cov-config=.coveragerc --cov=procrustes procrustes/test

Or simply run

.. code-block:: bash

   pytest .
