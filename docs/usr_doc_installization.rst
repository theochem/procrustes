..
    : Procrustes is a collection of interpretive chemical tools for
    : analyzing outputs of the quantum chemistry calculations.
    :
    : Copyright (C) 2017-2018 The Procrustes Development Team
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

Installization
##############

Downloading Code
================

The latest code can be obtained through Github (private at present),

.. code-block:: bash

   git clone git@github.com:QuantumElephant/procrustes.git

.. _usr_py_depend:

Dependencies
============

The following dependencies will be necessary for ChemTools to build properly,

* Python >= 2.7, >= 3.6: http://www.python.org/
* PIP >= 9.0: https://pip.pypa.io/
* SciPy >= 1.0.0: http://www.scipy.org/
* NumPy >= 1.14: http://www.numpy.org/
* Nosetests >= 1.3.7: http://readthedocs.org/docs/nose/en/latest/

Python Dependencies
~~~~~~~~~~~~~~~~~~~

To install the required dependencies (Python related dependencies):

* **Ubuntu Linux 18.04**

  .. code-block:: bash

     sudo apt-get install python-dev python-pip python-numpy python-scipy python-nose

* **Ubuntu Linux 15.04 & 14.04**

  .. code-block:: bash

     sudo apt-get install python-dev python-pip python-numpy python-scipy python-nose
     pip install --user --upgrade numpy scipy nose

* **Mac OS (using MacPorts)**

  .. code-block:: bash

     sudo port install python36; sudo port select --set python python36
     sudo port install py36-nose; sudo port select --set nosetests nosetests36
     sudo port install py36-numpy py36-scipy
     sudo port install py36-pip; sudo port select --set pip pip36

* **All other systems**


.. todo::
    #. Remains to be tested.
    #. Jupyter support


Installation
============

To install Procrustes:

.. code-block:: bash

   ./setup.py install --user

Or by using the python package manager pip:

.. code-block:: bash

   pip install -e ./ --user

If one wants to remove the package, people can just run:

.. code-block:: bash

   pip uninstall procrustes

.. todo::
    #. Add Anaconda installization support
    #. Add Macports installization support
    #. Add pip command line installization support

.. _usr_testing:

Testing
=======

To make sure the package is working properly, it is recommended to run tests:

.. code-block:: bash

   nosetests procrustes/.

