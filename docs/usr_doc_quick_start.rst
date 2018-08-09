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


Quick Start
###########

Procrustes is a set of interpretive geometric tools for analyzing matrices. To use the python
package, the user should make sure Procrustes have been installed properly including all the
dependencies. You may refer to the *Installization* part for more information.

How to use Procrustes
=====================

As a python library, Procrustes can be imported and used in python codes. People may get a better
understanding of how to use the package by the implemented examples.

Procrustes can be used as an computation library, either ending up with a script or a python
package. Here is a snippet for script.

.. code-block:: python
   :linenos:

   #!/usr/bin/env python

   # import Procrustes library
   from procrustes import *

   # Implement the script body here

Now you can make your script executable by:

.. code-block:: bash

   $ chmod +x script.py
   $ ./script.py

If you don't want to import every function in Procrustes, you can just import some or just one of
them.

.. code-block:: python
   :linenos:

   # import only on function
   from procrustes import symmetric

   # import some of the functions
   from procrustes import orthogonal, permutation

Getting Help
============

You may refer to the API documentation for technical details. You can also try this

.. code-block:: python

   # function name followed by a question mark
   help(permutation)

You will get a detailed information about the parameters and returns.

Here is an genric example showing how can Procrustes help people solve problems.

.. code-block:: python
   :linenos:

   # import the libraries
   import numpy as np
   from procrustes import orthogonal

   # Define a random array_a
   array_a = np.array([[-7.3, 2.8], [-7.1, -0.2],
                       [ 4. , 1.4], [ 1.3,  0. ]])
   # Define array_b
   array_b = np.array([[-5.90207845, -5.12791088],
                       [-6.74021234, -2.24043246],
                       [ 4.23759847,  0.05252849],
                       [ 1.22159856,  0.44463126]])
   # Find the orthogonal matrix by Procrustes to minimize the distance between them
   new_a, new_b, array_u, error_opt = orthogonal(array_a, array_b,
                                                 remove_zero_col=False,
                                                 remove_zero_row=False,
                                                 translate=False, scale=False)
   # Print the orthogonal matrix
   print('orthogonal matrix=', array_u)
   # Print the error
   print('erro=', error_opt)

