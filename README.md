Procrustes
==========

<a href='https://docs.python.org/2.7/'><img src='https://img.shields.io/badge/python-2.7-blue.svg'></a>
<a href='https://docs.python.org/3.6/'><img src='https://img.shields.io/badge/python-3.6-blue.svg'></a>


The Procrustes library provides a set of functions for transforming a matrix to make it
as similar as possible to a target matrix.


License
-------

Procrustes is distributed under GNU (Version 3) License.


Dependencies
------------

The following dependencies are required to run Procrustes properly,

* Python >= 2.7, or Python >= 3.6: http://www.python.org/
* PIP >= 9.0: https://pip.pypa.io/
* SciPy >= 1.0.0: http://www.scipy.org/
* NumPy >= 1.14: http://www.numpy.org/
* Nosetests >= 1.3.7: http://readthedocs.org/docs/nose/en/latest/


Installation
------------

To install using package manager, run:

```bash
   pip install -e ./ --user
```

To remove the package, run:

```bash
   pip uninstall procrustes
```

To install the cloned package, run:

```bash
   ./setup.py install --user
```

Testing
-------

To run tests:

```bash
    nosetests --with-coverage --cover-package procrustes/. --cover-tests --cover-erase
```

Development
-----------

Any contributor is welcome regardless of their background or programming proficiency.
You may refer to the developer guideline for details of how to contribute.


References
----------

If you are using this package, please reference:

*Add a link to download bib or ris files*

