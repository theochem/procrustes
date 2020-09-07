Procrustes
==========

<a href='https://docs.python.org/3.6/'><img src='https://img.shields.io/badge/python-3.6-blue.svg'></a>
<a href='https://docs.python.org/3.7/'><img src='https://img.shields.io/badge/python-3.7-blue.svg'></a>
<a href='https://docs.python.org/3.8/'><img src='https://img.shields.io/badge/python-3.8-blue.svg'></a>
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![Build Status](https://travis-ci.com/theochem/procrustes.svg?branch=master)](https://travis-ci.com/theochem/procrustes)
[![Documentation Status](https://readthedocs.org/projects/procrustes/badge/?version=latest)](https://procrustes.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/git@github.com:theochem/procrustes.git/master)

The Procrustes library provides a set of functions for transforming a matrix to make it
as similar as possible to a target matrix.
More documentation can be found at https://procrustes.readthedocs.io/en/latest/


License
-------

Procrustes is distributed under GNU (Version 3) License.


Dependencies
------------

The following dependencies are required to run Procrustes properly,

* Python >= 3.6: http://www.python.org/
* PIP >= 19.0: https://pip.pypa.io/
* SciPy >= 1.5.0: http://www.scipy.org/
* NumPy >= 1.18.5: http://www.numpy.org/
* PyTest >= 5.3.4: https://docs.pytest.org/
* PyTest-Cov >= 2.8.0: https://pypi.org/project/pytest-cov/


Installation
------------

Download the latest version of Procrustes with the following command.
```bash
    git clone git@github.com:theochem/procrustes.git
```

Then navigate to the Procrustes folder and run with package manager:

```bash
   cd procrustes
   pip install .
```

Or, if you want to install the package in an editable mode, run
```bash
    pip install -e .
```

To remove the package, run:

```bash
   pip uninstall procrustes
```

Testing
-------

To run tests with coverage report:

```bash
    pytest --cov-config=.coveragerc --cov=procrustes procrustes/test
```
Or if one does not want coverage report, run
```bash
    pytest .
```

Development
-----------

Any contributor is welcome regardless of their background or programming proficiency.
You may refer to the developer guideline for details of how to contribute.


References
----------

If you are using this package, please reference:

*Add a link to download bib or ris files*

