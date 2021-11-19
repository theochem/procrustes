Procrustes
==========

<a href='https://docs.python.org/3.6/'><img src='https://img.shields.io/badge/python-3.6-blue.svg'></a>
<a href='https://docs.python.org/3.7/'><img src='https://img.shields.io/badge/python-3.7-blue.svg'></a>
<a href='https://docs.python.org/3.8/'><img src='https://img.shields.io/badge/python-3.8-blue.svg'></a>
<a href='https://docs.python.org/3.9/'><img src='https://img.shields.io/badge/python-3.9-blue.svg'></a>
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![GitHub Actions CI Tox Status](https://github.com/theochem/procrustes/actions/workflows/ci_tox.yml/badge.svg?branch=master)](https://github.com/theochem/procrustes/actions/workflows/ci_tox.yml)
[![Documentation Status](https://readthedocs.org/projects/procrustes/badge/?version=latest)](https://procrustes.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/theochem/procrustes/master?filepath=doc%2Fnotebooks%2F)

The Procrustes library provides a set of functions for transforming a matrix to make it
as similar as possible to a target matrix. For more information, visit
[**Procrustes Documentation**](https://procrustes.readthedocs.io/en/latest/).


Citation
--------

Please use the following citation in any publication using Procrustes library:

> **"Procrustes: ", F. Meng, et al.**


Dependencies
------------

The following dependencies are required to run Procrustes properly,

* Python >= 3.6: http://www.python.org/
* NumPy >= 1.18.5: http://www.numpy.org/
* SciPy >= 1.5.0: http://www.scipy.org/
* PyTest >= 5.3.4: https://docs.pytest.org/
* PyTest-Cov >= 2.8.0: https://pypi.org/project/pytest-cov/
* PIP >= 19.0: https://pip.pypa.io/


Installation
------------


#### Anaconda
To install Procrustes using the conda package management system, install
[miniconda](https://conda.io/miniconda.html) or [anaconda](https://www.anaconda.com/download)
first, and then:

```bash
    # Create and activate myenv conda environment (optional, but recommended)
    conda create -n myenv python=3.6
    conda activate myenv

    # Install the stable release.
    conda install -c theochem/label/dev qc-procrustes
```

#### Pip
To install Procrustes with pip, you may want to create a
[virtual environment](https://docs.python.org/3/tutorial/venv.html), and then:


```bash
    # Install the stable release.
    pip install qc-procrustes
```

For more information, please see https://procrustes.readthedocs.io/en/latest/usr_doc_installization.html.

#### Source
To install Procrustes from source, use:

```bash
    # Clone the repository
    git clone git@github.com:theochem/procrustes.git
    cd procrustes
    
    # Create a conda environment
    conda create -n procrustes python=3.6
    conda activate procruestes
    
    # Install with pip
    pip install -e .
```


Quick Start
------------

As an example, consider the problem of finding the Procrustes distance between an array A, and 
that same array (orthogonally) transformed and shifted.

```python
import numpy as np
from scipy.stats import ortho_group
from procrustes import orthogonal

# random input 10x7 matrix A
a = np.random.rand(10, 7)

# random orthogonal 7x7 tranformation matrix T
t = ortho_group.rvs(7)

# target matrix B (which is a shifted AxT)
b = np.dot(a, t) + np.random.rand(1, 7)

# orthogonal Procrustes analysis with translation
result = orthogonal(a, b, scale=True, translate=True)

# display Procrustes results
print("Procrustes distance is: %s", result.error)
assert np.allclose(result.t, t)
```

Tutorials
------------

Tutorials are available online, and can be found at:
https://procrustes.readthedocs.io/en/latest/usr_doc_tutorials.html.

Tests
------------

To execute the tests, run:

```bash
pytest -v .
```
