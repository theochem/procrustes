# Procrustes Python Library

[![This project supports Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org/downloads)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![CI Tox](https://github.com/theochem/procrustes/actions/workflows/ci_tox.yml/badge.svg?branch=main)](https://github.com/theochem/procrustes/actions/workflows/ci_tox.yml)
[![docs](https://github.com/theochem/procrustes/actions/workflows/deploy_website.yaml/badge.svg?branch=main)](https://github.com/theochem/procrustes/actions/workflows/deploy_website.yaml)
[![CI CodeCov](https://github.com/theochem/procrustes/actions/workflows/ci_codecov.yml/badge.svg?branch=main)](https://github.com/theochem/procrustes/actions/workflows/ci_codecov.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/theochem/procrustes/main?filepath=doc%2Fnotebooks%2F)

The Procrustes library provides a set of functions for transforming a matrix to make it
as similar as possible to a target matrix. For more information, visit
[**Procrustes Documentation**](https://procrustes.qcdevs.org/).

## Citation

Please use [the following citation](https://doi.org/10.1016/j.cpc.2022.108334)
in any publication using Procrustes library:

```md
@article{Meng2022procrustes,
    title = {Procrustes: A python library to find transformations that maximize the similarity between matrices},
    author = {Fanwang Meng and Michael Richer and Alireza Tehrani and Jonathan La and Taewon David Kim and Paul W. Ayers and Farnaz Heidar-Zadeh},
    journal = {Computer Physics Communications},
    volume = {276},
    number = {108334},
    pages = {1--37},
    year = {2022},
    issn = {0010-4655},
    doi = {https://doi.org/10.1016/j.cpc.2022.108334},
    url = {https://www.sciencedirect.com/science/article/pii/S0010465522000522},
    keywords = {Procrustes analysis, Orthogonal, Symmetric, Rotational, Permutation, Softassign},
}
```

## Dependencies

The following dependencies are required to run Procrustes properly,

* Python >= 3.9: <http://www.python.org/>
* NumPy >= 1.21.5: <http://www.numpy.org/>
* SciPy >= 1.9.0: <http://www.scipy.org/>

To test Procrustes, the following dependencies are required,

* PyTest >= 8.3.0: <https://docs.pytest.org/>
* PyTest-Cov >= 5.0.0: <https://pypi.org/project/pytest-cov/>

## Installation

It is recommended to install `qc-procrustes` within a virtual environment.To create a virtual
environment, we can use the `venv` module (Python 3.3+,
https://docs.python.org/3/tutorial/venv.html), `miniconda` (https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), or
`pipenv` (https://pipenv.pypa.io/en/latest/).

### Installing from PyPI

To install `procrustes` with `pip`, we can install the latest stable release from the Python Package Index (PyPI) as follows:

```bash
    # install the stable release.
    pip install qc-procrustes
```

### Installing from The Prebuild Wheel Files

To download the prebuilt wheel files, visit the [PyPI page](https://pypi.org/project/qc-procrustes/)
and [GitHub releases](https://github.com/theochem/procrustes/tags).

```bash
    # download the wheel file first to your local machine
    # then install the wheel file
    pip install file_path/qc_procrustes-1.0.2a1-py3-none-any.whl
```

### Installing from the Source Code

In addition, we can install the latest development version from the GitHub repository as follows:

```bash
    # install the latest development version
    pip install git+https://github.com/theochem/procrustes.git
```

We can also clone the repository to access the latest development version, test it and install it as follows:

```bash
    # clone the repository
    git clone git@github.com:theochem/procrustes.git

    # change into the working directory
    cd procrustes
    # run the tests
    python -m pytest .

    # install the package
    pip install .

```

## More

See https://procrustes.qcdevs.org for full details.
