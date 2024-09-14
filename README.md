Procrustes
==========

[![This project supports Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org/downloads)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![CI Tox](https://github.com/theochem/procrustes/actions/workflows/ci_tox.yml/badge.svg?branch=main)](https://github.com/theochem/procrustes/actions/workflows/ci_tox.yml)
[![Documentation Status](https://readthedocs.org/projects/procrustes/badge/?version=latest)](https://procrustes.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/theochem/procrustes/branch/master/graph/badge.svg?token=3L96J5QQOT)](https://codecov.io/gh/theochem/procrustes)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/theochem/procrustes/master?filepath=doc%2Fnotebooks%2F)

The Procrustes library provides a set of functions for transforming a matrix to make it
as similar as possible to a target matrix. For more information, visit
[**Procrustes Documentation**](https://procrustes.qcdevs.org/).

Citation
--------

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

Dependencies
------------

The following dependencies are required to run Procrustes properly,

* Python >= 3.9: <http://www.python.org/>
* NumPy >= 1.21.5: <http://www.numpy.org/>
* SciPy >= 1.9.0: <http://www.scipy.org/>
* PyTest >= 5.3.4: <https://docs.pytest.org/>
* PyTest-Cov >= 2.8.0: <https://pypi.org/project/pytest-cov/>

Installation
------------

To install Procrustes using the conda package management system, install
[miniconda](https://conda.io/miniconda.html) or [anaconda](https://www.anaconda.com/download)
first, and then:

```bash
# Create and activate myenv conda environment (optional, but recommended)
conda create -n myenv python=3.11
conda activate myenv

# Install the stable release.
conda install -c theochem qc-procrustes
```

To install Procrustes with pip, you may want to create a
[virtual environment](https://docs.python.org/3/tutorial/venv.html), and then:

```bash
# Install the stable release.
pip install qc-procrustes
```

See <https://procrustes.qcdevs.org/usr_doc_installization.html> for full details.
