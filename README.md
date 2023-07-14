<h2 align="center">
  datagnosis
</h2>

<h4 align="center">
    A Data-Centric AI library for measuring hardness categorization.
</h4>


<div align="center">

<!-- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Vr2PJswgfFYBkJCm3hhVkuH-9dXnHeYV?usp=sharing)
[![Tests Quick Python](https://github.com/vanderschaarlab/datagnosis/actions/workflows/test_pr.yml/badge.svg)](https://github.com/vanderschaarlab/datagnosis/actions/workflows/test_pr.yml)
[![Tests Full Python](https://github.com/vanderschaarlab/datagnosis/actions/workflows/test_full.yml/badge.svg)](https://github.com/vanderschaarlab/datagnosis/actions/workflows/test_full.yml)
[![Tutorials](https://github.com/vanderschaarlab/datagnosis/actions/workflows/test_tutorials.yml/badge.svg)](https://github.com/vanderschaarlab/datagnosis/actions/workflows/test_tutorials.yml)
[![Documentation Status](https://readthedocs.org/projects/datagnosis/badge/?version=latest)](https://datagnosis.readthedocs.io/en/latest/?badge=latest)

[![](https://pepy.tech/badge/datagnosis)](https://pypi.org/project/datagnosis/) -->

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2301.07573-b31b1b.svg)](https://arxiv.org/abs/2301.07573) -->
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/vanderschaarlab/datagnosis/blob/main/LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![about](https://img.shields.io/badge/about-The%20van%20der%20Schaar%20Lab-blue)](https://www.vanderschaar-lab.com/)

<!-- ![image](https://github.com/vanderschaarlab/datagnosis/raw/main/docs/arch.png "datagnosis") -->

</div>


## Features:
- :key: Easy to extend pluginable architecture.
- :cyclone: Several evaluation metrics for correctness and privacy.
<!-- - :book: [Read the docs !](https://datagnosis.readthedocs.io/) -->
<!-- - :airplane: [Checkout the tutorials!](https://github.com/vanderschaarlab/datagnosis#-tutorials) -->

*Please note: datagnosis does not handle missing data and so these values must be imputed first [HyperImpute](https://github.com/vanderschaarlab/hyperimpute) can be used to do this.*

## :rocket: Installation

The library can be installed from PyPI using
```bash
$ pip install datagnosis
```
or from source, using
```bash
$ pip install .
```
Other library extensions:
 * Install the library with unit-testing support
```bash
 pip install datagnosis[testing]
```

## :boom: Sample Usage

### Generic data
 

## 🔑 Methods

### Generic methods
| Method | Type| Description | Score | Reference |
| --- | --- | --- | --- | --- |
| Area Under the Margin (AUM) | Generic | Characterizes data examples based on the margin of a classifier – i.e. the difference between the logit values of the correct class and the next class. | Hard - low scores. | [AUM Paper](https://arxiv.org/abs/2001.10528) |
| Cleanlab *(rename) | Generic |Confident learning estimates the joint distribution of noisy and true labels — characterizing data as easy and hard for mislabeling. | Hard - low scores | [Cleanlab Paper](https://arxiv.org/pdf/1911.00068.pdf) |
| Conf Agree | Generic | Agreement measures the agreement of predictions on the same example. | Hard - low scores | [ Conf Agree Paper](https://arxiv.org/pdf/1910.13427.pdf)|
| Data IQ |Generic | Data-IQ computes the aleatoric uncertainty and confidence to characterize the data into easy, ambiguous and hard examples. | Hard - low confidence scores. High Aleatoric Uncertainty scores define ambiguous | [Data-IQ Paper](https://arxiv.org/abs/2210.13043) |
| Data Maps | Generic |Data Maps focuses on measuring variability (epistemic uncertainty) and confidence to characterize the data into easy, ambiguous and hard examples.|Hard - low confidence scores. High Epistemic Uncertainty scores define ambiguous| [Data-Maps Paper](https://arxiv.org/abs/2009.10795)|
| Gradient Normed (GraNd) | Generic |GraNd measures the gradient norm to characterize data. | Hard - high scores | [GraNd Paper](https://arxiv.org/abs/2107.07075)|
| Error L2-Norm (EL2N) | Generic | EL2N calculates the L2 norm of error over training in order to characterize data for computational purposes. | Hard - high scores | [EL2N Paper](https://arxiv.org/abs/2107.07075)|
| Forgetting | Generic |Forgetting scores analyze example transitions through training. i.e., the time a sample correctly learned at one epoch is then forgotten. | Hard - high scores | [Forgetting Paper](https://arxiv.org/abs/1812.05159)|
| Large Loss | Generic |Large Loss characterizes data based on sample-level loss magnitudes. | Hard - high scores | [Large Loss Paper](https://arxiv.org/abs/2106.00445)|
| Prototypicalilty | Generic |Prototypicality calculates the latent space clustering distance of the sample to the class centroid as the metric to characterize data. | Hard - high scores |[Prototypicalilty Paper](https://arxiv.org/abs/2206.14486) |
| Variance of Gradients (VOG) | Generic |VoG (Variance of gradients) estimates the variance of gradients for each sample over training | Hard - high scores |[VOG Paper](https://arxiv.org/abs/2008.11600) |
| ALLSH | Images |ALLSH computes the KL divergence of softmax outputs between original and augmented samples to characterize data. | Hard - high scores||[ALLSH Paper](https://arxiv.org/abs/2205.04980) |

*Generic type plugins can be used for tabular or image data. Image type plugins only work for images.*


## :hammer: Tests

Install the testing dependencies using
```bash
pip install .[testing]
```
The tests can be executed using
```bash
pytest -vsx
```

# Contributing to datagnosis

We want to make contributing to datagnosis is as easy and transparent as possible. We hope to collaborate with as many people as we can.


## Development installation

First create a new environment. It is recommended that you use conda. This can be done as follows:
```bash
conda create -n your-datagnosis-env python=3.9
conda activate your-datagnosis-env
```
*Python versions , 3.8, 3.9, 3.10, 3.11 are all compatible, but it is best to use the most up to date version you can, as some models may not support older python versions.*

To get the development installation with all the necessary dependencies for
linting, testing, auto-formatting, and pre-commit etc. run the following:
```bash
git clone https://github.com/vanderschaarlab/datagnosis.git
cd datagnosis
pip install -e .[testing]
```

Please check that the pre-commit is properly installed for the repository, by running:
```bash
pre-commit run --all
```
This checks that you are set up properly to contribute, such that you will match the code style in the rest of the project. This is covered in more detail below.


## Our Development Process

### Code Style

We believe that having a consistent code style is incredibly important. Therefore datagnosis imposes certain rules on the code that is contributed and the automated tests will not pass, if the style is not adhered to. These tests passing is a requirement for a contribution being merged. However, we make adhering to this code style as simple as possible. First, all the libraries required to produce code that is compatible with datagnosis's Code Style are installed in the step above when you set up the development environment. Secondly, these libraries are all triggered by pre-commit, so once you are set-up, you don't need to do anything. When you run `git commit`, any simple changes to enforce the style will run automatically and other required changes are explained in the stdout for you to go through and fix.

datagnosis uses the [black](https://github.com/ambv/black) and [flake8](https://github.com/PyCQA/flake8) code formatter to enforce a common code style across the code base. No additional configuration should be needed (see the [black documentation](https://black.readthedocs.io/en/stable/installation_and_usage.html#usage) for advanced usage).

Also, datagnosis uses [isort](https://github.com/timothycrosley/isort) to sort imports alphabetically and separate into sections.


#### Type Hints

datagnosis is fully typed using python 3.7+ [type hints](https://www.python.org/dev/peps/pep-0484/). This is enforced for contributions by [mypy](https://github.com/python/mypy), which is a static type-checker.


#### Tests

To run the tests, you can either use `pytest` (again, installed with the testing extra above).
The following testing command is good for checking your code, as it skips the tests that take a long time to run.
```bash
pytest -vvvsx -m "not slow" --durations=50
```

But the full test suite can be run with the following command.
```bash
pytest -vvvs  --durations=50
```

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you have added code that should be tested, add tests in the same style as those already present in the repo.
3. If you have changed APIs, document the API change in the PR.
4. Ensure the test suite passes.
5. Make sure your code passes the pre-commit, this will be required in order to commit and push, if you have properly installed pre-commit, which is included in the testing extra.


## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.


## License

By contributing to datagnosis, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree. You should therefore, make sure that if you have introduced any dependencies that they also are covered by a license that allows the code to be used by the project and is compatible with the license in the root directory of this project.

<!-- ## Citing

If you use this code, please cite the associated paper:

```
@misc{https://doi.org/10.48550/arxiv.2301.07573,
  doi = {10.48550/ARXIV.2301.07573},
  url = {https://arxiv.org/abs/2301.07573},
  author = {Qian, Zhaozhi and Cebere, Bogdan-Constantin and van der Schaar, Mihaela},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {datagnosis: facilitating innovative use cases of synthetic data in different data modalities},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}
``` -->
