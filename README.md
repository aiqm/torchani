# <img src=https://raw.githubusercontent.com/aiqm/torchani/master/logo1.png width=180/>  Accurate Neural Network Potential on PyTorch

Metrics:

![PyPI](https://img.shields.io/pypi/v/torchani.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/torchani.svg)

Checks:

[![Actions Status](https://github.com/aiqm/torchani/workflows/docs/badge.svg)](https://github.com/aiqm/torchani/actions)
[![Actions Status](https://github.com/aiqm/torchani/workflows/flake8/badge.svg)](https://github.com/aiqm/torchani/actions)
[![Actions Status](https://github.com/aiqm/torchani/workflows/mypy/badge.svg)](https://github.com/aiqm/torchani/actions)
[![Actions Status](https://github.com/aiqm/torchani/workflows/runnable%20submodules/badge.svg)](https://github.com/aiqm/torchani/actions)
[![Actions Status](https://github.com/aiqm/torchani/workflows/tools/badge.svg)](https://github.com/aiqm/torchani/actions)
[![Actions Status](https://github.com/aiqm/torchani/workflows/unit%20tests/badge.svg)](https://github.com/aiqm/torchani/actions)
[![CodeFactor](https://www.codefactor.io/repository/github/aiqm/torchani/badge/master)](https://www.codefactor.io/repository/github/aiqm/torchani/overview/master)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/aiqm/torchani.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/aiqm/torchani/alerts/)

Deploy:

[![Actions Status](https://github.com/aiqm/torchani/workflows/deploy-docs/badge.svg)](https://github.com/aiqm/torchani/actions)
[![Actions Status](https://github.com/aiqm/torchani/workflows/deploy-pypi/badge.svg)](https://github.com/aiqm/torchani/actions)

We only provide compatibility with nightly PyTorch, but you can check if stable PyTorch happens to be supported by looking at the following badge:

[![Actions Status](https://github.com/aiqm/torchani/workflows/stable-torch/badge.svg)](https://github.com/aiqm/torchani/actions)


TorchANI is a pytorch implementation of ANI. It is currently under alpha release, which means, the API is not stable yet. If you find a bug of TorchANI, or have some feature request, feel free to open an issue on GitHub, or send us a pull request.

<img src=https://raw.githubusercontent.com/aiqm/torchani/master/logo2.png width=500/>

# Install

TorchANI requires the latest preview version of PyTorch. You can install PyTorch by the following commands (assuming cuda10):

```bash
pip install numpy
pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu100/torch_nightly.html
```

If you updated TorchANI, you may also need to update PyTorch:

```bash
pip install --upgrade --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu100/torch_nightly.html
```

After installing the correct PyTorch, you can install TorchANI by:

```bash
pip install torchani
```

See also [PyTorch's official site](https://pytorch.org/get-started/locally/) for instructions of installing latest preview version of PyTorch.

To run the tests and examples, you must manually download a data package

```bash
./download.sh
```

# Paper

The original ANI-1 paper is:

* Smith JS, Isayev O, Roitberg AE. ANI-1: an extensible neural network potential with DFT accuracy at force field computational cost. Chemical science. 2017;8(4):3192-203.

We are planning a seperate paper for TorchANI, it will be available when we are ready for beta release of TorchANI.

See also: [isayev/ASE_ANI](https://github.com/isayev/ASE_ANI)

# Develop

To install TorchANI from GitHub:

```bash
git clone https://github.com/aiqm/torchani.git
cd torchani
pip install -e .
```

After TorchANI has been installed, you can build the documents by running `sphinx-build docs build`. But make sure you
install dependencies:
```bash
pip install sphinx sphinx-gallery pillow matplotlib sphinx_rtd_theme
```

To manually run unit tests, do `python setup.py nosetests`

If you opened a pull request, you could see your generated documents at https://aiqm.github.io/torchani-test-docs/ after you `docs` check succeed.
Keep in mind that this repository is only for the purpose of convenience of development, and only keeps the latest push.
The CI runing for other pull requests might overwrite this repository. You could rerun the `docs` check to overwrite this repo to your build.

# Note to TorchANI developers

Never commit to the master branch directly. If you need to change something, create a new branch, submit a PR on GitHub.

You must pass all the tests on GitHub before your PR can be merged.

Code review is required before merging pull request.
