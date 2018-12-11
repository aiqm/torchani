# <img src=https://raw.githubusercontent.com/aiqm/torchani/master/logo1.png width=180/>  Accurate Neural Network Potential on PyTorch

[![flake8](https://zasdfgbnm.visualstudio.com/torchani/_apis/build/status/flake8)](https://zasdfgbnm.visualstudio.com/torchani/_build/latest?definitionId=3)
[![docs](https://zasdfgbnm.visualstudio.com/torchani/_apis/build/status/docs)](https://zasdfgbnm.visualstudio.com/torchani/_build/latest?definitionId=4)
[![runnable submodules](https://zasdfgbnm.visualstudio.com/torchani/_apis/build/status/runnable-submodules)](https://zasdfgbnm.visualstudio.com/torchani/_build/latest?definitionId=5)
[![unit tests](https://zasdfgbnm.visualstudio.com/torchani/_apis/build/status/unit-tests)](https://zasdfgbnm.visualstudio.com/torchani/_build/latest?definitionId=6)
[![tools](https://zasdfgbnm.visualstudio.com/torchani/_apis/build/status/tools)](https://zasdfgbnm.visualstudio.com/torchani/_build/latest?definitionId=7)
[![CodeFactor](https://www.codefactor.io/repository/github/aiqm/torchani/badge/master)](https://www.codefactor.io/repository/github/aiqm/torchani/overview/master)
[![codecov](https://codecov.io/gh/aiqm/torchani/branch/master/graph/badge.svg)](https://codecov.io/gh/aiqm/torchani)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/aiqm/torchani.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/aiqm/torchani/alerts/)

TorchANI is a pytorch implementation of ANI. It is currently under alpha release, which means, the API is not stable yet. If you find a bug of TorchANI, or have some feature request, feel free to open an issue on GitHub, or send us a pull request.

<img src=https://raw.githubusercontent.com/aiqm/torchani/master/logo2.png width=500/>

# Install

TorchANI requires the latest preview version of PyTorch. You can install PyTorch by

```bash
conda install pytorch-nightly -c pytorch
```
If you updated TorchANI, you may also need to update PyTorch:

```bash
conda update pytorch-nightly -c pytorch
```

After installing the correct PyTorch, you can install TorchANI by:

```bash
pip install torchani
```

See also [PyTorch's official site](https://pytorch.org/get-started/locally/) for instructions of installing latest preview version of PyTorch.

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

# Note to TorchANI developers

Never commit to the master branch directly. If you need to change something, create a new branch, submit a PR on GitHub.

You must pass all the tests on GitHub before your PR can be merged.

Code review is required before merging pull request.

To manually run unit tests, do `python setup.py test`
