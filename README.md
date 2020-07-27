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


# Citation

Please cite the following paper if you use TorchANI 

* Xiang Gao, Farhad Ramezanghorbani, Olexandr Isayev, Justin S. Smith, and Adrian E. Roitberg. *TorchANI: A Free and Open Source PyTorch Based Deep Learning Implementation of the ANI Neural Network Potentials*. Journal of Chemical Information and Modeling 2020 60 (7), 3408-3415, [![DOI for Citing](https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.0c00451-green.svg)](https://doi.org/10.1021/acs.jcim.0c00451)

[![JCIM Cover](https://pubs.acs.org/na101/home/literatum/publisher/achs/journals/content/jcisd8/2020/jcisd8.2020.60.issue-7/jcisd8.2020.60.issue-7/20200727/jcisd8.2020.60.issue-7.largecover.jpg)](https://pubs.acs.org/toc/jcisd8/60/7)

* Please refer to [isayev/ASE_ANI](https://github.com/isayev/ASE_ANI) for ANI model references.

# ANI model parameters
All the ANI model parameters including (ANI2x, ANI1x, and ANI1ccx) are accessible from the following repositories:
- [isayev/ASE_ANI](https://github.com/isayev/ASE_ANI)
- [aiqm/ani-model-zoo](https://github.com/aiqm/ani-model-zoo)


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
