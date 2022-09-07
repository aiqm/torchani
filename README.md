# <img src=https://raw.githubusercontent.com/aiqm/torchani/master/logo1.png width=180/>  Accurate Neural Network Potential on PyTorch

Metrics:

[![conda-release](https://github.com/roitberg-group/torchani_sandbox/actions/workflows/conda-release.yml/badge.svg)](https://github.com/roitberg-group/torchani_sandbox/actions/workflows/conda-release.yml)
[![conda-page](https://img.shields.io/badge/conda--package-page-blue)](https://roitberg.chem.ufl.edu/projects/conda-packages-uf-gainesville)
![PyPI](https://img.shields.io/pypi/v/torchani.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/torchani.svg)

Checks:

[![CodeFactor](https://www.codefactor.io/repository/github/aiqm/torchani/badge/master)](https://www.codefactor.io/repository/github/aiqm/torchani/overview/master)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/aiqm/torchani.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/aiqm/torchani/alerts/)
[![Actions Status](https://github.com/roitberg-group/torchani_sandbox/workflows/flake8/badge.svg)](https://github.com/roitberg-group/torchani_sandbox/actions)
[![Actions Status](https://github.com/roitberg-group/torchani_sandbox/workflows/clang-format/badge.svg)](https://github.com/roitberg-group/torchani_sandbox/actions)
[![Actions Status](https://github.com/roitberg-group/torchani_sandbox/workflows/mypy/badge.svg)](https://github.com/roitberg-group/torchani_sandbox/actions)
[![Actions Status](https://github.com/roitberg-group/torchani_sandbox/workflows/unittests/badge.svg)](https://github.com/roitberg-group/torchani_sandbox/actions)
[![Actions Status](https://github.com/roitberg-group/torchani_sandbox/workflows/cuda/badge.svg)](https://github.com/roitberg-group/torchani_sandbox/actions)
[![Actions Status](https://github.com/aiqm/torchani/workflows/docs/badge.svg)](https://github.com/aiqm/torchani/actions)
[![Actions Status](https://github.com/aiqm/torchani/workflows/runnable-submodules/badge.svg)](https://github.com/aiqm/torchani/actions)
[![Actions Status](https://github.com/roitberg-group/torchani_sandbox/workflows/tools/badge.svg)](https://github.com/roitberg-group/torchani_sandbox/actions)

Deploy:

[![Actions Status](https://github.com/aiqm/torchani/workflows/deploy-docs/badge.svg)](https://github.com/aiqm/torchani/actions)
[![Actions Status](https://github.com/aiqm/torchani/workflows/deploy-pypi/badge.svg)](https://github.com/aiqm/torchani/actions)

We only provide compatibility with nightly PyTorch, but you can check if stable PyTorch happens to be supported by looking at the following badge:

[![Actions Status](https://github.com/roitberg-group/torchani_sandbox/actions/workflows/stable-torch.yml/badge.svg)](https://github.com/roitberg-group/torchani_sandbox/actions/workflows/stable-torch.yml)


TorchANI is a pytorch implementation of ANI. It is currently under alpha release, which means, the API is not stable yet. If you find a bug of TorchANI, or have some feature request, feel free to open an issue on GitHub, or send us a pull request.

<img src=https://raw.githubusercontent.com/aiqm/torchani/master/logo2.png width=500/>


# Install
Supported version information when install from anaconda:
- pytorch 1.12.1
- cuda 11.3
- python 3.8

Install from anaconda by the following (Note that we are hosting the packages only for internal usage at: https://roitberg.chem.ufl.edu/projects/conda-packages-uf-gainesville)

```bash
conda create -n ani -c https://roitberg.chem.ufl.edu/projects/conda-packages-uf-gainesville -c pytorch -c nvidia -c defaults -c conda-forge sandbox python=3.8
```
Which is equivalent to the following:
```bash
conda create -n ani python=3.8
conda activate ani
conda install -c https://roitberg.chem.ufl.edu/projects/conda-packages-uf-gainesville -c pytorch -c nvidia -c defaults -c conda-forge sandbox
```
- The `conda install` command could also be used for your own conda environment or could be used to update to the latest nightly version.  
- In the case where multiple updates has been released within a day, you may need to add a `--force-reinstall` flag instead of waiting for the next nightly update.

You could also build torchani from source, check at [TorchANI CSRC](torchani/csrc).

Note that if you updated TorchANI, you may also need to update PyTorch.

To run the tests and examples, you must manually download a data package

```bash
./download.sh
pip install -r test_requirements.txt
cd tests
python -m pytest -v -s *.py
```

# Command Line Interface
After installation, there will be an executable script (torchani) available on you path, which contain some builtin utilities.

Check usage with: 
```
$ torchani --help
```

# TorchANI Extensions
[TorchANI CSRC](torchani/csrc) provides AEV CUDA Extension (speedup for AEV calculation) and MNP extension (Multi Net Parallel for inference) for speedup training and inference.

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
pip install -r docs_requirements.txt
```

To manually run unit tests, do

```bash
pytest -v
```

If you opened a pull request, you could see your generated documents at https://aiqm.github.io/torchani-test-docs/ after you `docs` check succeed.
Keep in mind that this repository is only for the purpose of convenience of development, and only keeps the latest push.
The CI runing for other pull requests might overwrite this repository. You could rerun the `docs` check to overwrite this repo to your build.


# Note to TorchANI developers

Never commit to the master branch directly. If you need to change something, create a new branch, submit a PR on GitHub.

You must pass all the tests on GitHub before your PR can be merged.

Code review is required before merging pull request.
