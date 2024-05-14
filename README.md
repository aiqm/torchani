# <img src=https://raw.githubusercontent.com/aiqm/torchani/master/logo1.png width=180/>  Accurate Neural Network Potential on PyTorch

Metrics:

[![conda-release](https://github.com/roitberg-group/torchani_sandbox/actions/workflows/conda-release.yml/badge.svg)](https://github.com/roitberg-group/torchani_sandbox/actions/workflows/conda-release.yml)
[![conda-page](https://img.shields.io/badge/conda--package-page-blue)](https://roitberg.chem.ufl.edu/projects/conda-packages-uf-gainesville)
![PyPI](https://img.shields.io/pypi/v/torchani.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/torchani.svg)

Checks:

[![CodeFactor](https://www.codefactor.io/repository/github/aiqm/torchani/badge/master)](https://www.codefactor.io/repository/github/aiqm/torchani/overview/master)
[![Actions Status](https://github.com/roitberg-group/torchani_sandbox/workflows/flake8/badge.svg)](https://github.com/roitberg-group/torchani_sandbox/actions)
[![Actions Status](https://github.com/roitberg-group/torchani_sandbox/workflows/clang-format/badge.svg)](https://github.com/roitberg-group/torchani_sandbox/actions)
[![Actions Status](https://github.com/roitberg-group/torchani_sandbox/workflows/mypy/badge.svg)](https://github.com/roitberg-group/torchani_sandbox/actions)
[![Actions Status](https://github.com/roitberg-group/torchani_sandbox/workflows/unittests/badge.svg)](https://github.com/roitberg-group/torchani_sandbox/actions)
[![Actions Status](https://github.com/roitberg-group/torchani_sandbox/actions/workflows/build-docker.yml/badge.svg)](https://github.com/roitberg-group/torchani_sandbox/actions/workflows/build-docker.yml)
[![Actions Status](https://github.com/aiqm/torchani/workflows/docs/badge.svg)](https://github.com/aiqm/torchani/actions)
[![Actions Status](https://github.com/aiqm/torchani/workflows/runnable-submodules/badge.svg)](https://github.com/aiqm/torchani/actions)
[![Actions Status](https://github.com/roitberg-group/torchani_sandbox/workflows/tools/badge.svg)](https://github.com/roitberg-group/torchani_sandbox/actions)

Deploy:

[![Actions Status](https://github.com/aiqm/torchani/workflows/deploy-docs/badge.svg)](https://github.com/aiqm/torchani/actions)
[![Actions Status](https://github.com/aiqm/torchani/workflows/deploy-pypi/badge.svg)](https://github.com/aiqm/torchani/actions)

TorchANI is a pytorch implementation of ANI. It is currently under alpha
release, which means, the API is not stable yet. If you find a bug of TorchANI,
or have some feature request, feel free to open an issue on GitHub, or send us
a pull request.

<img src=https://raw.githubusercontent.com/aiqm/torchani/master/logo2.png width=500/>

TorchANI is tested against (usually) the latest PyTorch version

## Install TorchANI

### Conda installation

TODO: Support this again (currently only building from source works)

To install TorchANI using conda run:

```bash
conda create -n ani python=3.8
conda activate ani
conda install -c https://roitberg.chem.ufl.edu/projects/conda-packages-uf-gainesville -c pytorch -c nvidia -c defaults -c conda-forge sandbox
```

Or in a single command:

```bash
conda create -n ani -c https://roitberg.chem.ufl.edu/projects/conda-packages-uf-gainesville -c pytorch -c nvidia -c defaults -c conda-forge sandbox python=3.8
```

Notes:

- We are hosting the packages only for internal usage at
  https://roitberg.chem.ufl.edu/projects/conda-packages-uf-gainesville
- The `conda install` command could also be used for your own conda environment
  or could be used to update to the latest nightly version.
- In the case where multiple updates has been released within a day, you may
  need to add a `--force-reinstall` flag instead of waiting for the next
  nightly update.

### Using pip

TODO: Support this again (currently only building from source works)

## Build TorchANI from source

To install TorchANI from GitHub run the following:

```bash
# Clone the repo and cd to the directory
git clone --recurse-submodules https://github.com/roitberg-group/torchani_sandbox.git
cd ./torchani_sandbox

# Create a conda (or mamba) environment
# Note that environment.yaml contains many optional dependencies needed to
# build the extensions, build the documentation, and run tests and tools
# You can skip these if you are not planning to do that
conda env create -f ./environment.yaml

# Install torchani
pip install -v --no-deps --no-build-isolation -e .

# Install CUDA and C++ extensions (optional)
# Use the "--ext-all-sms" flag instead of "--ext" if you want to build for all GPUs
pip install -v --no-deps --no-build-isolation -e . --global-option="--ext"

# Download files needed for testing and building the docs (optional)
bash ./download.sh

# Build the documentation (optional)
sphix-build docs docs-build

# Manually run unit tests (optional)
cd ./tests
pytest -v .
```

Usually this process works for most use cases, but for more details regarding
building the CUDA and C++ extensions refer to [TorchANI CSRC](torchani/csrc).

## CUDA / C++ extensions

A CUDA extension for speeding up AEV calculations and a C++ extension for
parallelizing networks (MNP or Multi Net Parallel) using MPI are compiled by
default in the conda build, and have to be built manually if installed from
github.

## Command Line Interface

After installation, there will be an executable script (torchani) available on
you path, which contain some builtin utilities. Check usage by calling
``torchani --help``.

## Citations

Please cite the following paper if you use TorchANI:

- Xiang Gao, Farhad Ramezanghorbani, Olexandr Isayev, Justin S. Smith, and
  Adrian E. Roitberg. *TorchANI: A Free and Open Source PyTorch Based Deep
  Learning Implementation of the ANI Neural Network Potentials*. Journal of
  Chemical Information and Modeling 2020 60 (7), 3408-3415,
  [![DOI for Citing](https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.0c00451-green.svg)](https://doi.org/10.1021/acs.jcim.0c00451)

[![JCIM Cover](https://pubs.acs.org/na101/home/literatum/publisher/achs/journals/content/jcisd8/2020/jcisd8.2020.60.issue-7/jcisd8.2020.60.issue-7/20200727/jcisd8.2020.60.issue-7.largecover.jpg)](https://pubs.acs.org/toc/jcisd8/60/7)

- Please refer to [isayev/ASE_ANI](https://github.com/isayev/ASE_ANI) for ANI
  model references.

## Documentation repository

(Note that this only applies for the public repo)

If you opened a pull request, you could see your generated documents at
https://aiqm.github.io/torchani-test-docs/ after you `docs` check succeed. Keep
in mind that this repository is only for the purpose of convenience of
development, and only keeps the latest push. The CI runing for other pull
requests might overwrite this repository. You could rerun the `docs` check to
overwrite this repo to your build.

## Notes to developers

- Never commit to the master branch directly. If you need to change something,
  create a new branch and submit a PR on GitHub.
- All the tests on GitHub must pass before your PR can be merged.
- Code review is required before merging a pull request.
