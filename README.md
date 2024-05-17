# <img src=https://raw.githubusercontent.com/aiqm/torchani/master/logo1.png width=180/>  Accurate Neural Network Potential on PyTorch

Metrics: (UNTRACKED FOR PRIVATE REPO)

![PyPI](https://img.shields.io/pypi/v/torchani.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/torchani.svg)

CI:

[![unittest workflow](
    https://github.com/roitberg-group/torchani_sandbox/actions/workflows/unittest.yml/badge.svg
)](https://github.com/roitberg-group/torchani_sandbox/actions/workflows/unittest.yml)
[![build-docker workflow](
    https://github.com/roitberg-group/torchani_sandbox/actions/workflows/build-docker.yml/badge.svg
)](https://github.com/roitberg-group/torchani_sandbox/actions/workflows/build-docker.yml)
[![lint workflow](
    https://github.com/roitberg-group/torchani_sandbox/actions/workflows/lint.yml/badge.svg
)](https://github.com/roitberg-group/torchani_sandbox/actions/workflows/lint.yml)
[![tools workflow](
        https://github.com/roitberg-group/torchani_sandbox/actions/workflows/tools.yml/badge.svg
)](https://github.com/roitberg-group/torchani_sandbox/actions/workflows/tools.yml)

Deployment: (STOPPED FOR PRIVATE REPO)

[![Custom conda badge, link to page](
        https://img.shields.io/badge/conda--package-page-blue
)](https://roitberg.chem.ufl.edu/projects/conda-packages-uf-gainesville)
[![conda-release workflow](
    https://github.com/roitberg-group/torchani_sandbox/actions/workflows/conda-release.yml/badge.svg
)](https://github.com/roitberg-group/torchani_sandbox/actions/workflows/conda-release.yml)
[![deploy-docs workflow](
    https://github.com/aiqm/torchani/workflows/deploy-docs/badge.svg
)](https://github.com/aiqm/torchani/actions)
[![deploy-pypi workflow](
    https://github.com/aiqm/torchani/workflows/deploy-pypi/badge.svg
)](https://github.com/aiqm/torchani/actions)

TorchANI is a pytorch implementation of ANI. It is currently under alpha
release, which means, the API is not stable yet. If you find a bug of TorchANI,
or have some feature request, feel free to open an issue on GitHub, or send us
a pull request.

<img src=https://raw.githubusercontent.com/aiqm/torchani/master/logo2.png width=500/>

TorchANI is tested against the (usually) latest PyTorch version

## Install TorchANI

### From Anaconda, using conda (BROKEN)

**TODO**: Support this again (currently only building from source works)

To install TorchANI using conda run:

```bash
conda create -n ani python=3.10
conda activate ani
# The following command is all one line
conda install \
    -c https://roitberg.chem.ufl.edu/projects/conda-packages-uf-gainesville \
    -c pytorch \
    -c nvidia \
    -c defaults \
    -c conda-forge \
    sandbox
```

Notes:

- The conda package is hosted at
  https://roitberg.chem.ufl.edu/projects/conda-packages-uf-gainesville only for
  internal use
- In the case where multiple updates were released within the same day, you may
  need to add a `--force-reinstall` flag instead of waiting for the next
  nightly update.

### From PyPI, using pip (BROKEN)

**TODO**: Support this again (currently only building from source works)

### From source, using conda

To build install TorchANI directly from the GitHub repo run the following:

```bash
# Clone the repo and cd to the directory
git clone --recurse-submodules https://github.com/roitberg-group/torchani_sandbox.git
cd ./torchani_sandbox

# Create a conda (or mamba) environment
# Note that environment-dev.yaml contains many optional dependencies needed to
# build the extensions, build the documentation, and run tests and tools
# You can comment these out if you are not planning to do that
conda env create -f ./environment-dev.yaml

# Install torchani
pip install -v --no-deps --no-build-isolation -e .

# Install CUDA and C++ extensions (optional)
# Use the "--ext-all-sms" flag instead of "--ext" if you want to build for all GPUs
pip install -v --no-deps --no-build-isolation -e . --global-option="--ext"

# Download files needed for testing and building the docs (optional)
bash ./download.sh

# Build the documentation (optional)
sphinx-build docs/src docs/build

# Manually run unit tests (optional)
cd ./tests
pytest -v .
```

Usually this process works for most use cases, but for more details regarding
building the CUDA and C++ extensions refer to [TorchANI CSRC](torchani/csrc).

### From source, using pip

Note: please use a venv if you are installing torchani using `pip`

```bash
# Clone the repo and cd to the directory
git clone --recurse-submodules https://github.com/roitberg-group/torchani_sandbox.git
cd ./torchani_sandbox

# TODO: create a venv
pip install -r ./dev_requirements.txt
pip install -v --no-build-isolation -e .

# In this case you will need to manually download the CUDA Toolkit to build
# the extensions.

pip install -v --no-build-isolation -e . --global-option="--ext"

# All the same optional steps as with conda are possible ...
```

## CUDA / C++ extensions

A CUDA extension for speeding up AEV calculations and a C++ extension for
parallelizing networks (MNP or Multi Net Parallel) using OpenMP are compiled by
default in the conda package. They have to be built manually if installed from
GitHub.

## Command Line Interface

Torchani provides an executable script, `torchani`, with some utilities. Check
usage by calling ``torchani --help``.

## Citations

Please cite the following paper if you use TorchANI:

- Xiang Gao, Farhad Ramezanghorbani, Olexandr Isayev, Justin S. Smith, and
  Adrian E. Roitberg. *TorchANI: A Free and Open Source PyTorch Based Deep
  Learning Implementation of the ANI Neural Network Potentials*. Journal of
  Chemical Information and Modeling 2020 60 (7), 3408-3415,
  [
    ![DOI for Citing](
        https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.0c00451-green.svg
    )
](https://doi.org/10.1021/acs.jcim.0c00451)

[
    ![JCIM Cover](
        https://pubs.acs.org/na101/home/literatum/publisher/achs/journals/content/jcisd8/2020/jcisd8.2020.60.issue-7/jcisd8.2020.60.issue-7/20200727/jcisd8.2020.60.issue-7.largecover.jpg)
    ](
        https://pubs.acs.org/toc/jcisd8/60/7
    )

- Please refer to [isayev/ASE_ANI](https://github.com/isayev/ASE_ANI) for ANI
  model references.

## Documentation (ONLY APPLIES TO PUBLIC REPO)

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
