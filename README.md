<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/roitberg-group/torchani_model_zoo/master/logos/torchani-3-logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/roitberg-group/torchani_model_zoo/master/logos/torchani-3-logo-light.svg">
  <img alt="TorchANI 3 logo" src="https://raw.githubusercontent.com/roitberg-group/torchani_model_zoo/master/logos/torchani-3-logo-light.svg">
</picture>

Metrics: (UNTRACKED FOR PRIVATE REPO)

![PyPI](https://img.shields.io/pypi/v/torchani.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/torchani.svg)

CI:

[![ci workflow](https://github.com/roitberg-group/torchani_sandbox/actions/workflows/ci.yaml/badge.svg)](
    https://github.com/roitberg-group/torchani_sandbox/actions/workflows/ci.yaml)
[![conda workflow](https://github.com/roitberg-group/torchani_sandbox/actions/workflows/build-conda-pkg.yaml/badge.svg)](
    https://github.com/roitberg-group/torchani_sandbox/actions/workflows/build-conda-pkg.yaml)

Deployment: (STOPPED FOR PRIVATE REPO)

(TODO: PyPI)
(TODO: anaconda.org)
(TODO: docs)
[![conda page](https://img.shields.io/badge/conda--package-page-blue)](
    https://roitberg.chem.ufl.edu/projects/conda-packages-uf-gainesville)

TorchANI is a pytorch implementation of ANI. If you find a bug of TorchANI, or have some
feature request, feel free to open an issue on GitHub, or directly open a pull request.

We test TorchANI against the (usually) latest PyTorch version.

## Installation

### From the internal Roitberg Group servers, using conda or pip

To install the internal version of TorchANI, hosted in the internal
roitberg-group servers using conda run:

```bash
conda create -n ani python=3.10
conda activate ani
# The following command is all one line, and channels must be specified in that order
conda install \
    -c http://moria.chem.ufl.edu/conda-pkgs \
    -c pytorch \
    -c nvidia \
    -c conda-forge \
    torchani
```

Note that this installation currently includes the compiled extensions by default,
but it doesn't include either the ase module, to use it run also:

```bash
conda install -c conda-forge ase
```

To install using pip: currently unavailable

#### From Anaconda.org, using conda

Currently unavailable

### From PyPI, using pip

Currently unavailable

### From source (GitHub repo), using conda or pip

To build and install TorchANI directly from the GitHub repo do the following:

```bash
# Clone the repo and cd to the directory
git clone https://github.com/roitberg-group/torchani_sandbox.git
cd ./torchani_sandbox

# Create a conda (or mamba) environment
# Note that dev_environment.yaml contains many optional dependencies needed to
# build the compiled extensions, build the documentation, and run tests and tools
# You can comment these out if you are not planning to do that
conda env create -f ./dev_environment.yaml
```

Instead of using a `conda` (or `mamba`) environment you can use a python `venv`,
and install the torchani optional dependencies
running `pip install -r dev_requirements.txt`.

Now you have two options, depending on whether you want to install the torchani
compiled extensions. To install torchani with no compiled extensions run:

```bash
pip install --no-deps -v .
```

To install torchani with the cuAEV and MNP compiled extensions run instead:

```bash
# Use 'ext-all-sms' instead of 'ext' if you want to build for all possible GPUs
pip install --config-settings=--global-option=ext --no-build-isolation --no-deps -v .
```

In both cases you can add the editable, `-e`, flag after the verbose, `-v`,
flag if you want an editable install (for developers). The `-v` flag can of
course be omitted, but it is sometimes handy to have some extra information
about the installation process.

After this you can perform some optional steps if you installed the required
dev dependencies:

```bash
# Download files needed for testing and building the docs (optional)
bash ./download.sh

# Build the documentation (optional)
sphinx-build docs/src docs/build

# Manually run unit tests (optional)
cd ./tests
pytest -v .
```

This process works for most use cases, but for more details regarding building
the CUDA and C++ extensions refer to [TorchANI CSRC](torchani/csrc).

#### From source in macOS

Note that there is no CUDA support on `macOS` and TorchANI is **untested** with
Apple Metal Performance Shaders (MPS). The `dev_environment.yaml` file needs
slight modifications if installing on `macOS`. Please consult the corresponding
file and modify it before creating the `conda` environment.

## GPU support

TorchANI can be run in CUDA-enabled GPUs. This is **highly recommended** unless
doing simple debugging or tests. If you don't run TorchANI on a GPU, expect
highly degraded performance. TorchANI is **untested** with AMD GPUs (ROCm |
HIP).

## CUDA and C++ extensions

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
  [![DOI for Citing](https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.0c00451-green.svg)](
    https://doi.org/10.1021/acs.jcim.0c00451)

[![JCIM Cover](https://pubs.acs.org/cms/10.1021/jcisd8.2020.60.issue-7/asset/17a630cf-2b17-630c-a2b1-a630cfa2b17a/jcisd8.2020.60.issue-7.largecover.jpg)](
    https://pubs.acs.org/toc/jcisd8/60/7)

- Refer to [isayev/ASE_ANI](https://github.com/isayev/ASE_ANI) for ANI model
  references.

## Notes for developers

- Never commit to the master branch directly. If you need to change something,
  create a new branch and submit a PR on GitHub.
- All the tests on GitHub must pass before your PR can be merged.
- Code review is required before merging a pull request.

### Documentation (ONLY APPLIES TO PUBLIC REPO)

If you opened a pull request, you could see your generated documents at
https://aiqm.github.io/torchani-test-docs/ after you `docs` check succeed. Keep
in mind that this repository is only for the purpose of convenience of
development, and only keeps the latest push. The CI runing for other pull
requests might overwrite this repository. You could rerun the `docs` check to
overwrite this repo to your build.

### Details on the conda packages needed to build cuAEV and MNP

The CUDA libraries specified by the pytorch-cuda metapackage are not enough to
build the extensions; the `*-dev` versions with the headers are required. We
also pin the version of `nvcc`, since `pytorch` can't directly compile
extensions with newer `nvcc`. We also pin the version of `cccl` due to torch's
usage of thrust. Explicitly specifying `setuptools` and `setuptools-scm` is
required since extensions have to be built with `--no-build-isolation`.
Finally, `g++` and `gcc` compilers that support `C++17` are required for
compilation (compiler version is pinned to ensure reproducibility). The
required conda pkgs are then (sans the version constraints):

```yaml
  - setuptools
  - setuptools-scm
  - gxx_linux-64
  - gcc_linux-64
  - nvidia::cuda-libraries-dev
  - nvidia::cuda-cccl
  - nvidia::cuda-nvcc
```

### Building the TorchANI conda package

The conda package can be built locally using the recipe in `./recipe`, by running:

```bash
cd ./torchani_sandbox
conda install conda-build conda-verify
mkdir ./conda-pkgs/  # This dir must exist before running conda-build
conda build \
    -c pytorch -c nvidia -c conda-forge \
    --no-anaconda-upload \
    --output-folder ./conda-pkgs/ \
    ./recipe
```

The `meta.yaml` in the recipe assumes that the extensions are built using the
system's CUDA Toolkit, located in `/usr/local/cuda`. If this is not possible,
add the following dependencies to the `host` environment:

- `nvidia::cuda-libraries-dev={{ cuda }}`
- `nvidia::cuda-nvcc={{ cuda }}`
- `nvidia::cuda-cccl={{ cuda }}`

and remove `cuda_home=/usr/local/cuda` from the build script. Note that doing
this may significantly increase build time.

The CI (GitHub Actions Workflow) that tests that the conda pkg builds correctly
runs only:

- on pull requests that contain the string `conda` in the branch name.

The workflow that deploys the conda pkg to the internal server runs only:

- on the default branch, at 00:00:00 every day
- on pull requests that contain both the strings `conda` and `release` in the
  branch name
