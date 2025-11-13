<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/aiqm/torchani/main/front-logo-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/aiqm/torchani/main/front-logo-light.png">
  <img alt="TorchANI 2 logo" src="https://raw.githubusercontent.com/aiqm/torchani/main/torchani-logo-light.png">
</picture>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![PyPI](https://img.shields.io/pypi/v/torchani.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/torchani.svg)
[![CI](https://github.com/aiqm/torchani/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/aiqm/torchani/actions/workflows/ci.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

TorchANI 2.0 is an open-source library that supports training, development, and research
of ANI-style neural network interatomic potentials. It was originally developed and is
currently maintained by the Roitberg group. For information and examples, please see the
comprehensive [documentation](https://aiqm.github.io/torchani/).

⚠️  **Important**: If you were using a previous version of TorchANI and your code does not work with
TorchANI 2.0 check out the [migration guide](https://aiqm.github.io/torchani/migrating-to-2.html#torchani-migrating), there
are very few breaking changes, most code should work with minimal modifications. If
you can't figure something out please open a GitHub issue, we are here to help!
In the meantime, you can pin torchani to version 2.2.4 (pip install 'torchani==2.2.4'), which does not
have breaking changes. If you require the old state dicts of ANI models you can access
them by calling `.legacy_state_dict()` instead of `.state_dict()`

If you find a bug in TorchANI 2.0, or have some feature request, also feel free to open
a GitHub issue. TorchANI 2.0 is currently tested against PyTorch 2.8 and CUDA 12.8

If you find this work useful please cite the following articles:
- *TorchANI 2.0: An extensible, high performance library for the design, training, and use of NN-IPs* <br>
    https://pubs.acs.org/doi/10.1021/acs.jcim.5c01853
- *TorchANI: A Free and Open Source PyTorch-Based Deep Learning Implementation of the ANI Neural Network Potentials* <br>
    https://pubs.acs.org/doi/10.1021/acs.jcim.0c00451

To run molecular dynamics (full ML or ML/MM) with [Amber (sander or
pmemd)](https://ambermd.org/AmberTools.php) check out
[the TorchANI-Amber interface](https://github.com/roitberg-group/torchani-amber),
and the relevant publications:
- *TorchANI-Amber: Bridging neural network potentials and classical biomolecular simulations* <br>
    https://doi.org/10.1021/acs.jpcb.5c05725 
- *Advancing Multiscale Molecular Modeling with Machine Learning-Derived Electrostatics* <br>
    For the ML/MM capabilities: https://pubs.acs.org/doi/10.1021/acs.jctc.4c01792

## Installation

We recommend installing `torchani` inside a `conda|mamba` environment, or a `venv`.

⚠️  **Important**: *Please install torchani with pip if you want the latest version, even
if using a conda env since the torchani conda package is currently not maintained.*

We also recommended you first install a specific torch version, with a specific CUDA
toolkit backend, for example:

```bash
pip install torch==2.8 --index-url https://download.pytorch.org/whl/cu129
```

for the version with CUDA 12.9. This is not strictly required, but is easier if you want
to control these versions. Note that TorchANI requires PyTorch >= 2.0.

Afterwards:

```bash
pip install torchani
```

TorchANI 2.0 provides C++ and CUDA extensions for accelerated computation of descriptors
and network inference. In order to build the extensions, first install the CUDA Toolkit
appropriate for your PyTorch version. You can follow the instructions in [the official
documentation](https://developer.nvidia.com/cuda-toolkit) for your system.
Alternatively, if you are using a conda environment, you can install the toolkit with
`conda install nvidia::cuda-toolkit=12.9`

After this, run:

```bash
ani build-extensions
```

By default the extensions are built for all detected SMs. If you want to build the
extensions for specific SMs run for instance:

```bash
ani build-extensions --sm 8.0 --sm 8.9 
```

### From source (GitHub repo)

To build and install TorchANI directly from the GitHub repo do the following:

```bash
# Clone the repo and cd to the directory
git clone https://github.com/aiqm/torchani.git
cd ./torchani

# Create a conda (or mamba) environment
# Note that environment.yaml contains many optional dependencies needed to
# build the compiled extensions, build the documentation, and run tests and tools
# You can comment these out if you are not planning to do that
conda env create -f ./environment.yaml
```

Instead of using a `conda` environment you can use a python `venv`,
and install the torchani optional dependencies
running `pip install -r dev_requirements.txt`.

```bash
pip install --no-deps -v -e .
```

Afterwards you can install the extensions with:

```bash
ani build-extensions
```

After this you can perform some optional steps if you installed the required
dev dependencies:

```bash
# Download files needed for testing and building the docs (optional)
bash ./download-dev-data.sh

# Build the documentation (optional)
sphinx-build docs/src docs/build

# Manually run unit tests (optional)
cd ./tests
pytest -v .
```

This process works for most use cases, for more details regarding building
the CUDA and C++ extensions refer to [TorchANI CSRC](torchani/csrc).

#### From source in macOS

There is no CUDA support on `macOS` and TorchANI is **untested** with
Apple Metal Performance Shaders (MPS). The `environment.yaml` file needs
slight modifications if installing on `macOS`. Please consult the corresponding
file and modify it before creating the `conda` environment.

## GPU support

TorchANI 2.0 can be run in CUDA-enabled GPUs. This is **highly recommended** unless
doing simple debugging or tests. If you don't run TorchANI on a GPU, expect degraded
performance. TorchANI is **untested** with AMD GPUs (ROCm | HIP).

## Command Line Interface

TorchANI 2.0 provides an executable script, `ani`, with some utilities. Check usage by
calling ``torchani --help``.

## Building the TorchANI conda package (for developers)

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
