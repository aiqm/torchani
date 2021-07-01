#!/bin/bash
set -ex

# USAGE:
# 1. test
# PYTHON_VERSION=3.8 ./build_conda.sh
# 2. release
# PYTHON_VERSION=3.8 CONDA_TOKEN=TOKEN ./build_conda.sh release

# helper functions
script_dir=$(dirname $(realpath $0))
. "$script_dir/pkg_helpers.bash"

# source root dir
export SOURCE_ROOT_DIR="$script_dir/../"

# setup variables
setup_build_version 2.2
setup_cuda_home
setup_conda_pytorch_constraint
setup_conda_cudatoolkit_constraint
export PACKAGE_NAME=sandbox
export USER=roitberg-group
# default python version is 3.8 if it's not set
PYTHON_VERSION="${PYTHON_VERSION:-3.8}"

# conda-build dependency
conda install conda-build conda-verify anaconda-client -y
export PATH="${CONDA_PREFIX}/bin:${CONDA}/bin:$PATH"  # anaconda bin location
which anaconda

# build package
conda build $CONDA_CHANNEL_FLAGS --no-anaconda-upload --no-copy-test-source-files --python "$PYTHON_VERSION" "$script_dir/torchani"

# upload to anaconda.org if has release_anaconda argument
if [[ $1 == release_anaconda ]]; then
    BUILD_FILE="${CONDA}/conda-bld/linux-64/${PACKAGE_NAME}-${BUILD_VERSION}-py${PYTHON_VERSION//./}_torch1.9.0_cuda11.1.tar.bz2"
    echo $BUILD_FILE
    anaconda -t $CONDA_TOKEN upload -u $USER $BUILD_FILE --force
fi

# upload to roitberg server if has release argument
if [[ $1 == release ]]; then
    BUILD_FILE="${CONDA}/conda-bld/linux-64/${PACKAGE_NAME}-${BUILD_VERSION}-py${PYTHON_VERSION//./}_torch1.9.0_cuda11.1.tar.bz2"
    echo $BUILD_FILE
    mkdir -p /release/conda-packages/linux-64
    cp $BUILD_FILE /release/conda-packages/linux-64
    conda index /release/conda-packages
    chown -R 1003:1003 /release/conda-packages
    apt install rsync -y
    rsync -av -e "ssh -p $SERVER_PORT" /release/conda-packages/ $SERVER_USERNAME@roitberg.chem.ufl.edu:/home/statics/conda-packages/
fi
