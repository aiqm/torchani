#!/bin/bash
set -ex

# USAGE:
# Build packages of torchani
# 1. test
# PYTHON_VERSION=3.8 PACKAGE=torchani ./build_conda.sh
# 2. release
# PYTHON_VERSION=3.8 PACKAGE=torchani CONDA_TOKEN=TOKEN ./build_conda.sh release

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
export USER=roitberg-group
# default python version is 3.8 if it's not set
PYTHON_VERSION="${PYTHON_VERSION:-3.8}"
# default package is torchani
PACKAGE="${PACKAGE:-torchani}"
export CONDA=$(conda info --base)

# set package-name and channel for torchani
if [[ $PACKAGE == torchani ]]; then
    export PACKAGE_NAME=sandbox
    export CONDA_CHANNEL_FLAGS="-c pytorch -c nvidia -c defaults -c conda-forge"
else
    echo PACKAGE must be torchani
    exit 1
fi

# conda-build dependency
conda install conda-build conda-verify anaconda-client -y
conda install conda-package-handling -y  # Update to newer version, Issue: https://github.com/conda/conda-package-handling/issues/71
export PATH="$PATH:${CONDA_PREFIX}/bin:${CONDA}/bin"  # anaconda bin location
which anaconda

# build package
conda build $CONDA_CHANNEL_FLAGS --no-anaconda-upload --no-copy-test-source-files --python "$PYTHON_VERSION" "$script_dir/$PACKAGE"

# upload to anaconda.org if has release_anaconda argument
if [[ $1 == release_anaconda ]]; then
    BUILD_FILE="${CONDA}/conda-bld/linux-64/${PACKAGE_NAME}-${BUILD_VERSION}-py${PYTHON_VERSION//./}_torch1.13.1_cuda11.6.tar.bz2"
    echo $BUILD_FILE
    anaconda -t $CONDA_TOKEN upload -u $USER $BUILD_FILE --force
fi

# upload to roitberg server if has release argument
if [[ $1 == release ]]; then
    BUILD_FILE="${CONDA}/conda-bld/linux-64/${PACKAGE_NAME}-${BUILD_VERSION}-py${PYTHON_VERSION//./}_torch1.13.1_cuda11.6.tar.bz2"
    echo $BUILD_FILE
    mkdir -p /release/conda-packages/linux-64
    cp $BUILD_FILE /release/conda-packages/linux-64
    rm -rf "${CONDA}/conda-bld/*"                             # remove conda-bld directory
    conda index /release/conda-packages
    chown -R 1003:1003 /release/conda-packages
    apt install rsync -y
    rsync -av --delete -e "ssh -p $SERVER_PORT -o StrictHostKeyChecking=no" /release/conda-packages/ $SERVER_USERNAME@roitberg.chem.ufl.edu:/home/statics/conda-packages/
fi
