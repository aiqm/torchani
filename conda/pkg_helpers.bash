#!/bin/bash
set -ex
# Assume CUDA is installed under /usr/local/cuda-{version}

setup_build_version() {
  if [[ -z "$BUILD_VERSION" ]]; then
    export BUILD_VERSION="$1.dev$(TZ='America/New_York' date "+%Y%m%d")$VERSION_SUFFIX"
  else
    export BUILD_VERSION="$BUILD_VERSION$VERSION_SUFFIX"
  fi
}

setup_conda_pytorch_constraint() {
  export CONDA_PYTORCH_CONSTRAINT="- pytorch 1.9.1 *cuda11.1*"
}

setup_conda_cudatoolkit_constraint(){
  export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit 11.1.*"
}

setup_cuda_home() {
  export CUDA_HOME=/usr/local/cuda-11.1
  export PATH=${CUDA_HOME}/bin:$PATH
  export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
}
