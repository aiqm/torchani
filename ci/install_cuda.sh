#!/bin/bash

# command copy-pasted from https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update

# from https://github.com/ptheywood/cuda-cmake-github-actions/blob/master/scripts/actions/install_cuda_ubuntu.sh
sudo apt-get -y install cuda-command-line-tools-11-1 cuda-libraries-dev-11-1
export CUDA_HOME=/usr/local/cuda-11.1
export PATH="$CUDA_HOME/bin:$PATH"
nvcc -V
