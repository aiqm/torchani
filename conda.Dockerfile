# This image has ubuntu 22.0, cuda 11.8, cudnn 8.7, python 3.10, pytorch 2.3.0
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel
WORKDIR /repo

# Set cuda env vars
ENV CUDA_HOME=/usr/local/cuda/
ENV PATH=${CUDA_HOME}/bin:$PATH
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install dependencies to:
# Get the program version from version control (git, needed by setuptools-scm)
# Download test data and maybe CUB (wget, unzip)
# Build C++/CUDA extensions faster (ninja-build)
RUN apt update && apt install -y wget git unzip ninja-build

# Install requirements to build conda pkg (first activate conda base env)
RUN \
    . /opt/conda/etc/profile.d/conda.sh \
    && conda activate \
    && conda update -n base conda \
    && conda install conda-build conda-verify anaconda-client

# Copy all of the repo files
COPY . /repo

# Create dummy tag for setuptools scm
RUN \
    git config --global user.email "user@domain.com" \
    && git config --global user.name "User" \
    && git tag -a "v2.3" -m "Version v2.3"
