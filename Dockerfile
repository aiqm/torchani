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
RUN apt update && apt install -y wget git unzip ninja-build rsync

# Download test data
COPY ./download.sh .
RUN ./download.sh

# Copy pip optional dependencies file
COPY dev_requirements.txt .

# Install optional dependencies
RUN pip install -r dev_requirements.txt

# Copy all other necessary repo files
COPY . /repo

# Initialize a git repo and create dummy tag for setuptools scm
RUN \
    git config --global user.email "user@domain.com" \
    && git config --global user.name "User" \
    && git init \
    && git add . \
    && git commit -m "Initial commit" \
    && git tag -a "v2.3" -m "Version v2.3"

# Install torchani + core requirements (+ extensions if BUILD_EXT build arg is provided)
# Usage:
# BUILD_EXT=0 -> Don't build extensions
# BUILD_EXT=all-sms -> Build extensions for all sms
# BUILD_EXT=smMajorMinor (e.g. BUILD_EXT=sm86)-> Build for specific Major.Minor SM
ARG BUILD_EXT=0
RUN \
if [ "$BUILD_EXT" = "0" ]; then \
    pip install -v . ; \
else \
    pip install \
        --no-build-isolation \
        --config-settings=--global-option=ext-"${BUILD_EXT}" \
        -v . ; \
fi
