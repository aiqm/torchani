# This image has ubuntu 22.0, cuda 11.8, cudnn 8.7, python 3.10, pytorch 2.3.0
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel
WORKDIR /torchani_sandbox

# Set cuda env vars
ENV CUDA_HOME=/usr/local/cuda/
ENV PATH=${CUDA_HOME}/bin:$PATH
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Get dependencies to:
# Download test data (wget)
# Get correct setuptools_scm version (git)
# Build C++/CUDA extensions fast (ninja-build)
RUN apt update && apt install -y git wget unzip ninja-build

# Download test data
COPY ./download.sh .
RUN ./download.sh

# Copy pip optional dependencies file
COPY dev_requirements.txt .

# Install optional dependencies
RUN pip install -r dev_requirements.txt

# Copy all other necessary repo files
COPY . /torchani_sandbox

# Install torchani + core requirements (+ extensions if "ext" build arg is provided)
ARG BUILD_EXT=0
RUN \
if [ "$BUILD_EXT" = "0" ]; then \
    pip install -v --no-build-isolation --editable . ; \
else \
    pip install -v --no-build-isolation --editable . && \
    pip install -v --no-build-isolation --editable . --global-option="--ext"; \
fi
