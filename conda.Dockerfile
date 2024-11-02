# This image has ubuntu 22.0, cuda 11.8, cudnn 9, python 3.11.10, pytorch 2.5.1
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel
WORKDIR /repo

# Set cuda env vars
ENV CUDA_HOME=/usr/local/cuda/
ENV PATH=${CUDA_HOME}/bin:$PATH
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install dependencies to:
# Get the program version from version control (git, needed by setuptools-scm)
# Download test data and maybe CUB (wget, unzip)
# Build C++/CUDA extensions faster (ninja-build)
# Upload pkg to internal server (rsync)
RUN apt update && apt install -y wget git unzip ninja-build rsync

# Install requirements to build conda pkg (first activate conda base env)
RUN \
    . /opt/conda/etc/profile.d/conda.sh \
    && conda activate \
    && conda update -n base conda \
    && conda install conda-build conda-verify anaconda-client

# Copy all of the repo files
COPY . /repo

# Initialize a git repo and create dummy tag for setuptools scm
RUN \
    git config --global user.email "user@domain.com" \
    && git config --global user.name "User" \
    && git init \
    && git add . \
    && git commit -m "Initial commit" \
    && git tag -a "3.0-dev" -m "Version 3.0-dev"

# Build conda pkg locally
RUN \
    mkdir ./conda-pkgs/ \
    && . /opt/conda/etc/profile.d/conda.sh \
    && conda activate \
    && conda build -c nvidia -c pytorch -c conda-forge \
        --no-anaconda-upload --output-folder ./conda-pkgs/ ./recipe

# Usage: To upload pkg to internal server
# --build-arg=INTERNAL_RELEASE=1 (or --build-arg=INTERNAL_RELEASE="true")
# DOCKER_PVTKEY must be passed as a secret
ARG INTERNAL_RELEASE=0
RUN --mount=type=secret,id=DOCKER_PVTKEY \
if [ "${INTERNAL_RELEASE}" = "1" ] || [ "${INTERNAL_RELEASE}" = "true" ] ; then \
    rsync -av --delete \
        -e "ssh -i /run/secrets/DOCKER_PVTKEY -o StrictHostKeyChecking=no -o ConnectTimeout=10" \
        ./conda-pkgs/ "ipickering@moria.chem.ufl.edu:/data/conda-pkgs/" ; \
else \
    printf "Not uploading to internal server" ; \
fi

# Usage: To upload pkg to anaconda.org
# --build-arg=PUBLIC_RELEASE=1 (or --build-arg=PUBLIC_RELEASE="true")
# CONDA_TOKEN must be passed as a secret
ARG PUBLIC_RELEASE=0
RUN --mount=type=secret,id=CONDA_TOKEN \
if [ "${PUBLIC_RELEASE}" = "1" ] || [ "${PUBLIC_RELEASE}" = "truej" ]; then \
    CONDA_TOKEN=`cat /run/secrets/CONDA_TOKEN` \
    && anaconda --token "${CONDA_TOKEN}" \
        upload --user roitberg-group --force ./conda-pkgs/linux-64/*.tar.gz ; \
else \
    printf "Not uploading to anaconda server" ; \
fi
