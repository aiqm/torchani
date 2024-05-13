# The conda environment in the nvcr pytorch docker is not working, so we
# us the pytorch docker here
# ARG PYT_VER=22.08
# FROM nvcr.io/nvidia/pytorch:$PYT_VER-py3
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# environment
# NGC Container forces using TF32, disable this
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# install some packages
RUN apt-get update && \
    apt-get install -y git wget unzip

# Copy files into container
COPY . /torchani_sandbox

ENV CUDA_HOME=/usr/local/cuda/
ENV PATH=${CUDA_HOME}/bin:$PATH
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install torchani and dependencies
RUN cd /torchani_sandbox \
    && pip install twine wheel \
    && pip install -r tests_requirements.txt \
    && pip install -r tools_requirements.txt \
    && pip install -r docs_requirements.txt \
    && pip install pytest \
    && ./download.sh \
    && python setup.py install --ext

WORKDIR /torchani_sandbox

