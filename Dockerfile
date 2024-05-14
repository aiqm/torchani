FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel
WORKDIR /torchani_sandbox

# NGC image forces TF32, but we disable this
# TODO: Do we need this for the pytorch images?
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
ENV CUDA_HOME=/usr/local/cuda/
ENV PATH=${CUDA_HOME}/bin:$PATH
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Install packages to run download and to detect version using setuptools-scm
RUN apt-get update && apt-get install -y git wget unzip

# Download and unzip test data
COPY ./download.sh .
RUN ./download.sh

# Copy requirements files into the image
COPY *_requirements.txt .

# Install requirements (note that only tests_requirements are needed for unit-tests)
RUN pip install twine wheel pytest \
    -r tests_requirements.txt \
    -r tools_requirements.txt \
    -r docs_requirements.txt

# Copy all other repo files into the image
COPY . /torchani_sandbox

# Install torchani and extensions
RUN pip install -v --no-build-isolation --editable . \
    && pip install -v --no-build-isolation --editable . --global-option="--ext"
