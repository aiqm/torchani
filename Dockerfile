FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel

# NGC image forces TF32, disable this
# TODO: Do we need this for the pytorch images?
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Install packages to run download and to detect version using setuptools-scm
RUN apt-get update && apt-get install -y git wget unzip

# Copy all repo files into the image
COPY . /torchani_sandbox
WORKDIR /torchani_sandbox

ENV CUDA_HOME=/usr/local/cuda/
ENV PATH=${CUDA_HOME}/bin:$PATH
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install torchani and dependencies
RUN pip install twine wheel \
    && pip install -r tests_requirements.txt \
    && pip install -r tools_requirements.txt \
    && pip install -r docs_requirements.txt \
    && pip install pytest \
    && ./download.sh \
    && pip install -v --no-build-isolation --editable . \
    && pip install -v --no-build-isolation --editable . --global-option="--ext"
