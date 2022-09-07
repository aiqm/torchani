ARG PYT_VER=22.08
FROM nvcr.io/nvidia/pytorch:$PYT_VER-py3

# environment
# NGC Container forces using TF32, disable it
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Copy files into container
COPY . /torchani_sandbox

# Install modulus and dependencies
RUN cd /torchani_sandbox \
    && pip install twine wheel \
    && pip install -r test_requirements.txt \
    && pip install -r docs_requirements.txt \
    && pip install pytest \
    && ./download.sh \
    && python setup.py install --ext

WORKDIR /torchani_sandbox

