#!/bin/bash

pip install --upgrade pip
pip install twine wheel
# pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu110/torch_nightly.html

# Upstream have bug on TorchScript CUDA extension support
# this is pytorch build with that bug fixed. This is only temporarily,
# and will be replaced by upstream pytorch once the fix is merged
wget --verbose http://fremont.ipv6.ai/torch-1.8.0a0-cp38-cp38-linux_x86_64.whl
pip install torch-1.8.0a0-cp38-cp38-linux_x86_64.whl

pip install -r test_requirements.txt
pip install -r docs_requirements.txt
