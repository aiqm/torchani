#!/bin/bash

pip install --upgrade pip
pip install twine wheel
# pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu110/torch_nightly.html
wget --verbose https://www.dropbox.com/s/f6pf5jzbhut3yu1/torch-1.8.0a0-cp38-cp38-linux_x86_64.whl?dl=0 -o torch.whl
pip install torch.whl

pip install -r test_requirements.txt
pip install -r docs_requirements.txt
