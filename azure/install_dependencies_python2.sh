#!/bin/bash

python -m pip install --upgrade pip
pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
pip install tqdm pyyaml future pkbar
pip2 install 'ase<=3.17'
