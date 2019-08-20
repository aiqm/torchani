#!/bin/bash

python -m pip install --upgrade pip
pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html
pip install tqdm pyyaml future pkbar
pip2 install 'ase<=3.17'
