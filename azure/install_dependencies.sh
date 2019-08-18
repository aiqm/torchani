#!/bin/bash

python -m pip install --upgrade pip
pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
pip install tqdm pyyaml future
pip install pkbar
pip install 'ase<=3.17'