#!/bin/bash

python -m pip install --upgrade pip
pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
pip install pytorch-ignite --no-deps
