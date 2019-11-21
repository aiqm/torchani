#!/bin/bash

pip install --upgrade pip
pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
pip install -r test_requirements.txt
