#!/bin/bash

pip install --upgrade pip
pip install twine wheel
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html --upgrade
pip install -r test_requirements.txt
pip install -r docs_requirements.txt
