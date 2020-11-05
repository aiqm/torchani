#!/bin/bash

pip install --upgrade pip
pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu110/torch_nightly.html
pip install -r test_requirements.txt
