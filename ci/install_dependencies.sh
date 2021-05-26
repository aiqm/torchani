#!/bin/bash

pip install --upgrade pip
pip install twine wheel
pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html
pip install -r test_requirements.txt
pip install -r docs_requirements.txt
