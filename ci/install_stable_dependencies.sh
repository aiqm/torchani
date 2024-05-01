#!/bin/bash

pip install --upgrade pip
pip install twine wheel
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu113 --upgrade
pip install -r tests_requirements.txt
pip install -r tools_requirements.txt
pip install -r docs_requirements.txt
