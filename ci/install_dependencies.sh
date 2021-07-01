#!/bin/bash
set -ex

pip install --upgrade pip
pip install twine wheel
# clean pip cache if it's larger than 10GB
file_size=$(du -bs $(pip cache dir) | cut -f1)
du -bsh $(pip cache dir)
if [[ $file_size -gt 10737418240 ]]; then pip cache purge; fi
pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html
pip install -r test_requirements.txt
pip install -r docs_requirements.txt
