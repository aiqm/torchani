#!/bin/bash

# Download the data
mkdir dev-data
echo "Downloading data ..."
wget --no-verbose https://huggingface.co/datasets/roitberg-group/torchani-tests-pickled-files/resolve/main/download.zip
echo "Extracting data ..."
unzip -q download.zip -d dev-data/hf-data || [[ $? == 2 ]]  # unzip return 2 for dropbox created zip file
rm download.zip
