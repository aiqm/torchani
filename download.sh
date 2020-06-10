#!/bin/bash

# download the data
wget https://www.dropbox.com/sh/2c8zdqc1hrqsgwy/AAD6l24ngoiFa6DRapF6HPk5a/ -O download.zip
unzip download.zip -d download || [[ $? == 2 ]]  # unzip return 2 for dropbox created zip file
