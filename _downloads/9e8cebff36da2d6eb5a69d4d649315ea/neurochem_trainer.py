# -*- coding: utf-8 -*-
"""
.. _neurochem-training:

Train Neural Network Potential From NeuroChem Input File
========================================================

This example shows how to use TorchANI's NeuroChem trainer to read and run
NeuroChem's training config file to train a neural network potential.
"""

###############################################################################
# To begin with, let's first import the modules we will use:
import torchani
import torch
import os
import sys
import tqdm

###############################################################################
# Now let's setup path for the dataset and NeuroChem input file. Note that
# these paths assumes the user run this script under the ``examples`` directory
# of TorchANI's repository. If you download this script, you should manually
# set the path of these files in your system before this script can run
# successfully. Also note that here for our demo purpose, we set both training
# set and validation set the ``ani_gdb_s01.h5`` in TorchANI's repository. This
# allows this program to finish very quick, because that dataset is very small.
# But this is wrong and should be avoided for any serious training.

try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()
cfg_path = os.path.join(path, '../tests/test_data/inputtrain.ipt')
training_path = os.path.join(path, '../dataset/ani1-up_to_gdb4/ani_gdb_s01.h5')  # noqa: E501
validation_path = os.path.join(path, '../dataset/ani1-up_to_gdb4/ani_gdb_s01.h5')  # noqa: E501

###############################################################################
# We also need to set the device to run the training:
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)


trainer = torchani.neurochem.Trainer(cfg_path, device, True, 'runs')
trainer.load_data(training_path, validation_path)


###############################################################################
# Once everything is set up, running NeuroChem is very easy. We simplify need a
# ``trainer.run()``. But here, in order for sphinx-gallery to be able to
# capture the output of tqdm, let's do some hacking first to make tqdm to print
# its progressbar to stdout.
def my_tqdm(*args, **kwargs):
    return tqdm.tqdm(*args, **kwargs, file=sys.stdout)


trainer.tqdm = my_tqdm

###############################################################################
# Now, let's go!
trainer.run()


###############################################################################
# Alternatively, you can run NeuroChem trainer directly using command line.
# There is no need for programming. Just run the following command for help
# ``python -m torchani.neurochem.trainer -h`` for usage. For this demo, the
# equivalent command is:
cmd = ['python', '-m', 'torchani.neurochem.trainer', '-d', device_str,
       '--tqdm', '--tensorboard', 'runs', cfg_path, training_path,
       validation_path]
print(' '.join(cmd))

###############################################################################
# Now let's invoke this command to see what we get. Again, we redirect stderr
# to stdout simplify for sphinx-gallery to be able to capture it when
# generating this document:
from subprocess import Popen, PIPE  # noqa: E402
print(Popen(cmd, stderr=PIPE).stderr.read().decode('utf-8'))
