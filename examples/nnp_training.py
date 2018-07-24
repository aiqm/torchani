import os
import sys
import torch
import pickle
import ignite
import torchani

chunk_size = 256
path = os.path.dirname(os.path.realpath(__file__))
dataset_checkpoint = os.path.join(path, '../dataset-checkpoint.dat')

if len(sys.argv) == 1:
    dataset_path = os.path.join(path, '../dataset')
elif len(sys.argv) == 2:
    dataset_path = sys.argv[1]
else:
    print('Usage:')
    print('python nnp_training.py [dataset path]')
    print('If no dataset path specified, the buildin dataset will be used')
    raise TypeError('Can only take 0 or 1 arguments')

training, validation, testing = torchani.data.maybe_create_checkpoint(
    dataset_checkpoint, dataset_path, chunk_size)