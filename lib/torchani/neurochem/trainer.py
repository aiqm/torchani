# -*- coding: utf-8 -*-
"""Besides running NeuroChem trainer by programming, we can also run it by
``python -m torchani.neurochem.trainer``, use the ``-h`` option for help.
"""

import torch
from . import Trainer


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path',
                        help='Path of the training config file `.ipt`')
    parser.add_argument('training_path',
                        help='Path of the training set, can be a hdf5 file \
                              or a directory containing hdf5 files')
    parser.add_argument('validation_path',
                        help='Path of the validation set, can be a hdf5 file \
                              or a directory containing hdf5 files')
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('-d', '--device', help='Device for training',
                        default=default_device)
    parser.add_argument('--tqdm', help='Whether to enable tqdm',
                        dest='tqdm', action='store_true')
    parser.add_argument('--tensorboard',
                        help='Directory to store tensorboard log files',
                        default=None)
    parser.add_argument('--checkpoint_name',
                        help='Name of checkpoint file',
                        default='model.pt')
    args = parser.parse_args()

    d = torch.device(args.device)
    trainer = Trainer(args.config_path, d, args.tqdm, args.tensorboard, args.checkpoint_name)
    trainer.load_data(args.training_path, args.validation_path)
    trainer.run()
