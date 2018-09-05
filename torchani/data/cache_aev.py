# -*- coding: utf-8 -*-
"""AEVs for a dataset can be precomputed by invoking
``python -m torchani.data.cache_aev``, this would dump the dataset and
computed aevs. Use the ``-h`` option for help.
"""

import torch
from . import cache_aev, builtin, default_device


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output',
                        help='Path of the output directory')
    parser.add_argument('dataset',
                        help='Path of the dataset, can be a hdf5 file \
                              or a directory containing hdf5 files')
    parser.add_argument('batchsize', help='batch size', type=int)
    parser.add_argument('--constfile',
                        help='Path of the constant file `.params`',
                        default=builtin.const_file)
    parser.add_argument('--properties', nargs='+',
                        help='Output properties to load.`',
                        default=['energies'])
    default_dtype = str(torch.get_default_dtype()).split('.')[1]
    parser.add_argument('--dtype', help='Data type', default=default_dtype)
    parser.add_argument('-d', '--device', help='Device for training',
                        default=default_device)
    parser.add_argument('--no-shuffle', help='Whether to shuffle dataset',
                        dest='shuffle', action='store_false')
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false',
                        help='Whether to use tqdm to display progress')
    parser.add_argument('--subtract-sae', dest='subtract_sae',
                        help='Whether to subtrace self atomic energies',
                        default=None, action='store_true')
    parser.add_argument('--sae-file', help='Path to SAE file',
                        default=builtin.sae_file)
    parser = parser.parse_args()

    cache_aev(parser.output, parser.dataset, parser.batchsize, parser.device,
              parser.constfile, parser.tqdm, shuffle=parser.shuffle,
              properties=parser.properties, dtype=getattr(torch, parser.dtype))
