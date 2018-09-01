# -*- coding: utf-8 -*-
"""AEVs for a dataset can be precomputed by invoking
``python -m torchani.data.cache-aev``, this would dump the dataset and
computed aevs. Use the ``-h`` option for help.
"""

import os
import torch
from .. import aev, neurochem
from . import BatchedANIDataset
import pickle


if __name__ == '__main__':
    import argparse
    builtin = neurochem.Builtins()
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
    parser.add_argument('--dtype',
                        help='Data type',
                        default=str(torch.get_default_dtype()).split('.')[1])
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('-d', '--device', help='Device for training',
                        default=default_device)
    parser.add_argument('--no-shuffle', help='Whether to shuffle dataset',
                        dest='shuffle', action='store_false')
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false',
                        help='Whether to use tqdm to display progress')
    parser = parser.parse_args()

    # if output directory does not exist, then create it
    if not os.path.exists(parser.output):
        os.makedirs(parser.output)

    device = torch.device(parser.device)
    consts = neurochem.Constants(parser.constfile)
    aev_computer = aev.AEVComputer(**consts).to(device)
    dataset = BatchedANIDataset(parser.dataset, consts.species_to_tensor,
                                parser.batchsize, shuffle=parser.shuffle,
                                properties=parser.properties, device=device,
                                dtype=getattr(torch, parser.dtype))

    # dump out the dataset
    filename = os.path.join(parser.output, 'dataset')
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

    if parser.tqdm:
        import tqdm
        indices = tqdm.trange(len(dataset))
    else:
        indices = range(len(dataset))
    for i in indices:
        input_, _ = dataset[i]
        aevs = [aev_computer(j) for j in input_]
        aevs = [(x.cpu(), y.cpu()) for x, y in aevs]
        filename = os.path.join(parser.output, '{}'.format(i))
        with open(filename, 'wb') as f:
            pickle.dump(aevs, f)
