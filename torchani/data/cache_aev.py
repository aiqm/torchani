# -*- coding: utf-8 -*-
"""AEVs for a dataset can be precomputed by invoking
``python -m torchani.data.cache_aev``, this would dump the dataset and
computed aevs. Use the ``-h`` option for help.
"""

import os
import torch
from .. import aev, neurochem
from . import BatchedANIDataset
import pickle


builtin = neurochem.Builtins()
default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
default_dtype = str(torch.get_default_dtype()).split('.')[1]


def cache_aev(output, dataset_path, batchsize, device=default_device,
              constfile=builtin.const_file, subtract_sae=False,
              sae_file=builtin.sae_file, enable_tqdm=True, **kwargs):
    # if output directory does not exist, then create it
    if not os.path.exists(output):
        os.makedirs(output)

    device = torch.device(device)
    consts = neurochem.Constants(constfile)
    aev_computer = aev.AEVComputer(**consts).to(device)

    if subtract_sae:
        energy_shifter = neurochem.load_sae(sae_file)
        transform = (energy_shifter.subtract_from_dataset,)
    else:
        transform = ()

    dataset = BatchedANIDataset(
        dataset_path, consts.species_to_tensor, batchsize,
        device=device, transform=transform, **kwargs
    )

    # dump out the dataset
    filename = os.path.join(output, 'dataset')
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

    if enable_tqdm:
        import tqdm
        indices = tqdm.trange(len(dataset))
    else:
        indices = range(len(dataset))
    for i in indices:
        input_, _ = dataset[i]
        aevs = [aev_computer(j) for j in input_]
        aevs = [(x.cpu(), y.cpu()) for x, y in aevs]
        filename = os.path.join(output, '{}'.format(i))
        with open(filename, 'wb') as f:
            pickle.dump(aevs, f)


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
