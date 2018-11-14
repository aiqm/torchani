# -*- coding: utf-8 -*-
"""Tools for loading, shuffling, and batching ANI datasets"""

from torch.utils.data import Dataset
from os.path import join, isfile, isdir
import os
from ._pyanitools import anidataloader
import torch
from .. import utils, neurochem, aev
import pickle

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def chunk_counts(counts, split):
    split = [x + 1 for x in split] + [None]
    count_chunks = []
    start = 0
    for i in split:
        count_chunks.append(counts[start:i])
        start = i
    chunk_conformations = [sum([y[1] for y in x]) for x in count_chunks]
    chunk_maxatoms = [x[-1][0] for x in count_chunks]
    return chunk_conformations, chunk_maxatoms


def split_cost(counts, split):
    split_min_cost = 40000
    cost = 0
    chunk_conformations, chunk_maxatoms = chunk_counts(counts, split)
    for conformations, maxatoms in zip(chunk_conformations, chunk_maxatoms):
        cost += max(conformations * maxatoms ** 2, split_min_cost)
    return cost


def split_batch(natoms, species, coordinates):
    # count number of conformation by natoms
    natoms = natoms.tolist()
    counts = []
    for i in natoms:
        if not counts:
            counts.append([i, 1])
            continue
        if i == counts[-1][0]:
            counts[-1][1] += 1
        else:
            counts.append([i, 1])
    # find best split using greedy strategy
    split = []
    cost = split_cost(counts, split)
    improved = True
    while improved:
        improved = False
        cycle_split = split
        cycle_cost = cost
        for i in range(len(counts)-1):
            if i not in split:
                s = sorted(split + [i])
                c = split_cost(counts, s)
                if c < cycle_cost:
                    improved = True
                    cycle_cost = c
                    cycle_split = s
        if improved:
            split = cycle_split
            cost = cycle_cost
    # do split
    start = 0
    species_coordinates = []
    chunk_conformations, _ = chunk_counts(counts, split)
    for i in chunk_conformations:
        s = species
        end = start + i
        s = species[start:end, ...]
        c = coordinates[start:end, ...]
        s, c = utils.strip_redundant_padding(s, c)
        species_coordinates.append((s, c))
        start = end
    return species_coordinates


class BatchedANIDataset(Dataset):
    """Load data from hdf5 files, create minibatches, and convert to tensors.

    This is already a dataset of batches, so when iterated, a batch rather
    than a single data point will be yielded.

    Since each batch might contain molecules of very different sizes, putting
    the whole batch into a single tensor would require adding ghost atoms to
    pad everything to the size of the largest molecule. As a result, huge
    amount of computation would be wasted on ghost atoms. To avoid this issue,
    the input of each batch, i.e. species and coordinates, are further divided
    into chunks according to some heuristics, so that each chunk would only
    have molecules of similar size, to minimize the padding required.

    So, when iterating on this dataset, a tuple will be yielded. The first
    element of this tuple is a list of (species, coordinates) pairs. Each pair
    is a chunk of molecules of similar size. The second element of this tuple
    would be a dictionary, where the keys are those specified in the argument
    :attr:`properties`, and values are a single tensor of the whole batch
    (properties are not splitted into chunks).

    Splitting batch into chunks leads to some inconvenience on training,
    especially when using high level libraries like ``ignite``. To overcome
    this inconvenience, :class:`torchani.ignite.Container` is created for
    working with ignite.

    Arguments:
        path (str): Path to hdf5 files. If :attr:`path` is a file, then that
            file would be loaded using `pyanitools.py`_. If :attr:`path` is
            a directory, then all files with suffix `.h5` or `.hdf5` will be
            loaded.
        species_tensor_converter (:class:`collections.abc.Callable`): A
            callable that convert species in the format of list of strings
            to 1D tensor.
        batch_size (int): Number of different 3D structures in a single
            minibatch.
        shuffle (bool): Whether to shuffle the whole dataset.
        properties (list): List of keys in the dataset to be loaded.
            ``'species'`` and ``'coordinates'`` are always loaded and need not
            to be specified here.
        transform (list): List of :class:`collections.abc.Callable` that
            transform the data. Callables must take species, coordinates,
            and properties of the whole dataset as arguments, and return
            the transformed species, coordinates, and properties.
        dtype (:class:`torch.dtype`): dtype of coordinates and properties to
            to convert the dataset to.
        device (:class:`torch.dtype`): device to put tensors when iterating.

    .. _pyanitools.py:
        https://github.com/isayev/ASE_ANI/blob/master/lib/pyanitools.py
    """

    def __init__(self, path, species_tensor_converter, batch_size,
                 shuffle=True, properties=['energies'], transform=(),
                 dtype=torch.get_default_dtype(), device=default_device):
        super(BatchedANIDataset, self).__init__()
        self.properties = properties
        self.device = device

        # get name of files storing data
        files = []
        if isdir(path):
            for f in os.listdir(path):
                f = join(path, f)
                if isfile(f) and (f.endswith('.h5') or f.endswith('.hdf5')):
                    files.append(f)
        elif isfile(path):
            files = [path]
        else:
            raise ValueError('Bad path')

        # load full dataset
        species_coordinates = []
        properties = {k: [] for k in self.properties}
        for f in files:
            for m in anidataloader(f):
                s = species_tensor_converter(m['species'])
                c = torch.from_numpy(m['coordinates']).to(torch.double)
                species_coordinates.append((s, c))
                for i in properties:
                    p = torch.from_numpy(m[i]).to(torch.double)
                    properties[i].append(p)
        species, coordinates = utils.pad_coordinates(species_coordinates)
        for i in properties:
            properties[i] = torch.cat(properties[i])

        # shuffle if required
        conformations = coordinates.shape[0]
        if shuffle:
            indices = torch.randperm(conformations)
            species = species.index_select(0, indices)
            coordinates = coordinates.index_select(0, indices)
            for i in properties:
                properties[i] = properties[i].index_select(0, indices)

        # do transformations on data
        for t in transform:
            species, coordinates, properties = t(species, coordinates,
                                                 properties)

        # convert to desired dtype
        species = species
        coordinates = coordinates.to(dtype)
        for k in properties:
            properties[k] = properties[k].to(dtype)

        # split into minibatches, and strip redundant padding
        natoms = (species >= 0).to(torch.long).sum(1)
        batches = []
        num_batches = (conformations + batch_size - 1) // batch_size
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, conformations)
            natoms_batch = natoms[start:end]
            # sort batch by number of atoms to prepare for splitting
            natoms_batch, indices = natoms_batch.sort()
            species_batch = species[start:end, ...].index_select(0, indices)
            coordinates_batch = coordinates[start:end, ...] \
                .index_select(0, indices)
            properties_batch = {
                k: properties[k][start:end, ...].index_select(0, indices)
                .to(self.device) for k in properties
            }
            # further split batch into chunks
            species_coordinates = split_batch(natoms_batch, species_batch,
                                              coordinates_batch)
            batch = species_coordinates, properties_batch
            batches.append(batch)
        self.batches = batches

    def __getitem__(self, idx):
        species_coordinates, properties = self.batches[idx]
        species_coordinates = [(s.to(self.device), c.to(self.device))
                               for s, c in species_coordinates]
        return species_coordinates, properties

    def __len__(self):
        return len(self.batches)


class AEVCacheLoader(Dataset):
    """Build a factory for AEV.

    The computation of AEV is the most time consuming part during training.
    Since during training, the AEV never changes, it is not hard to see that,
    If we have a fast enough storage (this is usually the case for good SSDs,
    but not for HDD), we could cache the computed AEVs into disk and load it
    rather than compute it from scratch everytime we use it.

    Arguments:
        disk_cache (str): Directory storing disk caches.
    """

    def __init__(self, disk_cache=None):
        super(AEVCacheLoader, self).__init__()
        self.disk_cache = disk_cache

        # load dataset from disk cache
        dataset_path = os.path.join(disk_cache, 'dataset')
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)

    def __getitem__(self, index):
        _, output = self.dataset.batches[index]
        aev_path = os.path.join(self.disk_cache, str(index))
        with open(aev_path, 'rb') as f:
            species_aevs = pickle.load(f)
        return species_aevs, output

    def __len__(self):
        return len(self.dataset)


builtin = neurochem.Builtins()


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
        filename = os.path.join(output, '{}'.format(i))
        with open(filename, 'wb') as f:
            pickle.dump(aevs, f)


__all__ = ['BatchedANIDataset', 'AEVCacheLoader', 'cache_aev']
