# -*- coding: utf-8 -*-
"""Tools for loading, shuffling, and batching ANI datasets"""

from torch.utils.data import Dataset
from os.path import join, isfile, isdir
import os
from ._pyanitools import anidataloader
import torch
from .. import utils, neurochem, aev, models
import pickle
import numpy as np
from scipy.sparse import bsr_matrix
import warnings

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def chunk_counts(counts, split):
    split = [x + 1 for x in split] + [None]
    count_chunks = []
    start = 0
    for i in split:
        count_chunks.append(counts[start:i])
        start = i
    chunk_molecules = [sum([y[1] for y in x]) for x in count_chunks]
    chunk_maxatoms = [x[-1][0] for x in count_chunks]
    return chunk_molecules, chunk_maxatoms


def split_cost(counts, split):
    split_min_cost = 40000
    cost = 0
    chunk_molecules, chunk_maxatoms = chunk_counts(counts, split)
    for molecules, maxatoms in zip(chunk_molecules, chunk_maxatoms):
        cost += max(molecules * maxatoms ** 2, split_min_cost)
    return cost


def split_batch(natoms, atomic_properties):

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
        for i in range(len(counts) - 1):
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
    chunk_molecules, _ = chunk_counts(counts, split)
    num_chunks = None
    for k in atomic_properties:
        atomic_properties[k] = atomic_properties[k].split(chunk_molecules)
        if num_chunks is None:
            num_chunks = len(atomic_properties[k])
        else:
            assert num_chunks == len(atomic_properties[k])
    chunks = []
    for i in range(num_chunks):
        chunk = {k: atomic_properties[k][i] for k in atomic_properties}
        chunks.append(utils.strip_redundant_padding(chunk))
    return chunks


def load_and_pad_whole_dataset(path, species_tensor_converter, shuffle=True,
                               properties=('energies',), atomic_properties=()):
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
    atomic_properties_ = []
    properties = {k: [] for k in properties}
    for f in files:
        for m in anidataloader(f):
            atomic_properties_.append(dict(
                species=species_tensor_converter(m['species']).unsqueeze(0),
                **{
                    k: torch.from_numpy(m[k]).to(torch.double)
                    for k in ['coordinates'] + list(atomic_properties)
                }
            ))
            for i in properties:
                p = torch.from_numpy(m[i]).to(torch.double)
                properties[i].append(p)
    atomic_properties = utils.pad_atomic_properties(atomic_properties_)
    for i in properties:
        properties[i] = torch.cat(properties[i])

    # shuffle if required
    molecules = atomic_properties['species'].shape[0]
    if shuffle:
        indices = torch.randperm(molecules)
        for i in properties:
            properties[i] = properties[i].index_select(0, indices)
        for i in atomic_properties:
            atomic_properties[i] = atomic_properties[i].index_select(0, indices)
    return atomic_properties, properties


def split_whole_into_batches_and_chunks(atomic_properties, properties, batch_size):
    molecules = atomic_properties['species'].shape[0]
    # split into minibatches
    for k in properties:
        properties[k] = properties[k].split(batch_size)
    for k in atomic_properties:
        atomic_properties[k] = atomic_properties[k].split(batch_size)

    # further split batch into chunks and strip redundant padding
    batches = []
    num_batches = (molecules + batch_size - 1) // batch_size
    for i in range(num_batches):
        batch_properties = {k: v[i] for k, v in properties.items()}
        batch_atomic_properties = {k: v[i] for k, v in atomic_properties.items()}
        species = batch_atomic_properties['species']
        natoms = (species >= 0).to(torch.long).sum(1)

        # sort batch by number of atoms to prepare for splitting
        natoms, indices = natoms.sort()
        for k in batch_properties:
            batch_properties[k] = batch_properties[k].index_select(0, indices)
        for k in batch_atomic_properties:
            batch_atomic_properties[k] = batch_atomic_properties[k].index_select(0, indices)

        batch_atomic_properties = split_batch(natoms, batch_atomic_properties)
        batches.append((batch_atomic_properties, batch_properties))

    return batches


class PaddedBatchChunkDataset(Dataset):

    def __init__(self, atomic_properties, properties, batch_size,
                 dtype=torch.get_default_dtype(), device=default_device):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # convert to desired dtype
        for k in properties:
            properties[k] = properties[k].to(dtype)
        for k in atomic_properties:
            if k == 'species':
                continue
            atomic_properties[k] = atomic_properties[k].to(dtype)

        self.batches = split_whole_into_batches_and_chunks(atomic_properties, properties, batch_size)

    def __getitem__(self, idx):
        atomic_properties, properties = self.batches[idx]
        atomic_properties, properties = atomic_properties.copy(), properties.copy()
        species_coordinates = []
        for chunk in atomic_properties:
            for k in chunk:
                chunk[k] = chunk[k].to(self.device)
            species_coordinates.append((chunk['species'], chunk['coordinates']))
        for k in properties:
            properties[k] = properties[k].to(self.device)
        properties['atomic'] = atomic_properties
        return species_coordinates, properties

    def __len__(self):
        return len(self.batches)


class BatchedANIDataset(PaddedBatchChunkDataset):
    """Same as :func:`torchani.data.load_ani_dataset`. This API has been deprecated."""

    def __init__(self, path, species_tensor_converter, batch_size,
                 shuffle=True, properties=('energies',), atomic_properties=(), transform=(),
                 dtype=torch.get_default_dtype(), device=default_device):
        self.properties = properties
        self.atomic_properties = atomic_properties
        warnings.warn("BatchedANIDataset is deprecated; use load_ani_dataset()", DeprecationWarning)

        atomic_properties, properties = load_and_pad_whole_dataset(
            path, species_tensor_converter, shuffle, properties, atomic_properties)

        # do transformations on data
        for t in transform:
            atomic_properties, properties = t(atomic_properties, properties)

        super().__init__(atomic_properties, properties, batch_size, dtype, device)


def load_ani_dataset(path, species_tensor_converter, batch_size, shuffle=True,
                     rm_outlier=False, properties=('energies',), atomic_properties=(),
                     transform=(), dtype=torch.get_default_dtype(), device=default_device,
                     split=(None,)):
    """Load ANI dataset from hdf5 files, and split into subsets.

    The return datasets are already a dataset of batches, so when iterated, a
    batch rather than a single data point will be yielded.

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
        rm_outlier (bool): Whether to discard the outlier energy conformers
            from a given dataset.
        properties (list): List of keys of `molecular` properties in the
            dataset to be loaded. Here `molecular` means, no matter the number
            of atoms that property always have fixed size, i.e. the tensor
            shape of molecular properties should be (molecule, ...). An example
            of molecular property is the molecular energies. ``'species'`` and
            ``'coordinates'`` are always loaded and need not to be specified
            anywhere.
        atomic_properties (list): List of keys of `atomic` properties in the
            dataset to be loaded. Here `atomic` means, the size of property
            is proportional to the number of atoms in the molecule, i.e. the
            tensor shape of atomic properties should be (molecule, atoms, ...).
            An example of atomic property is the forces. ``'species'`` and
            ``'coordinates'`` are always loaded and need not to be specified
            anywhere.
        transform (list): List of :class:`collections.abc.Callable` that
            transform the data. Callables must take atomic properties,
            properties as arguments, and return the transformed atomic
            properties and properties.
        dtype (:class:`torch.dtype`): dtype of coordinates and properties to
            to convert the dataset to.
        device (:class:`torch.dtype`): device to put tensors when iterating.
        split (list): as sequence of integers or floats or ``None``. Integers
            are interpreted as number of elements, floats are interpreted as
            percentage, and ``None`` are interpreted as the rest of the dataset
            and can only appear as the last element of :class:`split`. For
            example, if the whole dataset has 10000 entry, and split is
            ``(5000, 0.1, None)``, then this function will create 3 datasets,
            where the first dataset contains 5000 elements, the second dataset
            contains ``int(0.1 * 10000)``, which is 1000, and the third dataset
            will contains ``10000 - 5000 - 1000`` elements. By default this
            creates only a single dataset.

    Returns:
        An instance of :class:`torchani.data.PaddedBatchChunkDataset` if there is
        only one element in :attr:`split`, otherwise returns a tuple of the same
        classes according to :attr:`split`.

    .. _pyanitools.py:
        https://github.com/isayev/ASE_ANI/blob/master/lib/pyanitools.py
    """
    atomic_properties_, properties_ = load_and_pad_whole_dataset(
        path, species_tensor_converter, shuffle, properties, atomic_properties)

    molecules = atomic_properties_['species'].shape[0]
    atomic_keys = ['species', 'coordinates', *atomic_properties]
    keys = properties

    # do transformations on data
    for t in transform:
        atomic_properties_, properties_ = t(atomic_properties_, properties_)

    if rm_outlier:
        transformed_energies = properties_['energies']
        num_atoms = (atomic_properties_['species'] >= 0).to(transformed_energies.dtype).sum(dim=1)
        scaled_diff = transformed_energies / num_atoms.sqrt()

        mean = transformed_energies.mean()
        std = transformed_energies.std()
        tol = 15.0 * std + mean

        low_idx = (torch.abs(scaled_diff) < tol).nonzero().squeeze()
        outlier_count = molecules - low_idx.numel()
        # discard outlier energy conformers if exist
        if outlier_count > 0:
            print(f'Note: {outlier_count} outlier energy conformers have been discarded from dataset')
            for key, val in atomic_properties_.items():
                atomic_properties_[key] = val[low_idx]
            for key, val in properties_.items():
                properties_[key] = val[low_idx]

    # compute size of each subset
    split_ = []
    total = 0
    for index, size in enumerate(split):
        if isinstance(size, float):
            size = int(size * molecules)
        if size is None:
            assert index == len(split) - 1
            size = molecules - total
        split_.append(size)
        total += size

    # split
    start = 0
    splitted = []
    for size in split_:
        ap = {k: atomic_properties_[k][start:start + size] for k in atomic_keys}
        p = {k: properties_[k][start:start + size] for k in keys}
        start += size
        splitted.append((ap, p))

    # consturct batched dataset
    ret = []
    for ap, p in splitted:
        ds = PaddedBatchChunkDataset(ap, p, batch_size, dtype, device)
        ds.properties = properties
        ds.atomic_properties = atomic_properties
        ret.append(ds)
    if len(ret) == 1:
        return ret[0]
    return tuple(ret)


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
            for i, sa in enumerate(species_aevs):
                species, aevs = self.decode_aev(*sa)
                species_aevs[i] = (
                    species.to(self.dataset.device),
                    aevs.to(self.dataset.device)
                )
        return species_aevs, output

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def decode_aev(encoded_species, encoded_aev):
        return encoded_species, encoded_aev

    @staticmethod
    def encode_aev(species, aev):
        return species, aev


class SparseAEVCacheLoader(AEVCacheLoader):
    """Build a factory for AEV.

    The computation of AEV is the most time-consuming part of the training.
    AEV never changes during training and contains a large number of zeros.
    Therefore, we can store the computed AEVs as sparse representation and
    load it during the training rather than compute it from scratch. The
    storage requirement for ```'cache_sparse_aev'``` is considerably less
    than ```'cache_aev'```.

    Arguments:
        disk_cache (str): Directory storing disk caches.
    """

    @staticmethod
    def decode_aev(encoded_species, encoded_aev):
        species = torch.from_numpy(encoded_species.todense())
        aevs_np = np.stack([np.array(i.todense()) for i in encoded_aev], axis=0)
        aevs = torch.from_numpy(aevs_np)
        return species, aevs

    @staticmethod
    def encode_aev(species, aev):
        encoded_species = bsr_matrix(species.cpu().numpy())
        encoded_aev = [bsr_matrix(i.cpu().numpy()) for i in aev]
        return encoded_species, encoded_aev


ani1x = models.ANI1x()


def create_aev_cache(dataset, aev_computer, output, progress_bar=True, encoder=lambda *x: x):
    """Cache AEV for the given dataset.

    Arguments:
        dataset (:class:`torchani.data.PaddedBatchChunkDataset`): the dataset to be cached
        aev_computer (:class:`torchani.AEVComputer`): the AEV computer used to compute aev
        output (str): path to the directory where cache will be stored
        progress_bar (bool): whether to show progress bar
        encoder (:class:`collections.abc.Callable`): The callable
            (species, aev) -> (encoded_species, encoded_aev) that encode species and aev
    """
    # dump out the dataset
    filename = os.path.join(output, 'dataset')
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

    if progress_bar:
        import tqdm
        indices = tqdm.trange(len(dataset))
    else:
        indices = range(len(dataset))
    for i in indices:
        input_, _ = dataset[i]
        aevs = [encoder(*aev_computer(j)) for j in input_]
        filename = os.path.join(output, '{}'.format(i))
        with open(filename, 'wb') as f:
            pickle.dump(aevs, f)


def _cache_aev(output, dataset_path, batchsize, device, constfile,
               subtract_sae, sae_file, enable_tqdm, encoder, **kwargs):
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

    dataset = load_ani_dataset(
        dataset_path, consts.species_to_tensor, batchsize,
        device=device, transform=transform, **kwargs
    )

    create_aev_cache(dataset, aev_computer, output, enable_tqdm, encoder)


def cache_aev(output, dataset_path, batchsize, device=default_device,
              constfile=ani1x.const_file, subtract_sae=False,
              sae_file=ani1x.sae_file, enable_tqdm=True, **kwargs):
    _cache_aev(output, dataset_path, batchsize, device, constfile,
               subtract_sae, sae_file, enable_tqdm, AEVCacheLoader.encode_aev,
               **kwargs)


def cache_sparse_aev(output, dataset_path, batchsize, device=default_device,
                     constfile=ani1x.const_file, subtract_sae=False,
                     sae_file=ani1x.sae_file, enable_tqdm=True, **kwargs):
    _cache_aev(output, dataset_path, batchsize, device, constfile,
               subtract_sae, sae_file, enable_tqdm,
               SparseAEVCacheLoader.encode_aev, **kwargs)


__all__ = ['load_ani_dataset', 'BatchedANIDataset', 'AEVCacheLoader', 'SparseAEVCacheLoader', 'cache_aev', 'cache_sparse_aev']
