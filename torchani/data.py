from .pyanitools import anidataloader
from os import listdir
from os.path import join, isfile, isdir
from torch import tensor, full_like, long
from torch.utils.data import Dataset, Subset, TensorDataset, ConcatDataset
from torch.utils.data.dataloader import default_collate
from math import ceil
from . import default_dtype
from random import shuffle
from itertools import chain, accumulate


class ANIDataset(Dataset):
    """Dataset with extra information for ANI applications

    Attributes
    ----------
    dataset : Dataset
        The dataset
    sizes : sequence
        Number of conformations for each molecule
    cumulative_sizes : sequence
        Cumulative sizes
    """

    def __init__(self, dataset, sizes, species):
        super(ANIDataset, self).__init__()
        self.dataset = dataset
        self.sizes = sizes
        self.cumulative_sizes = list(accumulate(sizes))
        self.species = species

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


def load_dataset(path, dtype=default_dtype):
    """The returned dataset has cumulative_sizes and molecule_sizes"""
    # get name of files storing data
    files = []
    if isdir(path):
        for f in listdir(path):
            f = join(path, f)
            if isfile(f) and (f.endswith('.h5') or f.endswith('.hdf5')):
                files.append(f)
    elif isfile(path):
        files = [path]
    else:
        raise ValueError('Bad path')

    # read tensors from file and build a dataset
    species = []
    molecule_id = 0
    datasets = []
    for f in files:
        for m in anidataloader(f):
            coordinates = tensor(m['coordinates'], dtype=dtype)
            energies = tensor(m['energies'], dtype=dtype)
            _molecule_id = full_like(energies, molecule_id).type(long)
            datasets.append(TensorDataset(_molecule_id, coordinates, energies))
            species.append(m['species'])
            molecule_id += 1
    dataset = ConcatDataset(datasets)
    sizes = [len(x) for x in dataset.datasets]
    return ANIDataset(dataset, sizes, species)


class BatchSampler(object):

    def __init__(self, source, chunk_size, batch_chunks):
        if not isinstance(source, ANIDataset):
            raise ValueError("BatchSampler must take ANIDataset as input")
        self.source = source
        self.chunk_size = chunk_size
        self.batch_chunks = batch_chunks

    def _concated_index(self, molecule, conformation):
        """
        Get the index in the  dataset of the specified conformation
        of the specified molecule.
        """
        src = self.source
        cumulative_sizes = [0] + src.cumulative_sizes
        return cumulative_sizes[molecule] + conformation

    def __iter__(self):
        molecules = len(self.source.sizes)
        sizes = self.source.sizes
        """Number of conformations of each molecule"""
        unfinished = list(zip(range(molecules), [0] * molecules))
        """List of pairs (molecule, progress) storing the current progress
        of iterating each molecules."""

        batch = []
        batch_molecules = 0
        """The number of molecules already in batch"""
        while len(unfinished) > 0:
            new_unfinished = []
            for molecule, progress in unfinished:
                size = sizes[molecule]
                # the last incomplete chunk is not dropped
                end = min(progress + self.chunk_size, size)
                if end < size:
                    new_unfinished.append((molecule, end))
                batch += [self._concated_index(molecule, x)
                          for x in range(progress, end)]
                batch_molecules += 1
                if batch_molecules >= self.batch_chunks:
                    yield batch
                    batch = []
                    batch_molecules = 0
            unfinished = new_unfinished

        # the last incomplete batch is not dropped
        if len(batch) > 0:
            yield batch

    def __len__(self):
        sizes = self.source.sizes
        chunks = [ceil(x/self.chunk_size) for x in sizes]
        chunks = sum(chunks)
        return ceil(chunks / self.batch_chunks)


def collate(batch):
    by_molecules = {}
    for molecule_id, xyz, energy in batch:
        molecule_id = molecule_id.item()
        if molecule_id not in by_molecules:
            by_molecules[molecule_id] = []
        by_molecules[molecule_id].append((xyz, energy))
    for i in by_molecules:
        by_molecules[i] = default_collate(by_molecules[i])
    return by_molecules


def random_split(dataset, num_chunks, chunk_size):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths

    The splitting is by chunk, which makes it possible for batching: The whole
    dataset is first splitted into chunks of specified size, each chunk are
    different conformation of the same isomer/molecule, then these chunks are
    randomly shuffled and splitted accorting to the given `num_chunks`. After
    splitted, chunks belong to the same molecule/isomer of the same subset will
    be merged to allow larger batch.

    Parameters
    ----------
    dataset : Dataset:
        Dataset to be split
    num_chunks : sequence
        Number of chuncks of splits to be produced
    chunk_size : integer
        Size of each chunk
    """
    chunks = list(BatchSampler(dataset, chunk_size, 1))
    shuffle(chunks)
    if sum(num_chunks) != len(chunks):
        raise ValueError(
            """Sum of input number of chunks does not equal the length of the
            total dataset!""")
    offset = 0
    subsets = []
    for i in num_chunks:
        _chunks = chunks[offset:offset+i]
        offset += i
        # merge chunks by molecule
        by_molecules = {}
        for chunk in _chunks:
            molecule_id = dataset[chunk[0]][0].item()
            if molecule_id not in by_molecules:
                by_molecules[molecule_id] = []
            by_molecules[molecule_id] += chunk
        _chunks = list(by_molecules.values())
        shuffle(_chunks)
        # construct subset
        sizes = [len(j) for j in _chunks]
        indices = list(chain.from_iterable(_chunks))
        _dataset = Subset(dataset, indices)
        _dataset = ANIDataset(_dataset, sizes, dataset.species)
        subsets.append(_dataset)
    return subsets
