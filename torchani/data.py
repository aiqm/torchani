from .pyanitools import anidataloader
from os import listdir
from os.path import join, isfile, isdir
from torch import tensor, full_like, long
from torch.utils.data import TensorDataset, ConcatDataset
from torch.utils.data.dataloader import default_collate
from math import ceil
from . import default_dtype

def load_dataset(path, dtype=default_dtype):
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
    return species, ConcatDataset(datasets)

class BatchSampler(object):

    def __init__(self, concat_source, chunk_size, batch_chunks):
        self.concat_source = concat_source
        self.chunk_size = chunk_size
        self.batch_chunks = batch_chunks

    def _concated_index(self, molecule, conformation):
        """
        Get the index in the  dataset of the specified conformation
        of the specified molecule.
        """
        src = self.concat_source
        cumulative_sizes = [0] + src.cumulative_sizes
        return cumulative_sizes[molecule] + conformation

    def __iter__(self):
        src = self.concat_source
        molecules = len(src.datasets)
        sizes = [len(x) for x in src.datasets]
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
                batch += [self._concated_index(molecule, x) for x in range(progress, end)]
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
        sizes = [len(x) for x in self.concat_source.datasets]
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

def random_split(dataset, lengths, chunk_size):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths
    ds

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (iterable): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")