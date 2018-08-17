from torch.utils.data import Dataset
from os.path import join, isfile, isdir
import os
from .pyanitools import anidataloader
import torch
import torch.utils.data as data
import pickle
from .. import padding
import itertools
import math
import copy
import random
import tqdm


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


split_min_cost = 40000
def split_cost(counts, split):
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
        if len(counts) == 0:
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
        s, c = padding.strip_redundant_padding(s, c)
        species_coordinates.append((s, c))
        start = end
    return species_coordinates


class BatchedANIDataset(Dataset):

    def __init__(self, path, species, batch_size, shuffle=True,
                 properties=['energies'], transform=(),
                 dtype=torch.get_default_dtype(), device=torch.device('cpu')):
        super(BatchedANIDataset, self).__init__()
        self.path = path
        self.species = species
        self.species_indices = {
            self.species[i]: i for i in range(len(self.species))}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.properties = properties
        self.dtype = dtype
        self.device = device
        device = torch.device('cpu')

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
                species = m['species']
                indices = [self.species_indices[i] for i in species]
                species = torch.tensor(indices, dtype=torch.long,
                                       device=device)
                coordinates = torch.from_numpy(m['coordinates']) \
                                   .type(dtype).to(device)
                species_coordinates.append((species, coordinates))
                for i in properties:
                    properties[i].append(torch.from_numpy(m[i])
                                              .type(dtype).to(device))
        species, coordinates = padding.pad_and_batch(species_coordinates)
        for i in properties:
            properties[i] = torch.cat(properties[i])

        # shuffle if required
        conformations = coordinates.shape[0]
        if shuffle:
            indices = torch.randperm(conformations, device=device)
            species = species.index_select(0, indices)
            coordinates = coordinates.index_select(0, indices)
            for i in properties:
                properties[i] = properties[i].index_select(0, indices)

        # do transformations on data
        for t in transform:
            species, coordinates, properties = t(species, coordinates,
                                                 properties)

        # split into minibatches, and strip reduncant padding
        natoms = (species >= 0).to(torch.long).sum(1)
        batches = []
        num_batches = (conformations + batch_size - 1) // batch_size
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, conformations)
            natoms_batch = natoms[start:end]
            natoms_batch, indices = natoms_batch.sort()
            species_batch = species[start:end, ...].index_select(0, indices)
            coordinates_batch = coordinates[start:end, ...] \
                .index_select(0, indices)
            properties_batch = {
                k: properties[k][start:end, ...].index_select(0, indices)
                for k in properties
            }
            # further split batch into chunks
            species_coordinates = split_batch(natoms_batch, species_batch, coordinates_batch)
            batch = species_coordinates, properties_batch
            batches.append(batch)
        self.batches = batches

    def __getitem__(self, idx):
        species_coordinates, properties = self.batches[idx]
        species_coordinates = [(s.to(self.device), c.to(self.device)) for s,c in species_coordinates]
        properties = {
            k: properties[k].to(self.device) for k in properties
        }
        return species_coordinates, properties

    def __len__(self):
        return len(self.batches)


def load_or_create(checkpoint, batch_size, species, dataset_path,
                   *args, **kwargs):
    """Generate a 80-10-10 split of the dataset, and checkpoint
    the resulting dataset"""
    if not os.path.isfile(checkpoint):
        full_dataset = BatchedANIDataset(dataset_path, species, batch_size,
                                         *args, **kwargs)
        training_size = int(len(full_dataset) * 0.8)
        validation_size = int(len(full_dataset) * 0.1)
        testing_size = len(full_dataset) - training_size - validation_size
        lengths = [training_size, validation_size, testing_size]
        subsets = data.random_split(full_dataset, lengths)
        with open(checkpoint, 'wb') as f:
            pickle.dump(subsets, f)

    # load dataset from checkpoint file
    with open(checkpoint, 'rb') as f:
        training, validation, testing = pickle.load(f)
    return training, validation, testing
