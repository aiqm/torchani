from torch.utils.data import Dataset
from os.path import join, isfile, isdir
import os
from .pyanitools import anidataloader
import torch
import torch.utils.data as data
import pickle
from .. import padding


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
        properties = {self.properties[i]: [] for i in self.properties}
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
        batches = []
        num_batches = (conformations + batch_size - 1) / batch_size
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, conformations)
            species_batch = species[start:end, ...]
            coordinates_batch = coordinates[start:end, ...]
            properties_batch = {
                k: properties[k][start:end, ...] for k in properties
            }
            batches.append((species_batch, coordinates_batch),
                           properties_batch)
        self.batches = batches

    def __getitem__(self, idx):
        return self.batches[idx]

    def __len__(self):
        return len(self.batches)


def load_or_create(checkpoint, dataset_path, batch_size, *args, **kwargs):
    """Generate a 80-10-10 split of the dataset, and checkpoint
    the resulting dataset"""
    if not os.path.isfile(checkpoint):
        full_dataset = BatchedANIDataset(dataset_path, batch_size,
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
