from torch.utils.data import Dataset, DataLoader
from os.path import join, isfile, isdir
import os
from .pyanitools import anidataloader
from .env import default_dtype, default_device
import torch
import torch.utils.data as data
import pickle


class ANIDataset(Dataset):

    def __init__(self, path, chunk_size, shuffle=True, properties=['energies'],
                 transform=(), dtype=default_dtype, device=default_device):
        super(ANIDataset, self).__init__()
        self.path = path
        self.chunks_size = chunk_size
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

        # generate chunks
        chunks = []
        for f in files:
            for m in anidataloader(f):
                full = {
                    'coordinates': torch.from_numpy(m['coordinates'])
                                        .type(dtype).to(device)
                }
                conformations = full['coordinates'].shape[0]
                for i in properties:
                    full[i] = torch.from_numpy(m[i]).type(dtype).to(device)
                species = m['species']
                if shuffle:
                    indices = torch.randperm(conformations, device=device)
                else:
                    indices = torch.arange(conformations, dtype=torch.int64,
                                           device=device)
                num_chunks = (conformations + chunk_size - 1) // chunk_size
                for i in range(num_chunks):
                    chunk_start = i * chunk_size
                    chunk_end = min(chunk_start + chunk_size, conformations)
                    chunk_indices = indices[chunk_start:chunk_end]
                    chunk = {}
                    for j in full:
                        chunk[j] = full[j].index_select(0, chunk_indices)
                    chunk['species'] = species
                    for t in transform:
                        chunk = t(chunk)
                    chunks.append(chunk)
        self.chunks = chunks

    def __getitem__(self, idx):
        return self.chunks[idx]

    def __len__(self):
        return len(self.chunks)


def load_or_create(checkpoint, dataset_path, chunk_size, *args, **kwargs):
    """Generate a 80-10-10 split of the dataset, and checkpoint
    the resulting dataset"""
    if not os.path.isfile(checkpoint):
        full_dataset = ANIDataset(dataset_path, chunk_size, *args, **kwargs)
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


def _collate(batch):
    input_keys = ['coordinates', 'species']
    inputs = [{k: i[k] for k in input_keys} for i in batch]
    outputs = {}
    for i in batch:
        for j in i:
            if j in input_keys:
                continue
            if j not in outputs:
                outputs[j] = []
            outputs[j].append(i[j])
    for i in outputs:
        outputs[i] = torch.cat(outputs[i])
    return inputs, outputs


def dataloader(dataset, batch_chunks, shuffle=True, **kwargs):
    return DataLoader(dataset, batch_chunks, shuffle,
                      collate_fn=_collate, **kwargs)
