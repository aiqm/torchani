from torch.utils.data import Dataset, DataLoader
from os.path import join, isfile, isdir
from os import listdir
from .pyanitools import anidataloader
import torch


class ANIDataset(Dataset):

    def __init__(self, path, chunk_size, shuffle=True, properties=['energies']):
        super(ANIDataset, self).__init__()

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

        # generate chunks
        chunks = []
        for f in files:
            for m in anidataloader(f):
                full = {
                    'coordinates': torch.from_numpy(m['coordinates'])
                }
                conformations = full['coordinates'].shape[0]
                for i in properties:
                    full[i] = torch.from_numpy(m[i])
                species = m['species']
                if shuffle:
                    indices = torch.randperm(conformations)
                else:
                    indices = torch.arange(conformations, dtype=torch.int64)
                num_chunks = (conformations + chunk_size - 1) // chunk_size
                for i in range(num_chunks):
                    chunk_start = i * chunk_size
                    chunk_end = min(chunk_start + chunk_size, conformations)
                    chunk_indices = indices[chunk_start:chunk_end]
                    chunk = {}
                    for j in full:
                        chunk[j] = full[j].index_select(0, chunk_indices)
                    chunk['species'] = species
                    chunks.append(chunk)
        self.chunks = chunks

    def __getitem__(self, idx):
        return self.chunks[idx]

    def __len__(self):
        return len(self.chunks)


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


def dataloader(dataset, batch_chunks, **kwargs):
    return DataLoader(dataset, batch_chunks, dataset.shuffle,
                      collate_fn=_collate, **kwargs)