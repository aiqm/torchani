from torch.utils.data import Dataset
from os.path import join, isfile, isdir
from os import listdir
from .pyanitools import anidataloader
import torch


class ANIDataset(Dataset):

    def __init__(self, path, chunk_size, randomize_chunk=True):
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
                xyz = torch.from_numpy(m['coordinates'])
                conformations = xyz.shape[0]
                energies = torch.from_numpy(m['energies'])
                species = m['species']
                if randomize_chunk:
                    indices = torch.randperm(conformations)
                else:
                    indices = torch.arange(conformations, dtype=torch.int64)
                num_chunks = (conformations + chunk_size - 1) // chunk_size
                for i in range(num_chunks):
                    chunk_start = i * chunk_size
                    chunk_end = min(chunk_start + chunk_size, conformations)
                    chunk_indices = indices[chunk_start:chunk_end]
                    chunk_xyz = xyz.index_select(0, chunk_indices)
                    chunk_energies = energies.index_select(0, chunk_indices)
                    chunks.append((chunk_xyz, chunk_energies, species))
        self.chunks = chunks

    def __getitem__(self, idx):
        return self.chunks[idx]

    def __len__(self):
        return len(self.chunks)
