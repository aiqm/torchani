import h5py
import os
from .pyanitools import anidataloader
import random
import json
import copy
import torch
from . import default_dtype, default_device


class Dataset:
    """Class for reading ANI datasets.

    This class is not a subclass of `torch.utils.data.Dataset`, and has
    different API.
    """

    def __init__(self, filename=None, dtype=default_dtype, device=default_device):
        """Initialize Dataset.

        Parameters
        ----------
        filename : string
            Path to hdf5 files. If a single file is specified, then all the data
            will come from this file. If a directory is specified, then all  files
            with extension `.h5` or `.hdf5` in that directory will be loaded. If
            `None` is given, then the dataset will not be initialized. The user
            then have to manually load from a json file dump.
        device : torch.Device
            The device where the tensor should be put.
        """
        self.device = device
        self.dtype = dtype

        if filename is None:
            return
        if os.path.isdir(filename):
            complete_filenames = [os.path.join(
                filename, f) for f in os.listdir(filename)]
            filenames = [f for f in complete_filenames if os.path.isfile(
                f) and (f.endswith('.h5') or f.endswith('.hdf5'))]
        elif os.path.isfile(filename):
            filenames = [filename]
        else:
            raise ValueError('Bad file name')
        self._loaders = {}
        for f in filenames:
            self._loaders[f] = anidataloader(f)
        self._keys = [(f, i['path'])
                      for f in filenames for i in self._loaders[f]]
        self._subsets = {}

    def shuffle(self):
        """Shuffle the whole dataset.

        Note that this methond have no effect on already splitted subsets.
        If you want to shuffle everything, then you should shuffle the whole
        dataset and then resplit it into subsets.
        """
        random.shuffle(self._keys)

    def split(self, *subsets):
        """Split the whole dataset into subsets using the given ratio.

        If the dataset is already splited, then the old split will be overwritten.

        Parameters
        ----------
        *subsets : list of (string, float)
            List of names and ratios of subsets. Names must be different. All ratios
            must sum to 1. Name of sublist can not be 'all', which is reserved for the
            whole dataset.
        """
        self._subsets = {}
        total_ratio = 0
        start_index = 0
        for name, ratio in subsets:
            if name in self._subsets or name == 'all':
                raise ValueError('duplicate names')
            total_ratio += ratio
            end_index = int(total_ratio * len(self._keys))
            self._subsets[name] = self._keys[start_index:end_index]
            start_index = end_index
        if total_ratio != 1:
            raise ValueError('ratio must sum to 1')

    def save(self, filename):
        """Save the current shuffle and splits into file"""
        d = copy.copy(self._subsets)
        d['all'] = self._keys
        d = json.dumps(d)
        with open(filename, 'w') as f:
            f.write(d)

    def load(self, filename):
        """Load shuffle and splits from file"""
        # read keys and subsets
        with open(filename, 'r') as f:
            d = json.loads(f.read())
            self._keys = [(x[0], x[1]) for x in d['all']]
            del d['all']
            self._subsets = {}
            for k in d:
                self._subsets[k] = [(x[0], x[1]) for x in d[k]]
            self._loaders = {}

        # initialize loaders
        for f, _ in self._keys:
            if f not in self._loaders:
                self._loaders[f] = anidataloader(f)

    def iter(self, batch_size, subset=None):
        """Iterate through the selected subset. Each yield get batch size number of conformations

        Only conformations belong to the same molecule will be yielded for each batch. If the
        remaining conformations is less than the specified batch size, a smaller batch will be
        yielded. The iterator will visit the first batch of the first molecule, then the first
        batch of the second molecule, ..., after the first batch of all the molecules are visited,
        the second batch of the first molecule will be visited, then the second batch of the second
        molecule, ..., and so on.

        Parameters
        ----------
        batch_size : int
            Number of conformations to yield each time.
        subset : string
            The name of subset to iterate. If `None` or 'all' is given, then iterate on the
            whole dataset.

        Yields
        ------
        (torch.Tensor, torch.Tensor, list)
            Tuple of coordinates, energies, species
        """
        subset = self._keys if (
            subset is None or subset == 'all') else self._subsets[subset]
        progress = {}
        done = False
        while not done:
            done = True
            for key in subset:
                filename, path = key
                loader = self._loaders[filename]
                data = loader.store[path]
                species = [i.decode('ascii') for i in data['species']]
                coordinates = data['coordinates'][()]
                conformations = coordinates.shape[0]
                energies = data['energies'][()]
                if key not in progress:
                    progress[key] = 0
                start_index = progress[key]
                end_index = start_index + batch_size
                if start_index < conformations:
                    done = False
                    progress[key] = end_index
                    if end_index > conformations:
                        c = torch.from_numpy(coordinates[start_index:]).type(
                            self.dtype).to(self.device)
                        e = torch.from_numpy(energies[start_index:]).type(
                            self.dtype).to(self.device)
                        yield c, e, species
                    else:
                        c = torch.from_numpy(coordinates[start_index:end_index]).type(
                            self.dtype).to(self.device)
                        e = torch.from_numpy(energies[start_index:end_index]).type(
                            self.dtype).to(self.device)
                        yield c, e, species
