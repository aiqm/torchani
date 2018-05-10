import h5py
import os
from .pyanitools import anidataloader
import random
import json
import copy


class Dataset:
    """Class for reading ANI datasets.

    This class is not a subclass of `torch.utils.data.Dataset`, and has
    different API.
    """

    def __init__(self, filename=None):
        """Initialize Dataset.

        Parameters
        ----------
        filename : string
            Path to hdf5 files. If a single file is specified, then all the data
            will come from this file. If a directory is specified, then all  files
            with extension `.h5` or `.hdf5` in that directory will be loaded. If
            `None` is given, then the dataset will not be initialized. The user
            then have to manually load from a json file dump.
        """
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

        Only conformations belong to the same molecule will be yielded. If the remaining
        conformations is less than the specified batch size, a smaller batch will be yielded.

        Parameters
        ----------
        batch_size : int
            Number of conformations to yield each time.
        subset : string
            The name of subset to iterate. If `None` or 'all' is given, then iterate on the
            whole dataset.
        """
        subset = self._keys if (
            subset is None or subset == 'all') else self._subsets[subset]
        for filename, path in subset:
            loader = self._loaders[filename]
            data = loader.store[path]
            species = [i.decode('ascii') for i in data['species']]
            coordinates = data['coordinates'][()]
            conformations = coordinates.shape[0]
            energies = data['energies'][()]
            start_index = 0
            end_index = batch_size
            while start_index < conformations:
                if end_index > conformations:
                    yield coordinates[start_index:], energies[start_index:], species
                else:
                    yield coordinates[start_index:end_index], energies[start_index:end_index], species
                start_index = end_index
                end_index += batch_size
