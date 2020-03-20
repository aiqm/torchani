# -*- coding: utf-8 -*-
"""Tools for loading, shuffling, and batching ANI datasets"""

from torch.utils.data import Dataset
from os.path import join, isfile, isdir
import os
from ._pyanitools import anidataloader
import torch
from .. import utils
from .new import CachedDataset, ShuffledDataset, find_threshold

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def split_whole_into_batches(atomic_properties, properties, batch_size):
    molecules = atomic_properties['species'].shape[0]
    # split into minibatches
    for k in properties:
        properties[k] = properties[k].split(batch_size)
    for k in atomic_properties:
        atomic_properties[k] = atomic_properties[k].split(batch_size)

    # strip redundant padding
    batches = []
    num_batches = (molecules + batch_size - 1) // batch_size
    for i in range(num_batches):
        batch_properties = {k: v[i] for k, v in properties.items()}
        batch_atomic_properties = {k: v[i] for k, v in atomic_properties.items()}
        batch_atomic_properties = utils.strip_redundant_padding(batch_atomic_properties)
        batches.append((batch_atomic_properties, batch_properties))

    return batches


class PaddedDataset(Dataset):
    r""" Dataset that contains batches with padded structures"""

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

        self.batches = split_whole_into_batches(atomic_properties, properties, batch_size)

    def __getitem__(self, idx):
        atomic_properties, properties = self.batches[idx]

        def convert(v):
            v = v.to(self.device)
            if torch.is_floating_point(v):
                return v.to(self.dtype)
            return v
        atomic_properties_ = {k: convert(v) for k, v in atomic_properties.items()}
        properties_ = {k: convert(v) for k, v in properties.items()}
        return atomic_properties_, properties_

    def __len__(self):
        return len(self.batches)


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

        mean = scaled_diff[torch.abs(scaled_diff) < 15.0].mean()
        std = scaled_diff[torch.abs(scaled_diff) < 15.0].std()

        # -8 * std + mean < scaled_diff < +8 * std + mean
        tol = 8.0 * std + mean
        low_idx = (torch.abs(scaled_diff) < tol).nonzero().squeeze()
        outlier_count = molecules - low_idx.numel()

        # discard outlier energy conformers if exist
        if outlier_count > 0:
            print("Note: {} outlier energy conformers have been discarded from dataset".format(outlier_count))
            for key, val in atomic_properties_.items():
                atomic_properties_[key] = val[low_idx]
            for key, val in properties_.items():
                properties_[key] = val[low_idx]
            molecules = low_idx.numel()

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
        ds = PaddedDataset(ap, p, batch_size, dtype, device)
        ds.properties = properties
        ds.atomic_properties = atomic_properties
        ret.append(ds)
    if len(ret) == 1:
        return ret[0]
    return tuple(ret)


import numpy as np
import torch
import functools
from ._pyanitools import anidataloader
from importlib import util as u
import gc

PKBAR_INSTALLED = u.find_spec('pkbar') is not None
if PKBAR_INSTALLED:
    import pkbar


class CachedDataset(torch.utils.data.Dataset):
    """ Cached Dataset which is shuffled once, but the dataset keeps the same at every epoch.

    Arguments:
        file_path (str): Path to one hdf5 file.
        batch_size (int): batch size.
        device (str): ``'cuda'`` or ``'cpu'``, cache to CPU or GPU. Commonly, 'cpu' is already fast enough.
            Default is ``'cpu'``.
        chunk_threshold (int): threshould to split batch into chunks. Set to ``None`` will not split chunks.
            Use ``torchani.data.find_threshold`` to find resonable ``chunk_threshold``.
        other_properties (dict): A dict which is used to extract properties other than
            ``energies`` from dataset with correct padding, shape and dtype.\n
            The example below will extract ``dipoles`` and ``forces``.\n
            ``padding_values``: set to ``None`` means there is no need to pad for this property.

            .. code-block:: python

                other_properties = {'properties': ['dipoles', 'forces'],
                                    'padding_values': [None, 0],
                                    'padded_shapes': [(batch_size, 3), (batch_size, -1, 3)],
                                    'dtypes': [torch.float32, torch.float32]
                                    }

        include_energies (bool): Whether include energies into properties. Default is ``True``.
        species_order (list): a list which specify how species are transfomed to int.
            for example: ``['H', 'C', 'N', 'O']`` means ``{'H': 0, 'C': 1, 'N': 2, 'O': 3}``.
        subtract_self_energies (bool): whether subtract self energies from ``energies``.
        self_energies (list): if `subtract_self_energies` is True, the order should keep
            the same as ``species_order``.
            for example :``[-0.600953, -38.08316, -54.707756, -75.194466]`` will be converted
            to ``{'H': -0.600953, 'C': -38.08316, 'N': -54.707756, 'O': -75.194466}``.

    .. note::
        The resulting dataset will be:
        ``([chunk1, chunk2, ...], {'energies', 'force', ...})`` in which chunk1 is a
        tuple of ``(species, coordinates)``.

        e.g. the shape of\n
        chunk1: ``[[1807, 21], [1807, 21, 3]]``\n
        chunk2: ``[[193, 50], [193, 50, 3]]``\n
        'energies': ``[2000, 1]``
    """
    def __init__(self, file_path,
                 batch_size=1000,
                 device='cpu',
                 chunk_threshold=20,
                 other_properties={},
                 include_energies=True,
                 species_order=['H', 'C', 'N', 'O'],
                 subtract_self_energies=False,
                 self_energies=[-0.600953, -38.08316, -54.707756, -75.194466]):

        super(CachedDataset, self).__init__()

        # example of species_dict will looks like
        # species_dict: {'H': 0, 'C': 1, 'N': 2, 'O': 3}
        # self_energies_dict: {'H': -0.600953, 'C': -38.08316, 'N': -54.707756, 'O': -75.194466}
        species_dict = {}
        self_energies_dict = {}
        for i, s in enumerate(species_order):
            species_dict[s] = i
            self_energies_dict[s] = self_energies[i]

        self.batch_size = batch_size
        self.data_species = []
        self.data_coordinates = []
        data_self_energies = []
        self.data_properties = {}
        self.properties_info = other_properties

        # whether include energies to properties
        if include_energies:
            self.add_energies_to_properties()
        # let user check the properties will be loaded
        self.check_properties()

        # anidataloader
        anidata = anidataloader(file_path)
        anidata_size = anidata.group_size()
        self.enable_pkbar = anidata_size > 5 and PKBAR_INSTALLED
        if self.enable_pkbar:
            pbar = pkbar.Pbar('=> loading h5 dataset into cpu memory, total molecules: {}'.format(anidata_size), anidata_size)

        # load h5 data into cpu memory as lists
        for i, molecule in enumerate(anidata):
            # conformations
            num_conformations = len(molecule['coordinates'])
            # species and coordinates
            self.data_coordinates += list(molecule['coordinates'].reshape(num_conformations, -1).astype(np.float32))
            species = np.array([species_dict[x] for x in molecule['species']])
            self.data_species += list(np.tile(species, (num_conformations, 1)))
            # if subtract_self_energies
            if subtract_self_energies:
                self_energies = np.array(sum([self_energies_dict[x] for x in molecule['species']]))
                data_self_energies += list(np.tile(self_energies, (num_conformations, 1)))
            # properties
            for key in self.data_properties:
                self.data_properties[key] += list(molecule[key].reshape(num_conformations, -1))
            # pkbar update
            if self.enable_pkbar:
                pbar.update(i)

        # if subtract self energies
        if subtract_self_energies and 'energies' in self.properties_info['properties']:
            self.data_properties['energies'] = np.array(self.data_properties['energies']) - np.array(data_self_energies)
            del data_self_energies
            gc.collect()

        self.length = (len(self.data_species) + self.batch_size - 1) // self.batch_size
        self.device = device
        self.shuffled_index = np.arange(len(self.data_species))
        np.random.shuffle(self.shuffled_index)

        # clean trash
        anidata.cleanup()
        del num_conformations
        del species
        del anidata
        gc.collect()

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):

        if index >= self.length:
            raise IndexError()

        batch_indices = slice(index * self.batch_size, (index + 1) * self.batch_size)
        batch_indices_shuffled = self.shuffled_index[batch_indices]

        batch_species = [self.data_species[i] for i in batch_indices_shuffled]
        batch_coordinates = [self.data_coordinates[i] for i in batch_indices_shuffled]

        # properties
        properties = {}
        for i, key in enumerate(self.properties_info['properties']):
            # get a batch of property
            prop = [self.data_properties[key][i] for i in batch_indices_shuffled]
            # sort with number of atoms
            prop = self.sort_list_with_index(prop, sorted_atoms_idx.numpy())
            # padding and convert to tensor
            if self.properties_info['padding_values'][i] is None:
                prop = self.pad_and_convert_to_tensor([prop], no_padding=True)[0]
            else:
                prop = self.pad_and_convert_to_tensor([prop], padding_value=self.properties_info['padding_values'][i])[0]
            # set property shape and dtype
            padded_shape = list(self.properties_info['padded_shapes'][i])
            padded_shape[0] = prop.shape[0]  # the last batch may does not have one batch data
            properties[key] = prop.reshape(padded_shape).to(self.properties_info['dtypes'][i])

        return chunks, properties

    def __len__(self):
        return self.length

    def split(self, validation_split):
        """Split dataset into traning and validaiton.

        Arguments:
            validation_split (float): Float between 0 and 1. Fraction of the dataset to be used
                as validation data.
        """
        val_size = int(validation_split * len(self))
        train_size = len(self) - val_size

        ds = []
        if self.enable_pkbar:
            message = ('=> processing, splitting and caching dataset into cpu memory: \n'
                       + 'total batches: {}, train batches: {}, val batches: {}, batch_size: {}')
            pbar = pkbar.Pbar(message.format(len(self), train_size, val_size, self.batch_size),
                              len(self))
        for i, _ in enumerate(self):
            ds.append(self[i])
            if self.enable_pkbar:
                pbar.update(i)

        train_dataset = ds[:train_size]
        val_dataset = ds[train_size:]

        return train_dataset, val_dataset

    def load(self):
        """Cache dataset into CPU memory. If not called, dataset will be cached during the first epoch.
        """
        if self.enable_pkbar:
            pbar = pkbar.Pbar('=> processing and caching dataset into cpu memory: \ntotal '
                              + 'batches: {}, batch_size: {}'.format(len(self), self.batch_size),
                              len(self))
        for i, _ in enumerate(self):
            if self.enable_pkbar:
                pbar.update(i)

    def add_energies_to_properties(self):
        # if user does not provide energies info
        if 'properties' in self.properties_info and 'energies' not in self.properties_info['properties']:
            # setup energies info, so the user does not need to input energies
            self.properties_info['properties'].append('energies')
            self.properties_info['padding_values'].append(None)
            self.properties_info['padded_shapes'].append((self.batch_size, ))
            self.properties_info['dtypes'].append(torch.float64)
        # if no properties provided
        if 'properties' not in self.properties_info:
            self.properties_info = {'properties': ['energies'],
                                    'padding_values': [None],
                                    'padded_shapes': [(self.batch_size, )],
                                    'dtypes': [torch.float64],
                                    }

    def check_properties(self):
        # print properties information
        print('... The following properties will be loaded:')
        for i, prop in enumerate(self.properties_info['properties']):
            self.data_properties[prop] = []
            message = '{}: (dtype: {}, padding_value: {}, padded_shape: {})'
            print(message.format(prop, self.properties_info['dtypes'][i],
                                 self.properties_info['padding_values'][i],
                                 self.properties_info['padded_shapes'][i]))

    def pad_and_convert_to_tensor(self, inputs, padding_value=0, no_padding=False):
        if no_padding:
            for i, input_tmp in enumerate(inputs):
                inputs[i] = torch.from_numpy(np.stack(input_tmp)).to(self.device)
        else:
            for i, input_tmp in enumerate(inputs):
                inputs[i] = torch.nn.utils.rnn.pad_sequence(
                    [torch.from_numpy(b) for b in inputs[i]],
                    batch_first=True, padding_value=padding_value).to(self.device)
        return inputs

    def release_h5(self):
        del self.data_species
        del self.data_coordinates
        del self.data_energies
        gc.collect()


def ShuffledDataset(file_path,
                    batch_size=1000, num_workers=0, shuffle=True,
                    chunk_threshold=20,
                    other_properties={},
                    include_energies=True,
                    validation_split=0.0,
                    species_order=['H', 'C', 'N', 'O'],
                    subtract_self_energies=False,
                    self_energies=[-0.600953, -38.08316, -54.707756, -75.194466]):
    """ Shuffled Dataset which using `torch.utils.data.DataLoader`, it will shuffle at every epoch.

    Arguments:
        file_path (str): Path to one hdf5 file.
        batch_size (int): batch size.
        num_workers (int): multiple process to prepare dataset at background when
            training is going.
        shuffle (bool): whether to shuffle.
        chunk_threshold (int): threshould to split batch into chunks. Set to ``None`` will not split chunks.
            Use ``torchani.data.find_threshold`` to find resonable ``chunk_threshold``.
        other_properties (dict): A dict which is used to extract properties other than
            ``energies`` from dataset with correct padding, shape and dtype.\n
            The example below will extract ``dipoles`` and ``forces``.\n
            ``padding_values``: set to ``None`` means there is no need to pad for this property.

            .. code-block:: python

                other_properties = {'properties': ['dipoles', 'forces'],
                                    'padding_values': [None, 0],
                                    'padded_shapes': [(batch_size, 3), (batch_size, -1, 3)],
                                    'dtypes': [torch.float32, torch.float32]
                                    }

        include_energies (bool): Whether include energies into properties. Default is ``True``.
        validation_split (float): Float between 0 and 1. Fraction of the dataset to be used
            as validation data.
        species_order (list): a list which specify how species are transfomed to int.
            for example: ``['H', 'C', 'N', 'O']`` means ``{'H': 0, 'C': 1, 'N': 2, 'O': 3}``.
        subtract_self_energies (bool): whether subtract self energies from ``energies``.
        self_energies (list): if `subtract_self_energies` is True, the order should keep
            the same as ``species_order``.
            for example :``[-0.600953, -38.08316, -54.707756, -75.194466]`` will be
            converted to ``{'H': -0.600953, 'C': -38.08316, 'N': -54.707756, 'O': -75.194466}``.

    .. note::
        Return a dataloader that, when iterating, you will get

        ``([chunk1, chunk2, ...], {'energies', 'force', ...})`` in which chunk1 is a
        tuple of ``(species, coordinates)``.\n
        e.g. the shape of\n
        chunk1: ``[[1807, 21], [1807, 21, 3]]``\n
        chunk2: ``[[193, 50], [193, 50, 3]]``\n
        'energies': ``[2000, 1]``
    """

    dataset = TorchData(file_path,
                        batch_size,
                        other_properties,
                        include_energies,
                        species_order,
                        subtract_self_energies,
                        self_energies)
    properties_info = dataset.get_properties_info()

    if not chunk_threshold:
        chunk_threshold = np.inf

    def my_collate_fn(data, chunk_threshold=chunk_threshold, properties_info=properties_info):
        return collate_fn(data, chunk_threshold, properties_info)

    val_size = int(validation_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=shuffle,
                                                    num_workers=num_workers,
                                                    pin_memory=False,
                                                    collate_fn=my_collate_fn)
    if val_size == 0:
        return train_data_loader

    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers,
                                                  pin_memory=False,
                                                  collate_fn=my_collate_fn)

    return train_data_loader, val_data_loader


class TorchData(torch.utils.data.Dataset):

    def __init__(self, file_path,
                 batch_size,
                 other_properties,
                 include_energies,
                 species_order,
                 subtract_self_energies,
                 self_energies):

        super(TorchData, self).__init__()

        species_dict = {}
        self_energies_dict = {}
        for i, s in enumerate(species_order):
            species_dict[s] = i
            self_energies_dict[s] = self_energies[i]

        self.batch_size = batch_size
        self.data_species = []
        self.data_coordinates = []
        data_self_energies = []
        self.data_properties = {}
        self.properties_info = other_properties

        # whether include energies to properties
        if include_energies:
            self.add_energies_to_properties()
        # let user check the properties will be loaded
        self.check_properties()

        # anidataloader
        anidata = anidataloader(file_path)
        anidata_size = anidata.group_size()
        self.enable_pkbar = anidata_size > 5 and PKBAR_INSTALLED
        if self.enable_pkbar:
            pbar = pkbar.Pbar('=> loading h5 dataset into cpu memory, total molecules: {}'.format(anidata_size), anidata_size)

        # load h5 data into cpu memory as lists
        for i, molecule in enumerate(anidata):
            # conformations
            num_conformations = len(molecule['coordinates'])
            # species and coordinates
            self.data_coordinates += list(molecule['coordinates'].reshape(num_conformations, -1).astype(np.float32))
            species = np.array([species_dict[x] for x in molecule['species']])
            self.data_species += list(np.tile(species, (num_conformations, 1)))
            # if subtract_self_energies
            if subtract_self_energies:
                self_energies = np.array(sum([self_energies_dict[x] for x in molecule['species']]))
                data_self_energies += list(np.tile(self_energies, (num_conformations, 1)))
            # properties
            for key in self.data_properties:
                self.data_properties[key] += list(molecule[key].reshape(num_conformations, -1))
            # pkbar update
            if self.enable_pkbar:
                pbar.update(i)

        # if subtract self energies
        if subtract_self_energies and 'energies' in self.properties_info['properties']:
            self.data_properties['energies'] = np.array(self.data_properties['energies']) - np.array(data_self_energies)
            del data_self_energies
            gc.collect()

        self.length = len(self.data_species)

        # clean trash
        anidata.cleanup()
        del num_conformations
        del species
        del anidata
        gc.collect()

    def __getitem__(self, index):

        if index >= self.length:
            raise IndexError()

        species = torch.from_numpy(self.data_species[index])
        coordinates = torch.from_numpy(self.data_coordinates[index]).float()
        properties = {}
        for key in self.data_properties:
            properties[key] = torch.from_numpy(self.data_properties[key][index])

        return [species, coordinates, properties]

    def __len__(self):
        return self.length

    def add_energies_to_properties(self):
        # if user does not provide energies info
        if 'properties' in self.properties_info and 'energies' not in self.properties_info['properties']:
            # setup energies info, so the user does not need to input energies
            self.properties_info['properties'].append('energies')
            self.properties_info['padding_values'].append(None)
            self.properties_info['padded_shapes'].append((self.batch_size, ))
            self.properties_info['dtypes'].append(torch.float64)
        # if no properties provided
        if 'properties' not in self.properties_info:
            self.properties_info = {'properties': ['energies'],
                                    'padding_values': [None],
                                    'padded_shapes': [(self.batch_size, )],
                                    'dtypes': [torch.float64],
                                    }

    def check_properties(self):
        # print properties information
        print('... The following properties will be loaded:')
        for i, prop in enumerate(self.properties_info['properties']):
            self.data_properties[prop] = []
            message = '{}: (dtype: {}, padding_value: {}, padded_shape: {})'
            print(message.format(prop, self.properties_info['dtypes'][i],
                                 self.properties_info['padding_values'][i],
                                 self.properties_info['padded_shapes'][i]))

    def get_properties_info(self):
        return self.properties_info


def collate_fn(data, chunk_threshold, properties_info):
    """Creates a batch of chunked data.
    """

    # unzip a batch of molecules (each molecule is a list)
    batch_species, batch_coordinates, batch_properties = zip(*data)

    batch_size = len(batch_species)

    # padding - time: 13.2s
    batch_species = torch.nn.utils.rnn.pad_sequence(batch_species,
                                                    batch_first=True,
                                                    padding_value=-1)
    batch_coordinates = torch.nn.utils.rnn.pad_sequence(batch_coordinates,
                                                        batch_first=True,
                                                        padding_value=np.inf)

    # sort - time: 0.7s
    atoms = torch.sum(~(batch_species == -1), dim=-1, dtype=torch.int32)
    sorted_atoms, sorted_atoms_idx = torch.sort(atoms)

    batch_species = torch.index_select(batch_species, dim=0, index=sorted_atoms_idx)
    batch_coordinates = torch.index_select(batch_coordinates, dim=0, index=sorted_atoms_idx)

    # get chunk size - time: 2.1s
    output, count = torch.unique(atoms, sorted=True, return_counts=True)
    counts = torch.cat((output.unsqueeze(-1).int(), count.unsqueeze(-1).int()), dim=-1)
    chunk_size_list, chunk_max_list = split_to_chunks(counts, chunk_threshold=chunk_threshold * batch_size * 20)

    # split into chunks - time: 0.3s
    chunks_batch_species = torch.split(batch_species, chunk_size_list, dim=0)
    chunks_batch_coordinates = torch.split(batch_coordinates, chunk_size_list, dim=0)

    # truncate redundant padding - time: 1.3s
    chunks_batch_species = trunc_pad(list(chunks_batch_species), padding_value=-1)
    chunks_batch_coordinates = trunc_pad(list(chunks_batch_coordinates), padding_value=np.inf)

    for i, c in enumerate(chunks_batch_coordinates):
        chunks_batch_coordinates[i] = c.reshape(c.shape[0], -1, 3)

    chunks = list(zip(chunks_batch_species, chunks_batch_coordinates))

    for i, _ in enumerate(chunks):
        chunks[i] = (chunks[i][0], chunks[i][1])

    # properties
    properties = {}
    for i, key in enumerate(properties_info['properties']):
        # get a batch of property
        prop = tuple(p[key] for p in batch_properties)
        # padding and convert to tensor
        if properties_info['padding_values'][i] is None:
            prop = torch.stack(prop)
        else:
            prop = torch.nn.utils.rnn.pad_sequence(prop,
                                                   batch_first=True,
                                                   padding_value=properties_info['padding_values'][i])
        # sort with number of atoms
        prop = torch.index_select(prop, dim=0, index=sorted_atoms_idx)
        # set property shape and dtype
        padded_shape = list(properties_info['padded_shapes'][i])
        padded_shape[0] = prop.shape[0]  # the last batch may does not have one batch data
        properties[key] = prop.reshape(padded_shape).to(properties_info['dtypes'][i])

    # return: [chunk1, chunk2, ...], {"energies", "force", ...} in which chunk1=(species, coordinates)
    # e.g. chunk1 = [[1807, 21], [1807, 21, 3]], chunk2 = [[193, 50], [193, 50, 3]]
    # 'energies' = [2000, 1]
    return chunks, properties


def trunc_pad(chunks, padding_value=0):
    for i, _ in enumerate(chunks):
        lengths = torch.sum(~(chunks[i] == padding_value), dim=-1, dtype=torch.int32)
        chunks[i] = chunks[i][..., :lengths.max()]
    return chunks


__all__ = ['load_ani_dataset', 'PaddedDataset', 'CachedDataset', 'ShuffledDataset', 'find_threshold']
