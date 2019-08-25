import numpy as np
import torch
import functools
from ._pyanitools import anidataloader
import importlib
import gc

PKBAR_INSTALLED = importlib.util.find_spec('pkbar') is not None
if PKBAR_INSTALLED:
    import pkbar


def find_threshold(file_path, batch_size, threshold_max=100):
    """Find resonable threshold to split chunks before using ``torchani.data.CachedDataset`` or ``torchani.data.ShuffledDataset``.

    Arguments:
        file_path (str): Path to one hdf5 files.
        batch_size (int): batch size.
        threshold_max (int): max threshould to test.
    """
    ds = CachedDataset(file_path=file_path, batch_size=batch_size)
    ds.find_threshold(threshold_max + 1)


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

        self.chunk_threshold = chunk_threshold
        if not self.chunk_threshold:
            self.chunk_threshold = np.inf

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

        # get sort index
        num_atoms_each_mole = [b.shape[0] for b in batch_species]
        atoms = torch.tensor(num_atoms_each_mole, dtype=torch.int32)
        sorted_atoms, sorted_atoms_idx = torch.sort(atoms)

        # sort each batch of data
        batch_species = self.sort_list_with_index(batch_species, sorted_atoms_idx.numpy())
        batch_coordinates = self.sort_list_with_index(batch_coordinates, sorted_atoms_idx.numpy())

        # get chunk size
        output, count = torch.unique(atoms, sorted=True, return_counts=True)
        counts = torch.cat((output.unsqueeze(-1).int(), count.unsqueeze(-1).int()), dim=-1)
        chunk_size_list, chunk_max_list = split_to_chunks(counts, chunk_threshold=self.chunk_threshold * self.batch_size * 20)
        chunk_size_list = torch.stack(chunk_size_list).flatten()

        # split into chunks
        chunks_batch_species = self.split_list_with_size(batch_species, chunk_size_list.numpy())
        chunks_batch_coordinates = self.split_list_with_size(batch_coordinates, chunk_size_list.numpy())

        # padding each data
        chunks_batch_species = self.pad_and_convert_to_tensor(chunks_batch_species, padding_value=-1)
        chunks_batch_coordinates = self.pad_and_convert_to_tensor(chunks_batch_coordinates)

        # chunks
        chunks = list(zip(chunks_batch_species, chunks_batch_coordinates))
        for i, _ in enumerate(chunks):
            chunks[i] = (chunks[i][0], chunks[i][1].reshape(chunks[i][1].shape[0], -1, 3))

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

        # return: [chunk1, chunk2, ...], {"energies", "force", ...} in which chunk1=(species, coordinates)
        # e.g. chunk1 = [[1807, 21], [1807, 21, 3]], chunk2 = [[193, 50], [193, 50, 3]]
        # 'energies' = [2000, 1]
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

    @staticmethod
    def sort_list_with_index(inputs, index):
        return [inputs[i] for i in index]

    @staticmethod
    def split_list_with_size(inputs, split_size):
        output = []
        for i, _ in enumerate(split_size):
            start_index = np.sum(split_size[:i])
            stop_index = np.sum(split_size[:i + 1])
            output.append(inputs[start_index:stop_index])
        return output

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

    def find_threshold(self, threshold_max=100):
        batch_indices = slice(0, self.batch_size)
        batch_indices_shuffled = self.shuffled_index[batch_indices]

        batch_species = [self.data_species[i] for i in batch_indices_shuffled]

        num_atoms_each_mole = [b.shape[0] for b in batch_species]
        atoms = torch.tensor(num_atoms_each_mole, dtype=torch.int32)

        output, count = torch.unique(atoms, sorted=True, return_counts=True)
        counts = torch.cat((output.unsqueeze(-1).int(), count.unsqueeze(-1).int()), dim=-1)

        print('=> choose a reasonable threshold to split chunks')
        print('format is [chunk_size, chunk_max]')

        for b in range(0, threshold_max, 1):
            test_chunk_size_list, test_chunk_max_list = split_to_chunks(counts, chunk_threshold=b * self.batch_size * 20)
            size_max = []
            for i, _ in enumerate(test_chunk_size_list):
                size_max.append([list(test_chunk_size_list[i].numpy())[0],
                                list(test_chunk_max_list[i].numpy())[0]])
            print('chunk_threshold = {}'.format(b))
            print(size_max)

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
                                                        padding_value=0)

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
    chunks_batch_coordinates = trunc_pad(list(chunks_batch_coordinates))

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
            prop = torch.nn.utils.rnn.pad_sequence(batch_species,
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


def split_to_two_chunks(counts, chunk_threshold):
    counts = counts.cpu()
    # NB (@yueyericardo): In principle this dtype should be `torch.bool`, but unfortunately
    # `triu` is not implemented for bool tensor right now. This should be fixed when PyTorch
    # add support for it.
    left_mask = torch.triu(torch.ones([counts.shape[0], counts.shape[0]], dtype=torch.uint8))
    left_mask = left_mask.t()

    counts_atoms = counts[:, 0].repeat(counts.shape[0], 1)
    counts_counts = counts[:, 1].repeat(counts.shape[0], 1)

    counts_atoms_left = torch.where(left_mask, counts_atoms, torch.zeros_like(counts_atoms))
    counts_atoms_right = torch.where(~left_mask, counts_atoms, torch.zeros_like(counts_atoms))
    counts_counts_left = torch.where(left_mask, counts_counts, torch.zeros_like(counts_atoms))
    counts_counts_right = torch.where(~left_mask, counts_counts, torch.zeros_like(counts_atoms))

    # chunk max
    chunk_max_left = torch.max(counts_atoms_left, dim=-1, keepdim=True).values
    chunk_max_right = torch.max(counts_atoms_right, dim=-1, keepdim=True).values

    # chunk size
    chunk_size_left = torch.sum(counts_counts_left, dim=-1, keepdim=True, dtype=torch.int32)
    chunk_size_right = torch.sum(counts_counts_right, dim=-1, keepdim=True, dtype=torch.int32)

    # calculate cost
    min_cost_threshold = torch.tensor([chunk_threshold], dtype=torch.int32)
    cost = (torch.max(chunk_size_left * chunk_max_left * chunk_max_left, min_cost_threshold)
            + torch.max(chunk_size_right * chunk_max_right * chunk_max_right, min_cost_threshold))

    # find smallest cost
    cost_min, cost_min_index = torch.min(cost.squeeze(), dim=-1)

    # find smallest cost chunk_size, if not splitted, it will be [max_chunk_size, 0]
    final_chunk_size = [chunk_size_left[cost_min_index], chunk_size_right[cost_min_index]]
    final_chunk_max = [chunk_max_left[cost_min_index], chunk_max_right[cost_min_index]]

    # if not splitted
    if cost_min_index == (counts.shape[0] - 1):
        return False, counts, [final_chunk_size[0]], [final_chunk_max[0]], cost_min
    # if splitted
    return True, [counts[:cost_min_index + 1], counts[(cost_min_index + 1):]], \
        final_chunk_size, final_chunk_max, cost_min


def split_to_chunks(counts, chunk_threshold=np.inf):
    splitted, counts_list, chunk_size, chunk_max, cost = split_to_two_chunks(counts, chunk_threshold)
    final_chunk_size = []
    final_chunk_max = []

    if (splitted):
        for i, _ in enumerate(counts_list):
            tmp_chunk_size, tmp_chunk_max = split_to_chunks(counts_list[i], chunk_threshold)
            final_chunk_size.extend(tmp_chunk_size)
            final_chunk_max.extend(tmp_chunk_max)
        return final_chunk_size, final_chunk_max
    # if not splitted
    return chunk_size, chunk_max


def trunc_pad(chunks, padding_value=0):
    for i, _ in enumerate(chunks):
        lengths = torch.sum(~(chunks[i] == padding_value), dim=-1, dtype=torch.int32)
        chunks[i] = chunks[i][..., :lengths.max()]
    return chunks
