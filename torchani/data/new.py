import numpy as np
import torch
import functools
from ._pyanitools import anidataloader
import pkbar


class CacheDataset(torch.utils.data.Dataset):

    def __init__(self, file_path,
                 species_order='HCNO',
                 transform=True,
                 self_energies=[-0.600953, -38.08316, -54.707756, -75.194466],
                 batch_size=1000,
                 device='cpu',
                 bar=20,
                 test_bar_max=None):

        super(CacheDataset, self).__init__()

        # species_dict = {'H': 0, 'C': 1, 'N': 2, 'O': 3}
        # self_energies_dict = {'H': -0.600953, 'C': -38.08316, 'N': -54.707756, 'O': -75.194466}
        species_dict = {}
        self_energies_dict = {}
        for i, s in enumerate(species_order):
            species_dict[s] = i
            self_energies_dict[s] = self_energies[i]

        self.data_species = []
        self.data_coordinates = []
        self.data_energies = []
        self.data_self_energies = []

        anidata = anidataloader(file_path)
        anidata_size = anidata.group_size()
        if anidata_size > 5:
            pbar = pkbar.Pbar('=> loading h5 dataset into cpu memory, total molecules: {}'.format(anidata_size), anidata_size)

        for i, molecule in enumerate(anidata):
            num_conformations = len(molecule['coordinates'])
            self.data_coordinates += list(molecule['coordinates'].reshape(num_conformations, -1))
            self.data_energies += list(molecule['energies'].reshape((-1, 1)))
            species = np.array([species_dict[x] for x in molecule['species']])
            self.data_species += list(np.tile(species, (num_conformations, 1)))
            if transform:
                self_energies = np.array(sum([self_energies_dict[x] for x in molecule['species']]))
                self.data_self_energies += list(np.tile(self_energies, (num_conformations, 1)))
            if anidata_size > 5:
                pbar.update(i)

        if transform:
            self.data_energies = np.array(self.data_energies) - np.array(self.data_self_energies)
            import gc
            del self.data_self_energies
            del self_energies
            gc.collect()

        self.batch_size = batch_size
        self.length = (len(self.data_species) + self.batch_size - 1) // self.batch_size
        self.device = device
        self.test_bar_max = test_bar_max
        self.bar = bar
        if not self.bar:
            self.bar = 1000

        self.shuffled_index = np.arange(len(self.data_species))
        np.random.shuffle(self.shuffled_index)

        anidata.cleanup()
        import gc
        del num_conformations
        del species
        del anidata
        del molecule
        gc.collect()

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):

        if index >= self.length:
            raise IndexError()

        batch_indices = slice(index * self.batch_size, (index + 1) * self.batch_size)
        batch_indices_shuffled = self.shuffled_index[batch_indices]

        batch_species = [torch.from_numpy(self.data_species[i]) for i in batch_indices_shuffled]
        batch_coordinates = [torch.from_numpy(self.data_coordinates[i]) for i in batch_indices_shuffled]
        batch_energies = [torch.from_numpy(self.data_energies[i]) for i in batch_indices_shuffled]

        # padding - time: 13.2s
        batch_species = torch.nn.utils.rnn.pad_sequence(batch_species,
                                                        batch_first=True,
                                                        padding_value=-1).to(self.device)
        batch_coordinates = torch.nn.utils.rnn.pad_sequence(batch_coordinates,
                                                            batch_first=True,
                                                            padding_value=0).to(self.device)
        batch_energies = torch.stack(batch_energies).to(self.device)

        # sort - time: 0.7s
        atoms = torch.sum(~(batch_species == -1), dim=-1, dtype=torch.int32)
        sorted_atoms, sorted_atoms_idx = torch.sort(atoms, descending=False)

        batch_species = torch.index_select(batch_species, dim=0, index=sorted_atoms_idx)
        batch_coordinates = torch.index_select(batch_coordinates, dim=0, index=sorted_atoms_idx)
        batch_energies = torch.index_select(batch_energies, dim=0, index=sorted_atoms_idx)

        # get chunk size - time: 2.1s
        output, count = torch.unique(atoms, sorted=True, return_counts=True)
        counts = torch.cat((output.unsqueeze(-1).int(), count.unsqueeze(-1).int()), dim=-1)
        chunk_size_list, chunk_max_list = split_to_chunks(counts, bar=self.bar * self.batch_size * 20)

        # optimize bar, if test_bar_max is not None
        if self.test_bar_max:
            print('format is [chunk_size, chunk_max]')
            print('current bar = {}'.format(self.bar))
            size_max = []
            for i in range(len(chunk_size_list)):
                size_max.append([list(chunk_size_list[i])[0],
                                list(chunk_max_list[i])[0]])
            print(size_max)

            for b in range(0, self.test_bar_max + 1, 1):
                test_chunk_size_list, test_chunk_max_list = split_to_chunks(counts, bar=b * self.batch_size * 20)
                size_max = []
                for i in range(len(test_chunk_size_list)):
                    size_max.append([list(test_chunk_size_list[i])[0],
                                    list(test_chunk_max_list[i])[0]])
                print('bar = {}'.format(b))
                print(size_max)

        # split into chunks - time: 0.3s
        chunks_batch_species = torch.split(batch_species, chunk_size_list, dim=0)
        chunks_batch_coordinates = torch.split(batch_coordinates, chunk_size_list, dim=0)

        # truncate redundant padding - time: 1.3s
        chunks_batch_species = trunc_pad(list(chunks_batch_species), padding_value=-1)
        chunks_batch_coordinates = trunc_pad(list(chunks_batch_coordinates))

        for i, c in enumerate(chunks_batch_coordinates):
            chunks_batch_coordinates[i] = c.reshape(c.shape[0], -1, 3)

        datas = [chunks_batch_species, chunks_batch_coordinates]

        chunks = list(zip(*datas))

        for i in range(len(chunks)):
            chunks[i] = (chunks[i][0], chunks[i][1])

        properties = {'energies': batch_energies.flatten()}

        # return: [chunk1, chunk2, ...], {"energies", "force", ...} in which chunk1=(species, coordinates)
        # e.g. chunk1 = [[1807, 21], [1807, 21, 3]], chunk2 = [[193, 50], [193, 50, 3]]
        # 'energies' = [2000, 1]
        return chunks, properties

    def __len__(self):
        return self.length

    def release_h5(self):
        import gc
        del self.data_species
        del self.data_coordinates
        del self.data_energies
        gc.collect()


def ShuffleDataset(file_path,
                   species_order='HCNO',
                   transform=True,
                   self_energies=[-0.600953, -38.08316, -54.707756, -75.194466],
                   batch_size=1000, num_workers=0, shuffle=True,
                   bar=20, test_bar_max=None):

    dataset = TorchData(file_path, species_order, transform, self_energies)

    if test_bar_max:
        def my_collate_fn(data, bar=bar, test_bar_max=test_bar_max):
            return collate_fn(data, bar, test_bar_max)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=0,
                                                  pin_memory=False,
                                                  collate_fn=my_collate_fn)
        print('=> checking which bar should use (bar control how chunks will be splitted)')
        chunks, properties = iter(data_loader).next()

    def my_collate_fn(data, bar=bar, test_bar_max=None):
        return collate_fn(data, bar, test_bar_max)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=False,
                                              collate_fn=my_collate_fn)

    return data_loader


class TorchData(torch.utils.data.Dataset):

    def __init__(self, file_path, species_order, transform, self_energies):

        super(TorchData, self).__init__()

        # species_dict = {'H': 0, 'C': 1, 'N': 2, 'O': 3}
        # self_energies_dict = {'H': -0.600953, 'C': -38.08316, 'N': -54.707756, 'O': -75.194466}
        species_dict = {}
        self_energies_dict = {}
        for i, s in enumerate(species_order):
            species_dict[s] = i
            self_energies_dict[s] = self_energies[i]

        self.data_species = []
        self.data_coordinates = []
        self.data_energies = []
        self.data_self_energies = []

        anidata = anidataloader(file_path)
        anidata_size = anidata.group_size()
        if anidata_size > 5:
            pbar = pkbar.Pbar('=> loading h5 dataset into cpu memory, total molecules: {}'.format(anidata_size), anidata_size)

        for i, molecule in enumerate(anidata):
            num_conformations = len(molecule['coordinates'])
            self.data_coordinates += list(molecule['coordinates'].reshape(num_conformations, -1))
            self.data_energies += list(molecule['energies'].reshape((-1, 1)))
            species = np.array([species_dict[x] for x in molecule['species']])
            self.data_species += list(np.tile(species, (num_conformations, 1)))
            if transform:
                self_energies = np.array(sum([self_energies_dict[x] for x in molecule['species']]))
                self.data_self_energies += list(np.tile(self_energies, (num_conformations, 1)))
            if anidata_size > 5:
                pbar.update(i)

        if transform:
            self.data_energies = np.array(self.data_energies) - np.array(self.data_self_energies)
            import gc
            del self.data_self_energies
            del self_energies
            gc.collect()

        self.length = len(self.data_species)
        anidata.cleanup()

        import gc
        del num_conformations
        del species
        del anidata
        del molecule
        gc.collect()

    def __getitem__(self, index):

        if index >= self.length:
            raise IndexError()

        species = torch.from_numpy(self.data_species[index])
        coordinates = torch.from_numpy(self.data_coordinates[index]).float()
        energies = torch.from_numpy(self.data_energies[index]).float()

        return [species, coordinates, energies]

    def __len__(self):
        return self.length


def collate_fn(data, bar, test_bar_max):
    """Creates a batch of chunked data.
    Args:
        data: list of molecules, each molecule is a list.
              which contain [species, coordinates, energies, MO_energies]
            - species: numpy array, shape (?); variable length.
            - coordinates: numpy array, shape (?); variable length. (flattened)
            - energies: numpy array, shape (1); variable length.
            - MO_energies: numpy array, shape (?); variable length.
    Returns:
            - [chunk1, chunk2, ...] in which chunk1=(coordinates, species, energies)
    """

    # unzip a batch of molecules (each molecule is a list)
    batch_species, batch_coordinates, batch_energies = zip(*data)
    batch_size = len(batch_species)

    # padding - time: 13.2s
    batch_species = torch.nn.utils.rnn.pad_sequence(batch_species,
                                                    batch_first=True,
                                                    padding_value=-1)
    batch_coordinates = torch.nn.utils.rnn.pad_sequence(batch_coordinates,
                                                        batch_first=True,
                                                        padding_value=0)
    batch_energies = torch.stack(batch_energies)

    # sort - time: 0.7s
    atoms = torch.sum(~(batch_species == -1), dim=-1, dtype=torch.int32)
    sorted_atoms, sorted_atoms_idx = torch.sort(atoms, descending=False)

    batch_species = torch.index_select(batch_species, dim=0, index=sorted_atoms_idx)
    batch_coordinates = torch.index_select(batch_coordinates, dim=0, index=sorted_atoms_idx)
    batch_energies = torch.index_select(batch_energies, dim=0, index=sorted_atoms_idx)

    # get chunk size - time: 2.1s
    if not bar:
        bar = 1000
    output, count = torch.unique(atoms, sorted=True, return_counts=True)
    counts = torch.cat((output.unsqueeze(-1).int(), count.unsqueeze(-1).int()), dim=-1)
    chunk_size_list, chunk_max_list = split_to_chunks(counts, bar=bar * batch_size * 20)

    # optimize bar, if test_bar_max is not None
    if test_bar_max:
        print('format is [chunk_size, chunk_max]')
        print('current bar = {}'.format(bar))
        size_max = []
        for i in range(len(chunk_size_list)):
            size_max.append([list(chunk_size_list[i].numpy())[0],
                             list(chunk_max_list[i].numpy())[0]])
        print(size_max)

        for b in range(0, test_bar_max + 1, 1):
            test_chunk_size_list, test_chunk_max_list = split_to_chunks(counts, bar=b * batch_size * 20)
            size_max = []
            for i in range(len(test_chunk_size_list)):
                size_max.append([list(test_chunk_size_list[i].numpy())[0],
                                 list(test_chunk_max_list[i].numpy())[0]])
            print('bar = {}'.format(b))
            print(size_max)

    # split into chunks - time: 0.3s
    chunks_batch_species = torch.split(batch_species, chunk_size_list, dim=0)
    chunks_batch_coordinates = torch.split(batch_coordinates, chunk_size_list, dim=0)

    # truncate redundant padding - time: 1.3s
    chunks_batch_species = trunc_pad(list(chunks_batch_species), padding_value=-1)
    chunks_batch_coordinates = trunc_pad(list(chunks_batch_coordinates))

    for i, c in enumerate(chunks_batch_coordinates):
        chunks_batch_coordinates[i] = c.reshape(c.shape[0], -1, 3)

    datas = [chunks_batch_species, chunks_batch_coordinates]

    chunks = list(zip(*datas))

    for i in range(len(chunks)):
        chunks[i] = (chunks[i][0], chunks[i][1])

    properties = {'energies': batch_energies.flatten()}

    # return: [chunk1, chunk2, ...], {"energies", "force", ...} in which chunk1=(species, coordinates)
    # e.g. chunk1 = [[1807, 21], [1807, 21, 3]], chunk2 = [[193, 50], [193, 50, 3]]
    # 'energies' = [2000, 1]
    return chunks, properties


def split_to_two_chunks(counts, bar):

    counts = counts.cpu()
    left_mask = torch.triu(torch.ones([counts.shape[0], counts.shape[0]], dtype=torch.uint8))
    left_mask = torch.transpose(left_mask, dim0=0, dim1=1)

    counts_atoms = counts[:, 0].repeat(counts.shape[0], 1)
    counts_counts = counts[:, 1].repeat(counts.shape[0], 1)

    counts_atoms_left = torch.where(left_mask, counts_atoms, torch.zeros_like(counts_atoms))
    counts_atoms_right = torch.where(~left_mask, counts_atoms, torch.zeros_like(counts_atoms))
    counts_counts_left = torch.where(left_mask, counts_counts, torch.zeros_like(counts_atoms))
    counts_counts_right = torch.where(~left_mask, counts_counts, torch.zeros_like(counts_atoms))

    # chunk max
    chunk_max_left, _ = torch.max(counts_atoms_left, dim=-1, keepdim=True)
    chunk_max_right, _ = torch.max(counts_atoms_right, dim=-1, keepdim=True)

    # chunk size
    chunk_size_left = torch.sum(counts_counts_left, dim=-1, keepdim=True, dtype=torch.int32)
    chunk_size_right = torch.sum(counts_counts_right, dim=-1, keepdim=True, dtype=torch.int32)

    # calculate cost
    min_cost_bar = torch.tensor([bar], dtype=torch.int32)
    cost = (torch.max(chunk_size_left * chunk_max_left * chunk_max_left, min_cost_bar)
            + torch.max(chunk_size_right * chunk_max_right * chunk_max_right, min_cost_bar))

    # find smallest cost
    cost_min, cost_min_index = torch.min(cost.squeeze(), dim=-1)

    # find smallest cost chunk_size, if not splitted, it will be [max_chunk_size, 0]
    final_chunk_size = [chunk_size_left[cost_min_index], chunk_size_right[cost_min_index]]
    final_chunk_max = [chunk_max_left[cost_min_index], chunk_max_right[cost_min_index]]

    # if not splitted
    if cost_min_index == (counts.shape[0] - 1):
        return False, counts, [final_chunk_size[0]], [final_chunk_max[0]], cost_min
    # if splitted
    else:
        return True, [counts[:cost_min_index + 1], counts[(cost_min_index + 1):]], \
            final_chunk_size, final_chunk_max, cost_min


def split_to_chunks(counts, bar=50000):

    splitted, counts_list, chunk_size, chunk_max, cost = split_to_two_chunks(counts, bar)
    final_chunk_size = []
    final_chunk_max = []

    if (splitted):
        for i in range(len(counts_list)):
            tmp_chunk_size, tmp_chunk_max = split_to_chunks(counts_list[i], bar)
            final_chunk_size.extend(tmp_chunk_size)
            final_chunk_max.extend(tmp_chunk_max)
        return final_chunk_size, final_chunk_max
    else:
        return chunk_size, chunk_max


def trunc_pad(chunks, padding_value=0):
    for i in range(len(chunks)):
        lengths = torch.sum(~(chunks[i] == padding_value), dim=-1, dtype=torch.int32)
        chunks[i] = chunks[i][..., :lengths.max()]
    return chunks
