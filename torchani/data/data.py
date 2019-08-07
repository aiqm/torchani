import numpy as np
import torch
import functools
import time
import sys
from _pyanitools import anidataloader


class Progressbar(object):
    """ Personal progress bar
    Usage example
    ```
    pbar = ANI_D.utils.Progressbar('loading and processing dataset', 10)
    import time
    for i in range(10):
        time.sleep(0.1)
        pbar.update(i)
    ```
    """
    def __init__(self, name, target):
        self.name = name
        self.target = target
        self.start = time.time()
        self.numdigits = int(np.log10(self.target)) + 1
        self.width = 30
        print(self.name)

    def update(self, step):

        bar = ('%' + str(self.numdigits) + 'd/%d ') % (step + 1, self.target)

        status = ""

        if step < 0:
            step = 0
            status = "negtive?...\r\n"

        stop = time.time()

        status = '- {:.1f}s'.format((stop - self.start))

        progress = float(step + 1) / self.target

        # prog
        prog_width = int(self.width * progress)
        prog = ''
        if prog_width > 0:
            prog += ('=' * (prog_width - 1))
            if step + 1 < self.target:
                prog += '>'
            else:
                prog += '='
        prog += ('.' * (self.width - prog_width))

        # text = "\r{0} {1} [{2}] {3:.0f}% {4}".format(self.name, bar, prog, pregress, status)

        text = "\r{0} [{1}] {2}".format(bar, prog, status)
        sys.stdout.write(text)
        if step + 1 == self.target:
            sys.stdout.write('\n')
        sys.stdout.flush()


class CacheDataset(torch.utils.data.Dataset):

    def __init__(self, file_path, batch_size, device, bar, test_bar_max):

        super(CacheDataset, self).__init__()

        species_dict = {'H': 0, 'C': 1, 'N': 2, 'O': 3}

        self.data_species = []
        self.data_coordinates = []
        self.data_energies = []

        anidata = anidataloader(file_path)
        pbar = Progressbar('=> loading h5 dataset into cpu memory, total molecules: {}'.format(anidata.group_size()), anidata.group_size())

        for i, molecule in enumerate(anidata):
            num_conformations = len(molecule['coordinates'])
            self.data_coordinates += list(molecule['coordinates'].reshape(num_conformations, -1))
            self.data_energies += list(molecule['energies'].reshape((-1, 1)))
            species = np.array([species_dict[x] for x in molecule['species']])
            self.data_species += list(np.tile(species, (num_conformations, 1)))
            pbar.update(i)

        self.batch_size = batch_size
        self.length = (len(self.data_species) + self.batch_size - 1) // self.batch_size
        self.device = device

        self.bar = bar
        if not self.bar:
            self.bar = 1000

        self.shuffled_index = np.arange(len(self.data_species))
        np.random.shuffle(self.shuffled_index)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):

        if index >= self.length:
            raise IndexError()

        batch_indices = slice(index * self.batch_size, (index + 1) * self.batch_size)
        batch_indices_shuffled = self.shuffled_index[batch_indices]

        batch_species = [self.data_species[i] for i in batch_indices_shuffled]
        batch_coordinates = [self.data_coordinates[i] for i in batch_indices_shuffled]
        batch_energies = [self.data_energies[i] for i in batch_indices_shuffled]
        # batch_species = self.data_species[batch_indices]
        # batch_coordinates = self.data_coordinates[batch_indices]
        # batch_energies = self.data_energies[batch_indices]

        datas = [batch_species, batch_coordinates, batch_energies]

        # get sort index
        num_atoms_each_mole = [b.shape[0] for b in batch_species]
        atoms = torch.tensor(num_atoms_each_mole, dtype=torch.int32)
        sorted_atoms, sorted_atoms_idx = torch.sort(atoms, descending=False)

        # sort each batch of data
        for i, d in enumerate(datas):
            datas[i] = self.sort_list_with_index(d, sorted_atoms_idx.numpy())

        # get chunk size
        output, count = torch.unique(atoms, sorted=True, return_counts=True)
        counts = torch.cat((output.unsqueeze(-1).int(), count.unsqueeze(-1).int()), dim=-1)
        chunk_size_list, chunk_max_list = split_to_chunks(counts, bar=self.bar * self.batch_size * 20)
        chunk_size_list = torch.stack(chunk_size_list).flatten()
        chunk_max_list = torch.stack(chunk_max_list).flatten()

        # split into chunks
        for i, d in enumerate(datas):
            datas[i] = self.split_list_with_size(d, chunk_size_list.numpy())

        # padding each data
        # batch_species
        datas[0] = self.pad_and_convert_to_tensor(datas[0], padding_value=-1)
        # batch_coordinates
        datas[1] = self.pad_and_convert_to_tensor(datas[1])
        # batch_energies
        datas[2] = self.pad_and_convert_to_tensor(datas[2], no_padding=True)

        # return: [chunk1, chunk2, ...] in which chunk1=(coordinates, species, energies)
        chunks = list(zip(*datas))

        for i in range(len(chunks)):
            chunks[i] = {'coordinates': chunks[i][0], 'species': chunks[i][1], 'energies': chunks[i][2]}

        return chunks

    def __len__(self):
        return self.length

    @staticmethod
    def sort_list_with_index(input, index):

        return [input[i] for i in index]

    @staticmethod
    def split_list_with_size(input, split_size):

        output = []
        # split_size = np.array(split_size)
        for i, value in enumerate(split_size):
            start_index = np.sum(split_size[:i])
            stop_index = np.sum(split_size[:i + 1])
            output.append(input[start_index:stop_index])
        return output

    def pad_and_convert_to_tensor(self, inputs, padding_value=0, no_padding=False):

        if no_padding:
            for i, input_tmp in enumerate(inputs):
                inputs[i] = torch.from_numpy(np.stack(input_tmp)).to(self.device)
            return inputs
        else:
            for i, input_tmp in enumerate(inputs):
                inputs[i] = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(b) for b in inputs[i]],
                                                            batch_first=True,
                                                            padding_value=padding_value).to(self.device)
            return inputs

    def release_h5(self):
        import gc
        del self.data_species
        del self.data_coordinates
        del self.data_energies
        gc.collect()


def split_to_two_chunks(counts, bar):

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


if __name__ == "__main__":

    dspath = '/home/richard/dev/torchani/dataset/ani-1x/ANI-1x_complete.h5'
    dataset = CacheDataset(dspath, batch_size=2000, device='cpu', bar=20, test_bar_max=None)

    pbar = Progressbar('=> processing and caching dataset into cpu memory, total batches:'
                       + ' {}, batch_size: {}'.format(len(dataset), dataset.batch_size),
                       len(dataset))
    for i, d in enumerate(dataset):
        pbar.update(i)
    total_chunks = sum([len(d) for d in dataset])
    chunks_size = str([list(c['species'].size()) for c in dataset[0]])

    print('=> dataset cached, total chunks: '
          + '{}, first batch is splited to {}'.format(total_chunks, chunks_size))
    print('=> releasing h5 file memory, dataset is still cached')
    dataset.release_h5()

    pbar = Progressbar('=> test of loading all cached dataset', len(dataset))
    for i, d in enumerate(dataset):
        pbar.update(i)
