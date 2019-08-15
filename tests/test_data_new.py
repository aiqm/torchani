import torchani
import unittest
import pkbar

# TODO all chunks size together should be batch size
# TODO dtype check
# TODO check no redundant padding
# TODO no calculation is need for split chunks, use probability to split

dspath = '/home/richard/dev/torchani/dataset/ani-1x/ANI-1x_complete.h5'
batch_size = 2560


class TestData(unittest.TestCase):

    def setUp(self):
        print('1. test of shuffled dataset')
        self.ds = torchani.data.ShuffleDataset(dspath, batch_size=batch_size, num_workers=2, test_bar_max=None)
        print('=> the first batch is ([chunk1, chunk2, ...], {"energies", "force", ...}) in which chunk1=(species, coordinates)')
        chunks, properties = iter(self.ds).next()
        for i, chunk in enumerate(chunks):
            print('chunk{}'.format(i + 1), list(chunk[0].size()), chunk[0].dtype, list(chunk[1].size()), chunk[1].dtype)

        for key, value in properties.items():
            print(key, list(value.size()))

        pbar = pkbar.Pbar('=> loading and processing dataset into cpu memory, total '
                          + 'batches: {}, batch_size: {}'.format(len(self.ds), batch_size),
                          len(self.ds))
        for i, t in enumerate(self.ds):
            pbar.update(i)

    def _assertTensorEqual(self, t1, t2):
        self.assertLess((t1 - t2).abs().max().item(), 1e-6)


if __name__ == "__main__":
    unittest.main()

    test1 = True
    test2 = False

    if test2:
        print('2. test of cached dataset\n')
        dataset = torchani.data.CacheDataset(dspath, batch_size=2000, device='cpu', bar=20, test_bar_max=None)

        pbar = pkbar.Pbar('=> processing and caching dataset into cpu memory, total batches:'
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

        pbar = torchani.utils.Progressbar('=> test of loading all cached dataset', len(dataset))
        for i, d in enumerate(dataset):
            pbar.update(i)
