import sys

if sys.version_info.major >= 3:
    import os
    import unittest
    import torch
    import torchani
    import torchani.data
    import itertools

    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, '../dataset')
    chunksize = 32
    batch_chunks = 32
    dtype = torch.float32
    device = torch.device('cpu')

    class TestBatch(unittest.TestCase):

        def testBatchLoadAndInference(self):
            ds = torchani.data.ANIDataset(path, chunksize, device=device)
            loader = torchani.data.dataloader(ds, batch_chunks)
            aev_computer = torchani.SortedAEV(dtype=dtype, device=device)
            prepare = torchani.PrepareInput(aev_computer.species,
                                            aev_computer.device)
            nnp = torchani.models.NeuroChemNNP(aev_computer.species)
            model = torch.nn.Sequential(prepare, aev_computer, nnp)
            batch_nnp = torchani.models.BatchModel(model)
            for batch_input, batch_output in itertools.islice(loader, 10):
                batch_output_ = batch_nnp(batch_input).squeeze()
                self.assertListEqual(list(batch_output_.shape),
                                     list(batch_output['energies'].shape))

    if __name__ == '__main__':
        unittest.main()
