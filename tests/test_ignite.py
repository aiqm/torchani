import sys

if sys.version_info.major >= 3:
    import os
    import unittest
    import torch
    from ignite.engine import create_supervised_trainer
    import torchani
    import torchani.data

    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, 'dataset/ani_gdb_s01.h5')
    chunksize = 32
    batch_chunks = 32
    dtype = torch.float32
    device = torch.device('cpu')

    class TestIgnite(unittest.TestCase):

        def testIgnite(self):
            ds = torchani.data.ANIDataset(path, chunksize)
            loader = torchani.data.dataloader(ds, batch_chunks)
            aev_computer = torchani.SortedAEV(dtype=dtype, device=device)
            nnp = torchani.models.NeuroChemNNP(aev_computer)

            class Flatten(torch.nn.Module):

                def __init__(self, model):
                    super(Flatten, self).__init__()
                    self.model = model

                def forward(self, *input):
                    return self.model(*input).flatten()
            nnp = Flatten(nnp)
            batch_nnp = torchani.models.BatchModel(nnp)
            container = torchani.ignite.Container({'energies': batch_nnp})
            loss = torchani.ignite.DictLosses({'energies': torch.nn.MSELoss()})
            optimizer = torch.optim.SGD(container.parameters(),
                                        lr=0.001, momentum=0.8)
            trainer = create_supervised_trainer(container, optimizer, loss)
            trainer.run(loader, max_epochs=10)

    if __name__ == '__main__':
        unittest.main()
