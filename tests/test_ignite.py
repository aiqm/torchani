import sys

if sys.version_info.major >= 3:
    import os
    import unittest
    import torch
    from ignite.engine import create_supervised_trainer, \
        create_supervised_evaluator, Events
    import torchani
    import torchani.data

    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, '../dataset/ani_gdb_s01.h5')
    chunksize = 16
    threshold = 1e-5
    dtype = torch.float32
    device = torch.device('cpu')

    class TestIgnite(unittest.TestCase):

        def testIgnite(self):
            shift_energy = torchani.EnergyShifter()
            ds = torchani.data.ANIDataset(
                path, chunksize, device=device,
                transform=[shift_energy.dataset_subtract_sae])
            ds = torch.utils.data.Subset(ds, [0])
            loader = torchani.data.dataloader(ds, 1)
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
            optimizer = torch.optim.Adam(container.parameters())
            trainer = create_supervised_trainer(
                container, optimizer, torchani.ignite.energy_mse_loss)
            evaluator = create_supervised_evaluator(container, metrics={
                'RMSE': torchani.ignite.energy_rmse_metric
            })

            @trainer.on(Events.COMPLETED)
            def completes(trainer):
                evaluator.run(loader)
                metrics = evaluator.state.metrics
                self.assertLess(metrics['RMSE'], threshold)
                self.assertLess(trainer.state.output, threshold)
            
            trainer.run(loader, max_epochs=1000)

    if __name__ == '__main__':
        unittest.main()
