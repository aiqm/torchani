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
    chunksize = 4
    threshold = 1e-5

    class TestIgnite(unittest.TestCase):

        def testIgnite(self):
            aev_computer = torchani.SortedAEV()
            prepare = torchani.PrepareInput(aev_computer.species)
            nnp = torchani.models.NeuroChemNNP(aev_computer.species)
            shift_energy = torchani.EnergyShifter(aev_computer.species)
            ds = torchani.data.ANIDataset(
                path, chunksize,
                transform=[shift_energy.subtract_from_dataset])
            ds = torch.utils.data.Subset(ds, [0])
            loader = torchani.data.dataloader(ds, 1)

            class Flatten(torch.nn.Module):
                def forward(self, x):
                    return x[0], x[1].flatten()

            model = torch.nn.Sequential(prepare, aev_computer, nnp, Flatten())
            container = torchani.ignite.Container({'energies': model})
            optimizer = torch.optim.Adam(container.parameters())
            trainer = create_supervised_trainer(
                container, optimizer, torchani.ignite.MSELoss('energies'))
            evaluator = create_supervised_evaluator(container, metrics={
                'RMSE': torchani.ignite.RMSEMetric('energies')
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
