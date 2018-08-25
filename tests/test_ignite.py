import os
import unittest
import torch
import copy
from ignite.engine import create_supervised_trainer, \
    create_supervised_evaluator, Events
import torchani
import torchani.ignite

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '../dataset/ani_gdb_s01.h5')
batchsize = 4
threshold = 1e-5


class TestIgnite(unittest.TestCase):

    def testIgnite(self):
        buildins = torchani.neurochem.Buildins()
        aev_computer = buildins.aev_computer
        nnp = copy.deepcopy(buildins.models[0])
        shift_energy = buildins.energy_shifter
        ds = torchani.data.BatchedANIDataset(
            path, buildins.consts.species_to_tensor, batchsize,
            transform=[shift_energy.subtract_from_dataset])
        ds = torch.utils.data.Subset(ds, [0])

        class Flatten(torch.nn.Module):
            def forward(self, x):
                return x[0], x[1].flatten()

        model = torch.nn.Sequential(aev_computer, nnp, Flatten())
        container = torchani.ignite.Container({'energies': model})
        optimizer = torch.optim.Adam(container.parameters())
        loss = torchani.ignite.TransformedLoss(
            torchani.ignite.MSELoss('energies'),
            lambda x: torch.exp(x) - 1)
        trainer = create_supervised_trainer(
            container, optimizer, loss)
        evaluator = create_supervised_evaluator(container, metrics={
            'RMSE': torchani.ignite.RMSEMetric('energies')
        })

        @trainer.on(Events.COMPLETED)
        def completes(trainer):
            evaluator.run(ds)
            metrics = evaluator.state.metrics
            self.assertLess(metrics['RMSE'], threshold)
            self.assertLess(trainer.state.output, threshold)

        trainer.run(ds, max_epochs=1000)


if __name__ == '__main__':
    unittest.main()
