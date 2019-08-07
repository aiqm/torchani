import os
import unittest
import torch
import copy
from ignite.engine import create_supervised_trainer, \
    create_supervised_evaluator, Events
import torchani
import torchani.ignite

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '../dataset/ani1-up_to_gdb4/ani_gdb_s01.h5')
batchsize = 4
threshold = 1e-5


class TestIgnite(unittest.TestCase):

    def testIgnite(self):
        ani1x = torchani.models.ANI1x()
        aev_computer = ani1x.aev_computer
        nnp = copy.deepcopy(ani1x.neural_networks[0])
        shift_energy = ani1x.energy_shifter
        ds = torchani.data.load_ani_dataset(
            path, ani1x.consts.species_to_tensor, batchsize,
            transform=[shift_energy.subtract_from_dataset],
            device=aev_computer.EtaR.device)
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
