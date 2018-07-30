import sys
import torch
import ignite
import torchani
import model

chunk_size = 256
batch_chunks = 4
dataset_path = sys.argv[1]
dataset_checkpoint = 'dataset-checkpoint.dat'
model_checkpoint = 'checkpoint.pt'

shift_energy = torchani.EnergyShifter()
training, validation, testing = torchani.data.load_or_create(
    dataset_checkpoint, dataset_path, chunk_size,
    transform=[shift_energy.dataset_subtract_sae])
training = torchani.data.dataloader(training, batch_chunks)
validation = torchani.data.dataloader(validation, batch_chunks)

nnp = model.get_or_create_model(model_checkpoint)


class Flatten(torch.nn.Module):

    def __init__(self, model):
        super(Flatten, self).__init__()
        self.model = model

    def forward(self, *input):
        return self.model(*input).flatten()


batch_nnp = torchani.models.BatchModel(Flatten(nnp))
container = torchani.ignite.Container({'energies': batch_nnp})

loss = torchani.ignite.DictLoss('energies', torch.nn.MSELoss())
metric = torchani.ignite.DictMetric('energies',
                                    ignite.metrics.RootMeanSquaredError())
optimizer = torch.optim.Adam(nnp.parameters())
trainer = ignite.engine.create_supervised_trainer(container, optimizer, loss)
validator = ignite.engine.create_supervised_evaluator(container, metrics={
        'RMSE': metric
    })

trainer.run(training, max_epochs=100)
