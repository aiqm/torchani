import os
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
training, validation, testing = torchani.data.create_or_load(
    dataset_checkpoint, dataset_path,
    chunk_sizetransform=[shift_energy.load_or_create])
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

loss = torchani.ignite.DictLosses({'energies': torch.nn.MSELoss()})
optimizer = torch.optim.Adam(nnp.parameters())

trainer = ignite.engine.create_supervised_trainer(container, optimizer, loss)
trainer.run(training, max_epochs=100)

class Averager:

    def __init__(self):
        self.count = 0
        self.subtotal = 0

    def add(self, count, subtotal):
        self.count += count
        self.subtotal += subtotal

    def avg(self):
        return self.subtotal / self.count


energy_shifter = torchani.EnergyShifter()

loss = torch.nn.MSELoss(size_average=False)


def evaluate(model, coordinates, energies, species):
    count = coordinates.shape[0]
    pred = model(coordinates, species).squeeze()
    pred = energy_shifter.add_sae(pred, species)
    squared_error = loss(pred, energies)
    return count, squared_error
