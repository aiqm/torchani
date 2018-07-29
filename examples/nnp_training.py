import os
import sys
import torch
import pickle
import ignite
import torchani
import torchani.data

chunk_size = 256
path = os.path.dirname(os.path.realpath(__file__))
dataset_checkpoint = os.path.join(path, '../dataset-checkpoint.dat')

if len(sys.argv) == 1:
    dataset_path = os.path.join(path, '../dataset')
elif len(sys.argv) == 2:
    dataset_path = sys.argv[1]
else:
    print('Usage:')
    print('python nnp_training.py [dataset path]')
    print('If no dataset path specified, the build-in dataset will be used')
    raise TypeError('Can only take 0 or 1 arguments')

training, validation, testing = torchani.data.maybe_create_checkpoint(
    dataset_checkpoint, dataset_path, chunk_size)
training = torchani.data.dataloader(training, chunk_size)
validation = torch.utils.data.DataLoader(validation, batch_size=4, shuffle=True, collate_fn=lambda x:x)


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