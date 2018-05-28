import torch
import torchani
import torchani.data
import math
import timeit
import itertools
import os
import sys
import pickle
from tensorboardX import SummaryWriter
from tqdm import tqdm

class Averager:
    
    def __init__(self):
        self.count = 0
        self.subtotal = 0
        
    def add(self, count, subtotal):
        self.count += count
        self.subtotal += subtotal
        
    def avg(self):
        return self.subtotal / self.count


aev_computer = torchani.SortedAEV()

def celu(x, alpha):
    return torch.where(x > 0, x, alpha * (torch.exp(x/alpha)-1))

class AtomicNetwork(torch.nn.Module):
    
    def __init__(self):
        super(AtomicNetwork, self).__init__()
        self.output_length = 1
        self.layer1 = torch.nn.Linear(384,128).type(aev_computer.dtype).to(aev_computer.device)
        self.layer2 = torch.nn.Linear(128,128).type(aev_computer.dtype).to(aev_computer.device)
        self.layer3 = torch.nn.Linear(128,64).type(aev_computer.dtype).to(aev_computer.device)
        self.layer4 = torch.nn.Linear(64,1).type(aev_computer.dtype).to(aev_computer.device)
        
    def forward(self, aev):
        y = aev
        y = self.layer1(y)
        y = celu(y, 0.1)
        y = self.layer2(y)
        y = celu(y, 0.1)
        y = self.layer3(y)
        y = celu(y, 0.1)
        y = self.layer4(y)
        return y

model = torchani.ModelOnAEV(aev_computer, reducer=torch.sum,
                            per_species = {
                                'C' : AtomicNetwork(),
                                'H' : AtomicNetwork(),
                                'N' : AtomicNetwork(),
                                'O' : AtomicNetwork(),
                            })

energy_shifter = torchani.EnergyShifter()

loss = torch.nn.MSELoss(size_average=False)

chunk_size = 256
batch_chunks = 1024 // chunk_size

with open('dataset.dat', 'rb') as f:
    training, validation, testing = pickle.load(f)

    training_sampler = torchani.data.BatchSampler(training, chunk_size, batch_chunks)
    validation_sampler = torchani.data.BatchSampler(validation, chunk_size, batch_chunks)
    testing_sampler = torchani.data.BatchSampler(testing, chunk_size, batch_chunks)

    training_dataloader = torch.utils.data.DataLoader(training, batch_sampler=training_sampler, collate_fn=torchani.data.collate)
    validation_dataloader = torch.utils.data.DataLoader(validation, batch_sampler=validation_sampler, collate_fn=torchani.data.collate)
    testing_dataloader = torch.utils.data.DataLoader(testing, batch_sampler=testing_sampler, collate_fn=torchani.data.collate)

writer = SummaryWriter()

optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)
step = 0
epoch = 0
    
def evaluate(coordinates, energies, species):
    count = coordinates.shape[0]
    pred = model(coordinates, species).squeeze()
    pred = energy_shifter.add_sae(pred, species)
    squared_error = loss(pred, energies) / len(species)
    return count, squared_error

def subset_rmse(subset_dataloader):
    a = Averager()
    for batch in subset_dataloader:
        for molecule_id in batch:
            _species = subset_dataloader.dataset.species[molecule_id]
            coordinates, energies = batch[molecule_id]
            coordinates = coordinates.to(aev_computer.device).detach()
            energies = energies.to(aev_computer.device).detach()
            a.add(*evaluate(coordinates, energies, _species))
    mse = a.avg()
    rmse = math.sqrt(mse.item()) * 627.509
    return rmse

def optimize_step(a):
    mse = a.avg()
    rmse = math.sqrt(mse.item()) * 627.509
    writer.add_scalar('training_rmse_vs_step', rmse, step)
    loss = mse if epoch < 10 else 0.5 * torch.exp(2 * mse)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

best_validation_rmse = math.inf
best_epoch = 0
start = timeit.default_timer()
while True:
    for batch in tqdm(training_dataloader, desc='epoch {}'.format(epoch), total=len(training_sampler)):
        a = Averager()
        for molecule_id in batch:
            _species = training.species[molecule_id]
            coordinates, energies = batch[molecule_id]
            coordinates = coordinates.to(aev_computer.device)
            energies = energies.to(aev_computer.device)
            a.add(*evaluate(coordinates, energies, _species))
        optimize_step(a)
        step += 1

    validation_rmse = subset_rmse(validation_dataloader)
    elapsed = round(timeit.default_timer() - start, 2)
    print('Epoch:', epoch, 'time:', elapsed, 'validation rmse:', validation_rmse)
    writer.add_scalar('validation_rmse_vs_epoch', validation_rmse, epoch)
    writer.add_scalar('epoch_vs_step', epoch, step)
    writer.add_scalar('time_vs_epoch', elapsed, epoch)
    
    if validation_rmse < best_validation_rmse:
        best_validation_rmse = validation_rmse
        best_epoch = epoch
        writer.add_scalar('best_validation_rmse_vs_epoch', best_validation_rmse, best_epoch)
    elif epoch - best_epoch > 1000:
        print('Stop at best validation rmse:', best_validation_rmse)
        break

    epoch += 1
    
testing_rmse = subset_rmse(testing_dataloader)
print('Test rmse:', validation_rmse)