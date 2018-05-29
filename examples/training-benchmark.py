import torch
import torchani
import torchani.data
import tqdm
import math
import timeit

class Averager:
    
    def __init__(self):
        self.count = 0
        self.subtotal = 0
        
    def add(self, count, subtotal):
        self.count += count
        self.subtotal += subtotal
        
    def avg(self):
        return self.subtotal / self.count

aev_computer = torchani.SortedAEV(benchmark=True)

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
                            benchmark=True,
                            per_species = {
                                'C' : AtomicNetwork(),
                                'H' : AtomicNetwork(),
                                'N' : AtomicNetwork(),
                                'O' : AtomicNetwork(),
                            })
energy_shifter = torchani.EnergyShifter()
loss = torch.nn.MSELoss(size_average=False)

ds = torchani.data.load_dataset('../ANI-1x_complete.h5')
sampler = torchani.data.BatchSampler(ds, 256, 4)
dataloader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=torchani.data.collate)

optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)
    
def evaluate(coordinates, energies, species):
    count = coordinates.shape[0]
    pred = model(coordinates, species).squeeze()
    pred = energy_shifter.add_sae(pred, species)
    squared_error = loss(pred, energies)
    return count, squared_error

def optimize_step(a):
    mse = a.avg()
    optimizer.zero_grad()
    mse.backward()
    optimizer.step()

start = timeit.default_timer()
for batch in tqdm.tqdm(dataloader, total=len(sampler)):
    a = Averager()
    for molecule_id in batch:
        _species = ds.species[molecule_id]
        coordinates, energies = batch[molecule_id]
        coordinates = coordinates.to(aev_computer.device)
        energies = energies.to(aev_computer.device)
        a.add(*evaluate(coordinates, energies, _species))
    optimize_step(a)

elapsed = round(timeit.default_timer() - start, 2)
print('Epoch time:', elapsed)
print('Radial terms:', aev_computer.timers['radial terms'])
print('Angular terms:', aev_computer.timers['angular terms'])
print('Terms and indices:', aev_computer.timers['terms and indices'])
print('Partition:', aev_computer.timers['partition'])
print('Assemble:', aev_computer.timers['assemble'])
print('Total AEV:', aev_computer.timers['total'])
print('NN:', model.timers['nn'])
print('Total Forward:', model.timers['forward'])