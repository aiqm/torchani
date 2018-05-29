import torchani
import torch
from configs import benchmark, device

class Averager:
    
    def __init__(self):
        self.count = 0
        self.subtotal = 0
        
    def add(self, count, subtotal):
        self.count += count
        self.subtotal += subtotal
        
    def avg(self):
        return self.subtotal / self.count


aev_computer = torchani.SortedAEV(benchmark=benchmark, device=device)

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

model = torchani.ModelOnAEV(aev_computer, reducer=torch.sum, benchmark=benchmark,
                            per_species = {
                                'C' : AtomicNetwork(),
                                'H' : AtomicNetwork(),
                                'N' : AtomicNetwork(),
                                'O' : AtomicNetwork(),
                            })

energy_shifter = torchani.EnergyShifter()

loss = torch.nn.MSELoss(size_average=False)

def evaluate(coordinates, energies, species):
    count = coordinates.shape[0]
    pred = model(coordinates, species).squeeze()
    pred = energy_shifter.add_sae(pred, species)
    squared_error = loss(pred, energies)
    return count, squared_error