import torch
import torchani
import os

class Averager:

    def __init__(self):
        self.count = 0
        self.subtotal = 0

    def add(self, count, subtotal):
        self.count += count
        self.subtotal += subtotal

    def avg(self):
        return self.subtotal / self.count


def celu(x, alpha):
    return torch.where(x > 0, x, alpha * (torch.exp(x/alpha)-1))


class AtomicNetwork(torch.nn.Module):

    def __init__(self, aev_computer):
        super(AtomicNetwork, self).__init__()
        self.aev_computer = aev_computer
        self.output_length = 1
        self.layer1 = torch.nn.Linear(384, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, 64)
        self.layer4 = torch.nn.Linear(64, 1)

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


def get_or_create_model(filename, benchmark=False, device=configs.device):
    aev_computer = torchani.SortedAEV(benchmark=benchmark, device=device)
    model = torchani.models.CustomModel(
        aev_computer,
        reducer=torch.sum,
        benchmark=benchmark,
        per_species={
            'C': AtomicNetwork(aev_computer),
            'H': AtomicNetwork(aev_computer),
            'N': AtomicNetwork(aev_computer),
            'O': AtomicNetwork(aev_computer),
        })
    if os.path.isfile(filename):
        model.load_state_dict(torch.load(filename))
    else:
        torch.save(model.state_dict(), filename)
    return model


energy_shifter = torchani.EnergyShifter()

loss = torch.nn.MSELoss(size_average=False)


def evaluate(model, coordinates, energies, species):
    count = coordinates.shape[0]
    pred = model(coordinates, species).squeeze()
    pred = energy_shifter.add_sae(pred, species)
    squared_error = loss(pred, energies)
    return count, squared_error
