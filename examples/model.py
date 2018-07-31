import torch
import torchani
import os


def celu(x, alpha):
    return torch.where(x > 0, x, alpha * (torch.exp(x/alpha)-1))


class AtomicNetwork(torch.nn.Module):

    def __init__(self):
        super(AtomicNetwork, self).__init__()
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


def get_or_create_model(filename, benchmark=False,
                        device=torchani.default_device):
    aev_computer = torchani.SortedAEV(benchmark=benchmark, device=device)
    prepare = torchani.PrepareInput(aev_computer.species, aev_computer.device)
    model = torchani.models.CustomModel(
        reducer=torch.sum,
        benchmark=benchmark,
        per_species={
            'C': AtomicNetwork(),
            'H': AtomicNetwork(),
            'N': AtomicNetwork(),
            'O': AtomicNetwork(),
        })
    model = torch.nn.Sequential(prepare, aev_computer, model)
    if os.path.isfile(filename):
        model.load_state_dict(torch.load(filename))
    else:
        torch.save(model.state_dict(), filename)
    return model.to(device)
