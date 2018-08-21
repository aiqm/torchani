import torch
import torchani
import os


def atomic():
    model = torch.nn.Sequential(
        torch.nn.Linear(384, 128),
        torch.nn.CELU(0.1),
        torch.nn.Linear(128, 128),
        torch.nn.CELU(0.1),
        torch.nn.Linear(128, 64),
        torch.nn.CELU(0.1),
        torch.nn.Linear(64, 1)
    )
    return model


def get_or_create_model(filename, device=torch.device('cpu')):
    consts = torchani.neurochem.Constants()
    sae = torchani.neurochem.load_sae()
    aev_computer = torchani.AEVComputer(**consts)
    model = torchani.ANIModel([
        ('C', atomic()),
        ('H', atomic()),
        ('N', atomic()),
        ('O', atomic()),
    ])

    class Flatten(torch.nn.Module):

        def forward(self, x):
            return x[0], x[1].flatten()

    model = torch.nn.Sequential(aev_computer, model, Flatten())
    if os.path.isfile(filename):
        model.load_state_dict(torch.load(filename))
    else:
        torch.save(model.state_dict(), filename)
    return model.to(device), torchani.EnergyShifter(consts.species, sae)
