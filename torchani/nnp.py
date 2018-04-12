from .aev_base import AEVComputer
import torch
import torch.nn as nn

class NeuralNetworkOnAEV(nn.Module):

    def __init__(self, aev_computer, sizes, activation):
        super(NeuralNetworkOnAEV, self).__init__()
        if not isinstance(aev_computer, AEVComputer):
            raise TypeError("NeuralNetworkPotential: aev_computer must be a subclass of AEVComputer")
        self.aev_computer = aev_computer
        self.aev_size = aev_computer.radial_length() + aev_computer.angular_length()
        self.layers = len(sizes)
        self.activation = activation
        sizes = [self.aev_size] + sizes
        for i in aev_computer.species:
            for j in range(self.layers):
                linear = nn.Linear(sizes[j], sizes[j+1]).type(aev_computer.dtype)
                setattr(self, '{}{}'.format(i,j), linear)

    def forward(self, coordinates, species):
        radial_aev, angular_aev = self.aev_computer(coordinates, species)
        fullaev = torch.cat([radial_aev, angular_aev], dim=2)
        atoms = len(species)
        per_atom_outputs = []
        for i in range(atoms):
            s = species[i]
            y = fullaev[:,i,:]
            for j in range(self.layers):
                linear = getattr(self, '{}{}'.format(s,j))
                y = linear(y)
                y = self.activation(y)
            per_atom_outputs.append(y)
        per_atom_outputs = torch.stack(per_atom_outputs)
        molecule_output = torch.sum(per_atom_outputs, dim=0)
        return torch.squeeze(molecule_output)

    def reset_parameters(self):
        for s in self.aev_computer.species:
            for j in range(self.layers):
                getattr(self, '{}{}'.format(s,j)).reset_parameters()