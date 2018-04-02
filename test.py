from ani import torchani
import pyanitools
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.by_species = {}
        for i in torchani.species:
            linear = nn.Linear(384, 128).cuda()
            linear.reset_parameters()
            setattr(self, i + '0', linear)
            linear = nn.Linear(128, 128).cuda()
            linear.reset_parameters()
            setattr(self, i + '1', linear)
            linear = nn.Linear(128, 64).cuda()
            linear.reset_parameters()
            setattr(self, i + '2', linear)
            linear = nn.Linear(64, 1).cuda()
            linear.reset_parameters()
            setattr(self, i + '3', linear)

    def forward(self, x, species):
        atoms = len(species)
        per_atom_energies = []
        for i in range(atoms):
            s = species[i]
            y = x[i]
            y = getattr(self, s + '0')(y)
            y = torch.exp(-y**2)
            y = getattr(self, s + '1')(y)
            y = torch.exp(-y**2)
            y = getattr(self, s + '2')(y)
            y = torch.exp(-y**2)
            y = getattr(self, s + '3')(y)
            per_atom_energies.append(y)
        per_atom_energies = torch.stack(per_atom_energies)
        molecule_energies = torch.sum(per_atom_energies, dim=0)
        return molecule_energies


net = Net()
print(net)

optimizer = optim.RMSprop(net.parameters())

def print_parameters(net):
    for s in torchani.species:
        for i in ['0', '1', '2', '3']:
            print('parameter sum:', torch.sum(getattr(net, s + i).weight.data[0]))

adl = pyanitools.anidataloader("ani_gdb_s02.h5")
for epoch in range(100000):
    print('epoch', epoch)
    for data in adl:
        coordinates = torch.from_numpy(data['coordinates']).cuda()
        conformations = coordinates.shape[0]
        species = data['species']
        smiles = ''.join(data['smiles'])
        label = Variable(torch.from_numpy(torchani.shift_energy(data['energies'], species)).float().cuda(), requires_grad=False)
        aev = Variable(torchani.compute_aev(coordinates, species), requires_grad=False)

        # print_parameters(net)
        optimizer.zero_grad()
        pred = net(aev, species)
        mse = torch.sum((label - pred) ** 2)
        print('rmse on {} conformations: {} kcal/mol'.format(conformations, 627.509 * torch.sqrt(mse / conformations).data[0] ))
        loss = mse / conformations # 0.5 * torch.exp(2 * mse)
        loss.backward()
        optimizer.step()
