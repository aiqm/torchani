import torchani
import pyanitools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


ani = torchani.ani('data/rHCNO-4.6R_16-3.1A_a4-8_3.params', 'data/sae_linfit.dat')

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.by_species = {}
        for i in ani.species:
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
            y = x[:,i,:]
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
        return torch.squeeze(molecule_energies)


net = Net()
print(net)

optimizer = optim.Adam(net.parameters(), amsgrad=True)
conformations_per_batch = -1

def step(batch_squared_error, batch_size, training):
    # start = time.time()
    mse = batch_squared_error  / batch_size
    rmse_kcalmol = 627.509 * torch.sqrt(mse)
    print('rmse:', rmse_kcalmol.item(), 'kcal/mol')
    if training:
        loss = mse # 0.5 * torch.exp(2 * mse)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # end = time.time()
    # print('time backwards:', end - start)

def visit_file(filename, training):
    print('file:', filename)
    adl = pyanitools.anidataloader(filename)
    for data in adl:
        # start = time.time()
        coordinates = torch.from_numpy(data['coordinates']).cuda()
        conformations = coordinates.shape[0]
        species = data['species']
        label = torch.from_numpy(ani.shift_energy(data['energies'], species)).float().cuda()
        # end = time.time()
        # print('time loading data:', end - start)

        # start = time.time()
        radial_aev, angular_aev = ani.compute_aev(coordinates, species)
        aev = torch.cat([radial_aev, angular_aev], dim=2)
        end = time.time()
        # print('time computing aev:', end - start)

        # start = time.time()
        pred = net(aev, species)
        squared_error = torch.sum((label - pred) ** 2)
        end = time.time()
        # print('time forward:', end - start)

        step(squared_error, conformations, training)


mint = 9999999999
for i in range(10):
    start = time.time()
    visit_file('data/ani_gdb_s08.h5', True)
    end = time.time()
    t = end - start
    if t < mint:
        mint = t
print('minimum file time:', mint)