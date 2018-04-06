import torchani
import pyanitools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

def step(batch_squared_error, batch_size, training, epoch):
    mse = batch_squared_error  / batch_size
    rmse_kcalmol = 627.509 * torch.sqrt(mse)
    print('rmse:', rmse_kcalmol.item(), 'kcal/mol')
    if training:
        loss = 0.5 * torch.exp(2 * mse) if epoch > 10 else mse
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def visit_file(filename, training, epoch):
    print('file:', filename)
    adl = pyanitools.anidataloader(filename)
    for data in adl:
        coordinates = torch.from_numpy(data['coordinates']).cuda()
        conformations = coordinates.shape[0]
        species = data['species']
        label = torch.from_numpy(ani.shift_energy(data['energies'], species)).float().cuda()
        radial_aev, angular_aev = ani.compute_aev(coordinates, species)
        aev = torch.cat([radial_aev, angular_aev], dim=2)

        pred = net(aev, species)
        squared_error = torch.sum((label - pred) ** 2)

        step(squared_error, conformations, training, epoch)

for epoch in range(10000):
    print('epoch:', epoch)
    print('training')
    for filename in ['data/ani_gdb_s0{}.h5'.format(i) for i in range(1, 7)]:
        visit_file(filename, True, epoch)
    print('testing')
    for filename in ['data/ani_gdb_s0{}.h5'.format(i) for i in range(7, 8)]:
        visit_file(filename, False, epoch)
        
            

       
        
