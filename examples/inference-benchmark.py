import argparse
import torchani
import torch
import os
import timeit


path = os.path.dirname(os.path.realpath(__file__))

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('filename',
                    help='Path to the xyz file.')
parser.add_argument('-d', '--device',
                    help='Device of modules and tensors',
                    default=('cuda' if torch.cuda.is_available() else 'cpu'))
parser = parser.parse_args()

# set up benchmark
device = torch.device(parser.device)
buildins = torchani.neurochem.Buildins()
nnp = torch.nn.Sequential(
    buildins.aev_computer,
    buildins.models[0],
    buildins.energy_shifter
).to(device)


# load XYZ files
class XYZ:

    def __init__(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        # parse lines
        self.mols = []
        atom_count = None
        species = []
        coordinates = []
        state = 'ready'
        for i in lines:
            i = i.strip()
            if state == 'ready':
                atom_count = int(i)
                state = 'comment'
            elif state == 'comment':
                state = 'atoms'
            else:
                s, x, y, z = i.split()
                x, y, z = float(x), float(y), float(z)
                species.append(s)
                coordinates.append([x, y, z])
                atom_count -= 1
                if atom_count == 0:
                    state = 'ready'
                    species = buildins.consts.species_to_tensor(species) \
                                      .to(device)
                    coordinates = torch.tensor(coordinates, device=device)
                    self.mols.append((species, coordinates))
                    coordinates = []
                    species = []

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, i):
        return self.mols[i]


xyz = XYZ(parser.filename)

print(len(xyz), 'conformations')
print()

# test batch mode
print('[Batch mode]')
species, coordinates = torch.utils.data.dataloader.default_collate(list(xyz))
coordinates = torch.tensor(coordinates, requires_grad=True)
start = timeit.default_timer()
energies = nnp((species, coordinates))[1]
mid = timeit.default_timer()
print('Energy time:', mid - start)
force = -torch.autograd.grad(energies.sum(), coordinates)[0]
print('Force time:', timeit.default_timer() - mid)
print()

# test single mode
print('[Single mode]')
start = timeit.default_timer()
for species, coordinates in xyz:
    species = species.unsqueeze(0)
    coordinates = torch.tensor(coordinates.unsqueeze(0), requires_grad=True)
    energies = nnp((species, coordinates))[1]
    force = -torch.autograd.grad(energies.sum(), coordinates)[0]
print('Time:', timeit.default_timer() - start)
