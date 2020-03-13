import os
import torch
import torchani
from torchani.data._pyanitools import anidataloader
from torchani.units import hartree2kcalmol
import argparse
import math
import tqdm


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('dir', help='Path to the COMP6 directory')
parser.add_argument('-b', '--batchatoms', type=int, default=4096,
                    help='Maximum number of ATOMs in each batch')
parser.add_argument('-d', '--device',
                    help='Device of modules and tensors',
                    default=('cuda' if torch.cuda.is_available() else 'cpu'))
parser = parser.parse_args()

# run benchmark
ani1x = torchani.models.ANI1x().to(parser.device)


def recursive_h5_files(base):
    inside = os.listdir(base)
    for i in inside:
        path = os.path.join(base, i)
        if os.path.isfile(path) and path.endswith(".h5"):
            yield from anidataloader(path)
        elif os.path.isdir(path):
            yield from recursive_h5_files(path)


def by_batch(species, coordinates, model):
    shape = species.shape
    batchsize = max(1, parser.batchatoms // shape[1])
    coordinates = coordinates.clone().detach().requires_grad_(True)
    species = torch.split(species, batchsize)
    coordinates = torch.split(coordinates, batchsize)
    energies = []
    forces = []
    for s, c in zip(species, coordinates):
        e = model((s, c)).energies
        f, = torch.autograd.grad(e.sum(), c)
        energies.append(e)
        forces.append(f)
    return torch.cat(energies).detach(), torch.cat(forces).detach()


class Averager:

    def __init__(self):
        self.count = 0
        self.cumsum = 0

    def update(self, new):
        assert len(new.shape) == 1
        self.count += new.shape[0]
        self.cumsum += new.sum().item()

    def compute(self):
        return self.cumsum / self.count


def relative_energies(energies):
    a, b = torch.combinations(energies, r=2).unbind(1)
    return a - b


def do_benchmark(model):
    dataset = recursive_h5_files(parser.dir)
    mae_averager_energy = Averager()
    mae_averager_relative_energy = Averager()
    mae_averager_force = Averager()
    rmse_averager_energy = Averager()
    rmse_averager_relative_energy = Averager()
    rmse_averager_force = Averager()
    for i in tqdm.tqdm(dataset, position=0, desc="dataset"):
        # read
        coordinates = torch.tensor(i['coordinates'], device=parser.device)
        species = model.species_to_tensor(i['species']) \
                       .unsqueeze(0).expand(coordinates.shape[0], -1)
        energies = torch.tensor(i['energies'], device=parser.device)
        forces = torch.tensor(i['forces'], device=parser.device)
        # compute
        energies2, forces2 = by_batch(species, coordinates, model)
        ediff = energies - energies2
        relative_ediff = relative_energies(energies) - \
            relative_energies(energies2)
        fdiff = forces.flatten() - forces2.flatten()
        # update
        mae_averager_energy.update(ediff.abs())
        mae_averager_relative_energy.update(relative_ediff.abs())
        mae_averager_force.update(fdiff.abs())
        rmse_averager_energy.update(ediff ** 2)
        rmse_averager_relative_energy.update(relative_ediff ** 2)
        rmse_averager_force.update(fdiff ** 2)
    mae_energy = hartree2kcalmol(mae_averager_energy.compute())
    rmse_energy = hartree2kcalmol(math.sqrt(rmse_averager_energy.compute()))
    mae_relative_energy = hartree2kcalmol(mae_averager_relative_energy.compute())
    rmse_relative_energy = hartree2kcalmol(math.sqrt(rmse_averager_relative_energy.compute()))
    mae_force = hartree2kcalmol(mae_averager_force.compute())
    rmse_force = hartree2kcalmol(math.sqrt(rmse_averager_force.compute()))
    print("Energy:", mae_energy, rmse_energy)
    print("Relative Energy:", mae_relative_energy, rmse_relative_energy)
    print("Forces:", mae_force, rmse_force)


do_benchmark(ani1x)
