import os
import torch
import torchani
from torchani.data._pyanitools import anidataloader
import argparse
import math
import tqdm


HARTREE2KCAL = 627.509

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('dir', help='Path to the COMP6 directory')
parser.add_argument('-b', '--batchsize', type=int, default=2048)
parser.add_argument('-d', '--device',
                    help='Device of modules and tensors',
                    default=('cuda' if torch.cuda.is_available() else 'cpu'))
parser = parser.parse_args()

# run benchmark
ani1x = torchani.models.ANI1x().to(torch.float64).to(parser.device)
ani1ccx = torchani.models.ANI1ccx().to(torch.float64).to(parser.device)


def recursive_h5_files(base):
    inside = os.listdir(base)
    for i in inside:
        path = os.path.join(base, i)
        if os.path.isfile(path) and path.endswith(".h5"):
            yield from anidataloader(path)
        elif os.path.isdir(path):
            yield from recursive_h5_files(path)


def by_batch(it):
    for i in it:
        # read
        coordinates = torch.tensor(
            i['coordinates'], dtype=torch.float64, device=parser.device,
            requires_grad=True)
        species = model.species_to_tensor(i['species']) \
                       .unsqueeze(0).expand(coordinates.shape[0], -1)
        energies = torch.tensor(i['energies'], dtype=torch.float64,
                                device=parser.device)
        forces = torch.tensor(i['forces'], dtype=torch.float64,
                              device=parser.device)
        yield from zip(
            torch.split(coordinates, parser.batchsize),
            torch.split(species, parser.batchsize),
            torch.split(energies, parser.batchsize),
            torch.split(forces, parser.batchsize),
        )


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
    dataset = by_batch(recursive_h5_files(parser.dir))
    mae_averager_energy = Averager()
    mae_averager_relative_energy = Averager()
    mae_averager_force = Averager()
    rmse_averager_energy = Averager()
    rmse_averager_relative_energy = Averager()
    rmse_averager_force = Averager()
    for coordinates, species, energies, forces in tqdm.tqdm(dataset):
        # compute
        _, energies2 = model((species, coordinates))
        forces2, = torch.autograd.grad(energies2.sum(), coordinates)
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
    mae_energy = mae_averager_energy.compute() * HARTREE2KCAL
    rmse_energy = math.sqrt(rmse_averager_energy.compute()) * HARTREE2KCAL
    mae_relative_energy = mae_averager_relative_energy.compute() * HARTREE2KCAL
    rmse_relative_energy = math.sqrt(rmse_averager_relative_energy.compute()) \
        * HARTREE2KCAL
    mae_force = mae_averager_force.compute() * HARTREE2KCAL
    rmse_force = math.sqrt(rmse_averager_force.compute()) * HARTREE2KCAL
    print("Energy:", mae_energy, rmse_energy)
    print("Relative Energy:", mae_relative_energy, rmse_relative_energy)
    print("Forces:", mae_force, rmse_force)


for model in [ani1x, ani1ccx]:
    print(type(model).__name__)
    do_benchmark(model)
    print()
