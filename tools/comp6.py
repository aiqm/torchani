import os
import torch
import torchani
from torchani.data._pyanitools import anidataloader
import argparse


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('dir', help='Path to the COMP6 directory')
parser.add_argument('-d', '--device',
                    help='Device of modules and tensors',
                    default=('cuda' if torch.cuda.is_available() else 'cpu'))
parser = parser.parse_args()

# run benchmark
ani1x = torchani.models.ANI1x().to(torch.float64)
ani1ccx = torchani.models.ANI1ccx().to(torch.float64)


def recursive_h5_files(base):
    inside = os.listdir(base)
    for i in inside:
        path = os.path.join(base, i)
        if os.path.isfile(path) and path.endswith(".h5"):
            yield from anidataloader(path)
        elif os.path.isdir(path):
            yield from recursive_h5_files(path)


def do_benchmark(model):
    dataset = recursive_h5_files(parser.dir)
    for i in dataset:
        coordinates = torch.tensor(i['coordinates'],
            dtype=torch.float64, requires_grad=True)
        energies = torch.tensor(i['energies'], dtype=torch.float64)
        species = model.species_to_tensor(i['species'])
        forces = torch.tensor(i['forces'], dtype=torch.float64)


for model in [ani1x, ani1ccx]:
    print(type(model).__name__)
    do_benchmark(model)
    print()
