import argparse
import timeit

import tqdm
import torch

from torchani.models import ANI1x
from torchani.grad import energies_and_forces, forces

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("filename", help="Path to the xyz file.")
parser.add_argument(
    "-d",
    "--device",
    help="Device of modules and tensors",
    default=("cuda" if torch.cuda.is_available() else "cpu"),
)
parser.add_argument(
    "--tqdm",
    dest="tqdm",
    action="store_true",
    help="Whether to use tqdm to display progress",
)
args = parser.parse_args()

# set up benchmark
device = torch.device(args.device)
ani1x = ANI1x(model_index=0, periodic_table_index=False).to(device)


# load XYZ files
class XYZ:
    def __init__(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()

        # parse lines
        self.mols = []
        atom_count = None
        species = []
        _coordinates = []
        state = "ready"
        for i in lines:
            i = i.strip()
            if state == "ready":
                atom_count = int(i)
                state = "comment"
            elif state == "comment":
                state = "atoms"
            else:
                s, _x, _y, _z = i.split()
                x, y, z = float(_x), float(_y), float(_z)
                species.append(s)
                _coordinates.append([x, y, z])
                if atom_count is None:
                    raise RuntimeError("Atom count not present")
                atom_count -= 1
                if atom_count == 0:
                    state = "ready"
                    species = ani1x.species_to_tensor(species).to(device)
                    coordinates = torch.tensor(_coordinates, device=device)
                    self.mols.append((species, coordinates))
                    _coordinates = []
                    species = []

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, i):
        return self.mols[i]

    def __iter__(self):
        return iter(self.mols)


xyz = XYZ(args.filename)

print(len(xyz), "conformations")
print()

# test batch mode
print("[Batch mode]")
species, coordinates = torch.utils.data.dataloader.default_collate(list(xyz))
coordinates.requires_grad_(True)
start = timeit.default_timer()
energies = ani1x((species, coordinates)).energies
mid = timeit.default_timer()
print("Energy time:", mid - start)
force = forces(energies, coordinates)
print("Force time:", timeit.default_timer() - mid)
print()

# test single mode
print("[Single mode]")
start = timeit.default_timer()
if args.tqdm:
    xyz = tqdm.tqdm(xyz)
for species, coordinates in xyz:
    species = species.unsqueeze(0)
    coordinates = coordinates.unsqueeze(0)
    _, _ = energies_and_forces(ani1x, species, coordinates.detach())
print("Time:", timeit.default_timer() - start)
