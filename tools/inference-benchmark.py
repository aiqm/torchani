import argparse
import timeit

import tqdm
import torch

from torchani.models import ANI1x
from torchani.utils import ATOMIC_NUMBERS
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
nnp = ANI1x()[0].to(device)


# load XYZ files
class XYZ:
    def __init__(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()

        # parse lines
        self.mols = []
        atom_count = None
        species = []
        coordinates = []
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
                coordinates.append([x, y, z])
                if atom_count is None:
                    raise RuntimeError("Atom count not present in xyz")
                atom_count -= 1
                if atom_count == 0:
                    state = "ready"
                    _species = torch.tensor(
                        [ATOMIC_NUMBERS[k] for k in species],
                        dtype=torch.long,
                        device=device,
                    )
                    _coordinates = torch.tensor(
                        coordinates, dtype=torch.float, device=device
                    )
                    self.mols.append((_species, _coordinates))
                    coordinates = []
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
energies = nnp((species, coordinates)).energies
mid = timeit.default_timer()
print("Energy time:", mid - start)
_ = forces(energies, coordinates)
print("Force time:", timeit.default_timer() - mid)
print()

# test single mode
print("[Single mode]")
start = timeit.default_timer()
if args.tqdm:
    xyz = tqdm.tqdm(xyz)
for species, coordinates in xyz:
    _, _ = energies_and_forces(
        nnp,
        species.unsqueeze(0),
        coordinates.unsqueeze(0).detach(),
    )
print("Time:", timeit.default_timer() - start)
