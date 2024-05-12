import typing as tp
from pathlib import Path
import math

import torch
from torch import Tensor

from torchani.utils import PERIODIC_TABLE


def make_methane(device=None, eq_bond=1.09):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = eq_bond * 2 / math.sqrt(3)
    coordinates = (
        torch.tensor(
            [[[0.0, 0.0, 0.0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5]]],
            device=device,
            dtype=torch.double,
        )
        * d
    )
    species = torch.tensor([[1, 1, 1, 1, 6]], device=device, dtype=torch.long)
    return species, coordinates.double()


def make_water(device=None, eq_bond=0.957582, eq_angle=104.485):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = eq_bond
    t = (math.pi / 180) * eq_angle  # convert to radians
    coordinates = torch.tensor(
        [[d, 0, 0], [d * math.cos(t), d * math.sin(t), 0], [0, 0, 0]], device=device
    ).double()
    species = torch.tensor([[1, 1, 8]], device=device, dtype=torch.long)
    return species, coordinates.double()


def tensor_from_xyz(path):
    with open(path, "r") as f:
        lines = f.readlines()
        num_atoms = int(lines[0])
        coordinates = []
        species = []
        _, _, a, b, c = lines[1].split()
        cell = torch.diag(torch.tensor([float(a), float(b), float(c)]))
        for line in lines[2:]:
            values = line.split()
            if values:
                s = values[0].strip()
                x = float(values[1])
                y = float(values[2])
                z = float(values[3])
                coordinates.append([x, y, z])
                species.append(PERIODIC_TABLE.index(s))
        coordinates = torch.tensor(coordinates)
        species = torch.tensor(species, dtype=torch.long)
        assert coordinates.shape[0] == num_atoms
        assert species.shape[0] == num_atoms
    return species, coordinates, cell


def tensor_to_xyz(
    path,
    species_coordinates: tp.Tuple[Tensor, Tensor],
    no_exponent: bool = True,
):
    path = Path(path).resolve()
    # input species must be atomic numbers
    species, coordinates = species_coordinates
    num_atoms = species.shape[1]

    assert coordinates.dim() == 3, "bad number of dimensions for coordinates"
    assert species.dim() == 2, "bad number of dimensions for species"
    assert coordinates.shape[0] == 1, "Batch printing not implemented"
    assert species.shape[0] == 1, "Batch printing not implemented"

    coordinates = coordinates.view(-1, 3)
    species = species.view(-1)

    with open(path, "w") as f:
        f.write(f"{num_atoms}\n")
        f.write("\n")
        for s, c in zip(species, coordinates):
            if no_exponent:
                line = f"{c[0]:.15f} {c[1]:.15f} {c[2]:.15f}\n"
            else:
                line = f"{c[0]} {c[1]} {c[2]}\n"
            line = f"{PERIODIC_TABLE[s]} " + line
            f.write(line)
