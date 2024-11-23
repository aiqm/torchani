r"""Convenience functions for reading and writing molecules in ".xyz" format"""

import shlex
import typing as tp
from pathlib import Path

import torch
from torch import Tensor

from torchani.annotations import StrPath, Device, DType
from torchani.constants import ATOMIC_NUMBER, PERIODIC_TABLE
from torchani.utils import pad_atomic_properties


class TorchaniIOError(IOError):
    pass


__all__ = ["write_xyz", "read_xyz"]


def write_xyz(
    species: Tensor,
    coordinates: Tensor,
    dest: StrPath,
    cell: tp.Optional[Tensor] = None,
    pad: bool = False,
    pad_coord_value: float = 0.0,
    pad_species_value: int = 100,
) -> None:
    r"""Write an xyz file with possibly many coordinates and species.

    The shapes of the input tensors should be ``(C, A)`` and ``(C, A, 3)`` respectively,
    where C is the number of conformations, and A the maximum number of atoms. Output is
    compatible with ASE's 'extxyz'. If a cell is passed it is written in the comment
    line of all conformations.

    If ``pad=True``, output species are padded with atomic number pad_species_value (100
    by default) and with coordinates pad_coord_value (0.0 by default), otherwise atoms
    with species number -1 are not written to the file. Note that many programs (e.g.
    vmd) require that all conformations in multi-conformation ``*.xyz`` files have the
    same number of atoms, so padding may be useful for visualization in some cases.
    """
    dest = Path(dest).resolve()
    # Input validation
    if species.dim() != 2:
        raise ValueError("Species should be a 2 dim tensor")
    if coordinates.shape != (species.shape[0], species.shape[1], 3):
        raise ValueError("Coordinates should have shape (molecules, atoms, 3)")
    if cell is not None and cell.shape != (3, 3):
        raise ValueError("Cell should be a tensor of shape (3, 3)")

    with open(dest, mode="wt", encoding="utf-8") as f:
        for j, (znums, coords) in enumerate(zip(species, coordinates)):
            if not pad:
                mask = znums != -1
                coords = coords[mask]
                znums = znums[mask]
            else:
                if (znums == pad_species_value).any():
                    raise ValueError(
                        "Can't pad if there are elements with atomic number 100"
                    )
                mask = znums == -1
                znums[mask] = pad_species_value
                coords[mask] = pad_coord_value
            f.write(f"{len(coords)}\n")
            props = "species:S:1:pos:R:3"
            if cell is not None:
                cell_elements = " ".join(
                    [(f"{e:.10f}" if e != 0.0 else "0.0") for e in cell.view(-1)]
                )
                f.write(f'Lattice="{cell_elements}" Properties={props} pbc="T T T"\n')
            else:
                f.write(f'Properties={props} pbc="F F F"\n')
            for z, atom in zip(znums, coords):
                symbol = PERIODIC_TABLE[z]
                f.write(f"{symbol} {atom[0]:.10f} {atom[1]:.10f} {atom[2]:.10f}\n")


def read_xyz(
    path: StrPath,
    dtype: DType = None,
    device: Device = None,
    detect_padding: bool = True,
    pad_species_value: int = 100,
) -> tp.Tuple[Tensor, Tensor, tp.Optional[Tensor], tp.Optional[Tensor]]:
    r"""Read an xyz file with possibly many molecules. Return them in tensor form

    Returns a (species, coordinates) tuple of tensors. The shapes of the tensors are (C,
    A) and (C, A, 3) respectively, where C is the number of conformations, A the maximum
    number of atoms (conformations with less atoms are padded with species=-1 and
    coords = 0.0).

    If detect_padding is True, species with atomic number pad_species_value (100 by
    default) are considered "padding atoms" and are changed to -1. All their coordinates
    are set to 0.0, no matter their value.

    Cell is read from the first conformation.
    """
    path = Path(path).resolve()
    cell: tp.Optional[Tensor] = None
    properties: tp.List[tp.Dict[str, Tensor]] = []
    with open(path, mode="rt", encoding="utf-8") as f:
        lines = iter(f)
        conformation_num = 0
        while True:
            species = []
            coordinates = []
            try:
                num = int(next(lines))
            except StopIteration:
                break
            comment = next(lines)
            if "lattice" in comment.lower():
                if (cell is None) and (conformation_num != 0):
                    raise TorchaniIOError(
                        "If cell is present it should be in the first conformation"
                    )
                parts = shlex.split(comment)
                for part in parts:
                    key, value = part.split("=")
                    if key.lower() == "lattice":
                        # cell order is x0 y0 z0 x1 y1 z1 x2 y2 z2 for the
                        # 3 unit vectors
                        conformation_cell = torch.tensor(
                            [float(s) for s in value.split()],
                            dtype=dtype,
                            device=device,
                        ).view(3, 3)
                        if cell is None:
                            cell = conformation_cell
                        elif not (cell == conformation_cell).all():
                            raise TorchaniIOError(
                                "Found two conformations with non-matching cells"
                            )
            for _ in range(num):
                line = next(lines)
                s, x, y, z = line.split()
                if s in ATOMIC_NUMBER:
                    atomic_num = ATOMIC_NUMBER[s]
                else:
                    atomic_num = int(s)
                if atomic_num == pad_species_value and detect_padding:
                    atomic_num = -1
                    x, y, z = "0.0", "0.0", "0.0"
                species.append(atomic_num)
                coordinates.append([float(x), float(y), float(z)])
            conformation_num += 1
            properties.append(
                {
                    "coordinates": torch.tensor(
                        [coordinates],
                        dtype=dtype,
                        device=device,
                    ),
                    "species": torch.tensor(
                        [species],
                        dtype=torch.long,
                        device=device,
                    ),
                }
            )
    pad_properties = pad_atomic_properties(properties)
    pbc = torch.tensor([True, True, True], device=device) if cell is not None else None
    return pad_properties["species"], pad_properties["coordinates"], cell, pbc
