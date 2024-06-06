import shlex
import typing as tp
from pathlib import Path

import torch
from torch import Tensor

from torchani.constants import ATOMIC_NUMBER
from torchani.utils import pad_atomic_properties


class TorchaniIOError(IOError):
    pass


def read_xyz(
    path: tp.Union[str, Path],
    dtype: torch.dtype = torch.float,
    device: tp.Union[torch.device, tp.Literal["cpu", "cuda"]] = "cpu",
) -> tp.Tuple[Tensor, Tensor, tp.Optional[Tensor]]:
    r"""
    Read an xyz file with possibly many coordinates and species and return a
    (species, coordinates) tuple of tensors. The shapes of the tensors are (C,
    A) and (C, A, 3) respectively, where C is the number of conformations, A
    the maximum number of atoms (conformations with less atoms are padded with
    species=-1 and coordinates=0.0).

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
                    species.append(ATOMIC_NUMBER[s])
                else:
                    species.append(int(s))
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
    return pad_properties["species"], pad_properties["coordinates"], cell
