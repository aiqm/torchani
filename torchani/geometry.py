r""" Utilities to generate some specific geometries"""
import typing as tp

import torch
from torch import Tensor

from torchani.annotations import Device
from torchani.utils import AtomicNumbersToMasses

Reference = tp.Literal["center_of_mass", "center_of_geometry", "origin"]


class Displacer(torch.nn.Module):
    r"""
    Displace coordinates to the center-of-mass (center_of_mass=True) frame, or
    center_of_geometry frame (center_of_mass=False) input species must be
    atomic numbers, padding atoms can be included with -1 as padding. Returns
    the centered coordinates
    """

    def __init__(
        self,
        masses: tp.Iterable[float] = (),
        reference: Reference = "center_of_mass",
        device: Device = "cpu",
        dtype: torch.dtype = torch.float,
    ) -> None:
        super().__init__()
        self._atomic_masses: Tensor = torch.tensor(masses, device=device, dtype=dtype)
        self._center_of_mass = (reference == "center_of_mass")
        self._skip = (reference == "origin")
        self._converter = AtomicNumbersToMasses(masses, dtype=dtype, device=device)

    def forward(self, species: Tensor, coordinates: Tensor) -> Tensor:
        # Do nothing if reference is origin
        if self._skip:
            return coordinates

        mask = species == -1
        if self._center_of_mass:
            masses = self._converter(species)
            mass_sum = masses.unsqueeze(-1).sum(dim=1, keepdim=True)
            weights = masses.unsqueeze(-1) / mass_sum
        else:
            is_not_dummy = ~mask
            not_dummy_sum = is_not_dummy.unsqueeze(-1).sum(dim=1, keepdim=True)
            weights = is_not_dummy.unsqueeze(-1) / not_dummy_sum
        com_coordinates = coordinates * weights
        com_coordinates = com_coordinates.sum(dim=1, keepdim=True)
        centered_coordinates = coordinates - com_coordinates
        centered_coordinates[mask, :] = 0.0
        return centered_coordinates
