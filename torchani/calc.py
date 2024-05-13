import typing as tp

import torch
from torch import Tensor

from torchani.geometry import Displacer, Reference
from torchani.constants import ATOMIC_MASSES

__all__ = ["DipoleComputer", "compute_dipole"]


class DipoleComputer(torch.nn.Module):
    """
    Compute dipoles in eA

    Arguments:
        species (torch.Tensor): (M, N), species must be atomic numbers.
        coordinates (torch.Tensor): (M, N, 3), unit should be Angstrom.
        charges (torch.Tensor): (M, N), unit should be e.
        center_of_mass (Bool): When calculating dipole for charged molecule,
            it is necessary to displace the coordinates to the center-of-mass frame.
    Returns:
        dipoles (torch.Tensor): (M, 3)
    """

    def __init__(
        self,
        masses: tp.Iterable[float] = ATOMIC_MASSES,
        reference: Reference = "center_of_mass",
        device: tp.Union[torch.device, tp.Literal["cpu"], tp.Literal["cuda"]] = "cpu",
        dtype: torch.dtype = torch.float,
    ) -> None:
        super().__init__()
        self._displacer = Displacer(
            masses,
            reference,
            device,
            dtype,
        )

    def forward(self, species: Tensor, coordinates: Tensor, charges: Tensor) -> Tensor:
        assert species.shape == charges.shape == coordinates.shape[:-1]
        charges = charges.unsqueeze(-1)
        coordinates = self._displacer(species, coordinates)
        dipole = torch.sum(charges * coordinates, dim=1)
        return dipole


# Convenience fn around DipoleComputer that is non-jittable
def compute_dipole(
    species: Tensor,
    coordinates: Tensor,
    charges: Tensor,
    reference: Reference = "center_of_mass",
) -> Tensor:
    if torch.jit.is_scripting():
        raise RuntimeError(
            "'torchani.calc.compute_dipole' doesn't support JIT, "
            " consider using torchani.calc.DipoleComputer instead"
        )
    return DipoleComputer(
        reference=reference,
        device=species.device,
        dtype=coordinates.dtype,
    )(species, coordinates, charges)
