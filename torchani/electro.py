r"""Utilities for working with charged systems

Useful for systems that have charges and coulomb-like electrostatic interactions.
"""

import typing as tp

import torch
from torch import Tensor
import typing_extensions as tpx

from torchani.annotations import Device, DType
from torchani.utils import AtomicNumbersToMasses
from torchani.constants import ELECTRONEGATIVITY, HARDNESS, ATOMIC_NUMBER

__all__ = ["DipoleComputer", "compute_dipole", "ChargeNormalizer"]


Reference = tp.Literal["center_of_mass", "center_of_geometry", "origin"]


class ChargeNormalizer(torch.nn.Module):
    r"""
    Usage:

    .. code-block::python

        normalizer = ChargeNormalizer()
        charge = 0
        elem_idxs = torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.long)
        raw_charges = torch.tensor([[0.3, 0.5, -0.5]], dtype=torch.float)
        norm_charges = normalizer(elem_idxs, raw_charges, charge)
        # norm_charges will sum to zero
    """

    atomic_numbers: Tensor
    weights: Tensor

    def __init__(
        self,
        symbols: tp.Sequence[str],
        weights: tp.Sequence[float] = (),
        scale_weights_by_charges_squared: bool = False,
    ):
        super().__init__()
        self.atomic_numbers = torch.tensor(
            [ATOMIC_NUMBER[e] for e in symbols], dtype=torch.long
        )
        if not weights:
            weights = [1.0] * len(symbols)

        self.register_buffer(
            "weights",
            torch.tensor(weights, dtype=torch.float, device=torch.device("cpu")),
            persistent=False,
        )
        self.scale_weights_by_charges_squared = scale_weights_by_charges_squared

    @classmethod
    def from_electronegativity_and_hardness(
        cls,
        symbols: tp.Sequence[str],
        electronegativity: tp.Sequence[float] = (),
        hardness: tp.Sequence[float] = (),
        scale_weights_by_charges_squared: bool = False,
    ) -> tpx.Self:
        atomic_numbers = [ATOMIC_NUMBER[e] for e in symbols]
        # Get constant values from literature if not provided
        if not electronegativity:
            electronegativity = [ELECTRONEGATIVITY[j] for j in atomic_numbers]
        if not hardness:
            hardness = [HARDNESS[j] for j in atomic_numbers]
        weights = [(e / h) ** 2 for e, h in zip(electronegativity, hardness)]
        return cls(symbols, weights, scale_weights_by_charges_squared)

    def factor(self, elem_idxs: Tensor, raw_charges: Tensor) -> Tensor:
        weights = self.weights[elem_idxs]
        weights = weights.masked_fill(elem_idxs == -1, 0.0)
        if self.scale_weights_by_charges_squared:
            weights = weights * raw_charges**2
        return weights / torch.sum(weights, dim=-1, keepdim=True)

    def forward(
        self, elem_idxs: Tensor, raw_charges: Tensor, charge: int = 0
    ) -> Tensor:
        r"""Normalize charges so that they add up to a total charge"""
        excess = charge - raw_charges.sum(dim=-1, keepdim=True)
        return raw_charges + excess * self.factor(elem_idxs, raw_charges)


class DipoleComputer(torch.nn.Module):
    r"""
    Compute dipoles in eA

    Args:
        masses: Sequence of atomic masses
        reference: Reference frame
            to use when calculating dipole.
    """

    def __init__(
        self,
        masses: tp.Iterable[float] = (),
        reference: Reference = "center_of_mass",
        device: Device = None,
        dtype: DType = None,
    ) -> None:
        super().__init__()

        self._atomic_masses: Tensor = torch.tensor(masses, device=device, dtype=dtype)
        self._center_of_mass = reference == "center_of_mass"
        self._skip = reference == "origin"
        self._converter = AtomicNumbersToMasses(masses, dtype=dtype, device=device)

    def forward(
        self, atomic_nums: Tensor, coordinates: Tensor, charges: Tensor
    ) -> Tensor:
        r"""
        Calculate the dipoles

        Args:
            atomic_nums: |atomic_nums|
            coordinates: |coords|
            charges: Float tensor of atomic charges ``(M, N)``. Unit should be e.

        Returns:
            Dipoles with shape ``(molecules, 3)``
        """
        assert atomic_nums.shape == charges.shape == coordinates.shape[:-1]
        charges = charges.unsqueeze(-1)
        coordinates = self._displace_to_reference(atomic_nums, coordinates)
        return torch.sum(charges * coordinates, dim=1)

    def _displace_to_reference(self, species: Tensor, coordinates: Tensor) -> Tensor:
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


def compute_dipole(
    species: Tensor,
    coordinates: Tensor,
    charges: Tensor,
    reference: Reference = "center_of_mass",
) -> Tensor:
    r"""
    Compute dipoles in eA

    Convenience wrapper over `DipoleComputer`. Non-jittable.
    """
    if torch.jit.is_scripting():
        raise RuntimeError(
            "'torchani.electro.compute_dipole' doesn't support JIT, "
            " consider using torchani.electro.DipoleComputer instead"
        )
    return DipoleComputer(
        reference=reference,
        device=species.device,
        dtype=coordinates.dtype,
    )(species, coordinates, charges)
