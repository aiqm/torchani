r"""Utilities for working with systems that have explicit electrostatic interactions"""
import typing as tp

import torch
from torch import Tensor
import typing_extensions as tpx

from torchani.annotations import Device
from torchani.utils import AtomicNumbersToMasses
from torchani.atomics import AtomicContainer
from torchani.constants import ELECTRONEGATIVITY, HARDNESS, ATOMIC_NUMBER
# Needed for _AdaptedChargesContainer hack
from torchani.nn import ANIModel
from torchani.tuples import SpeciesEnergies

__all__ = ["DipoleComputer", "compute_dipole", "ChargeNormalizer"]


Reference = tp.Literal["center_of_mass", "center_of_geometry", "origin"]


class ChargeNormalizer(torch.nn.Module):
    r"""
    Usage:

    .. code-block::python

        normalizer = ChargeNormalizer()
        total_charge = 0.0
        element_idxs = torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.long)
        raw_charges = torch.tensor([[0.3, 0.5, -0.5]], dtype=torch.float)
        norm_charges = normalizer(element_idxs, raw_charges, total_charge)
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

    def factor(self, element_idxs: Tensor, raw_charges: Tensor) -> Tensor:
        weights = self.weights[element_idxs]
        weights = weights.masked_fill(element_idxs == -1, 0.0)
        if self.scale_weights_by_charges_squared:
            weights = weights * raw_charges**2
        return weights / torch.sum(weights, dim=-1, keepdim=True)

    def forward(
        self,
        element_idxs: Tensor,
        raw_charges: Tensor,
        total_charge: float = 0.0,
    ) -> Tensor:
        total_raw_charge = torch.sum(raw_charges, dim=-1, keepdim=True)
        charge_excess = total_charge - total_raw_charge
        factor = self.factor(element_idxs, raw_charges)
        return raw_charges + charge_excess * factor


class DipoleComputer(torch.nn.Module):
    """
    Compute dipoles in eA

    Arguments:
        species (torch.Tensor): (M, N), species must be atomic numbers.
            padding atoms with pad = -1 can be included.
        coordinates (torch.Tensor): (M, N, 3), unit should be Angstrom.
        charges (torch.Tensor): (M, N), unit should be e.
        center_of_mass (Bool): When calculating dipole for charged molecule,
            it is necessary to displace the coordinates to the center-of-mass frame.
    Returns:
        dipoles (torch.Tensor): (M, 3)
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

    def forward(self, species: Tensor, coordinates: Tensor, charges: Tensor) -> Tensor:
        assert species.shape == charges.shape == coordinates.shape[:-1]
        charges = charges.unsqueeze(-1)
        coordinates = self._displace_to_reference(species, coordinates)
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
            "'torchani.electro.compute_dipole' doesn't support JIT, "
            " consider using torchani.electro.DipoleComputer instead"
        )
    return DipoleComputer(
        reference=reference,
        device=species.device,
        dtype=coordinates.dtype,
    )(species, coordinates, charges)


# Hack: Grab a network with "bad energies", discard them and only outputs the
# charges
class _AdaptedChargesContainer(ANIModel):
    @torch.jit.export
    def _atomic_energies(
        self,
        species_aev: tp.Tuple[Tensor, Tensor],
    ) -> Tensor:
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]

        species_ = species.flatten()
        aev = aev.flatten(0, 1)
        output = aev.new_zeros(species_.shape)
        for i, m in enumerate(self.atomics.values()):
            midx = (species_ == i).nonzero().view(-1)
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output.index_add_(0, midx, m(input_)[:, 1].view(-1))
        output = output.view_as(species)
        return output.unsqueeze(0)

    def forward(
        self,
        species_aev: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> SpeciesEnergies:
        atomic_energies = self._atomic_energies(species_aev).squeeze(0)
        return SpeciesEnergies(species_aev[0], atomic_energies)

    def to_infer_model(self, use_mnp: bool = False) -> AtomicContainer:
        return self
