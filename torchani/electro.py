r"""Utilities for working with systems that have explicit electrostatic interactions"""
import typing as tp
from collections import OrderedDict

import torch
from torch import Tensor
import typing_extensions as tpx

from torchani.geometry import Displacer, Reference
from torchani.atomics import AtomicContainer
from torchani.constants import ELECTRONEGATIVITY, HARDNESS, ATOMIC_NUMBER

__all__ = ["DipoleComputer", "compute_dipole", "ChargeNormalizer"]


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
        device: tp.Union[torch.device, tp.Literal["cpu", "cuda"]] = "cpu",
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
class _AdaptedChargesContainer(AtomicContainer):

    # Needed for bw compatibility
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        old_keys = list(state_dict.keys())
        for k in old_keys:
            suffix = k.split(prefix)[-1] if prefix else k
            if not suffix.startswith("atomics."):
                state_dict["".join((prefix, "atomics.", suffix))] = state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @staticmethod
    def ensureOrderedDict(modules):
        if isinstance(modules, OrderedDict):
            return modules
        od = OrderedDict()
        for i, m in enumerate(modules):
            od[str(i)] = m
        return od

    def __init__(self, modules):
        super().__init__()
        self.atomics = torch.nn.ModuleDict(self.ensureOrderedDict(modules))
        self.num_species = len(self.atomics)
        self.num_networks = 1

    def forward(
        self,
        species_aevs: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> tp.Tuple[Tensor, Tensor]:
        element_idxs, aevs = species_aevs
        element_idxs_ = element_idxs.flatten()
        aevs = aevs.flatten(0, 1)
        output = aevs.new_zeros(element_idxs_.shape)
        for i, module in enumerate(self.atomics.values()):
            selected_idx = (element_idxs_ == i).nonzero().view(-1)
            if selected_idx.shape[0] > 0:
                input_ = aevs.index_select(0, selected_idx)
                output.index_add_(0, selected_idx, module(input_)[:, 1].view(-1))
        atomic_charges = output.view_as(element_idxs)
        return element_idxs, atomic_charges
