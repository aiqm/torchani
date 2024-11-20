r"""Potentials that calculate dispersion (i.e. van der Waals) interactions"""

import math
import typing as tp

import torch
import h5py
from torch import Tensor
from torch.jit import Final
import typing_extensions as tpx

from torchani.constants import (
    ATOMIC_NUMBER,
    FUNCTIONAL_D3BJ_CONSTANTS,
    COVALENT_RADIUS,
    SQRT_EMPIRICAL_CHARGE,
)
from torchani.cutoffs import CutoffArg
from torchani.neighbors import Neighbors
from torchani.potentials.core import BasePairPotential
from torchani.paths import _resources_dir


# TODO: trainable?
# NOTE: Precalculated C6 constants for D3
# - Precalculated C6 coefficients
# shape (Elements, Elements, Ref, Ref), where "Ref" is the number of references
# (Grimme et. al. provides 5)
# This means for each pair of elements and reference indices there is an
# associated precalc C6 coeff
# - Precalculated coordination numbers
# shape (Elements, Elements, Ref, Ref, 2)
# Where the final axis indexes the coordination number of the first or second
# atom respectively.
# This means for each pair of elements and reference indices there is an
# associated coordination number for the first and second items.
def _load_c6_constants() -> tp.Tuple[Tensor, Tensor, Tensor]:
    with h5py.File(str(_resources_dir() / "c6.h5"), "r") as f:
        c6_constants = torch.from_numpy(f["all/constants"][:])
        c6_coordnums_a = torch.from_numpy(f["all/coordnums_a"][:])
        c6_coordnums_b = torch.from_numpy(f["all/coordnums_b"][:])
    return c6_constants, c6_coordnums_a, c6_coordnums_b


class BeckeJohnsonDamp(torch.nn.Module):
    r"""Implementation of Becke-Johnson style damping

    Damp functions are like cutoff functions, but modulate potentials close to
    zero.

    For modulating potentials of different "order" (e.g. ``1 / r ** 6`` => order 6),
    different parameters may be needed.

    For BJ damping style, the cutoff radii are by default calculated directly
    from the order 8 and order 6 coeffs, via the square root of the effective
    charges. Note that the cutoff radii is a matrix of T x T where T are the
    possible atom types and that these cutoff radii are in AU (Bohr)
    """

    _a1: Final[float]
    _a2: Final[float]

    def __init__(
        self,
        symbols: tp.Sequence[str],
        a1: float,
        a2: float,
        sqrt_empirical_charge: tp.Sequence[float] = (),
    ):
        super().__init__()
        self.atomic_numbers = torch.tensor(
            [ATOMIC_NUMBER[e] for e in symbols], dtype=torch.long
        )
        if not sqrt_empirical_charge:
            sqrt_empirical_charge = [
                SQRT_EMPIRICAL_CHARGE[j] for j in self.atomic_numbers
            ]

        if not len(sqrt_empirical_charge) == len(symbols):
            raise ValueError(
                "len(sqrt_empirical_charge), if provided, must match len(symbols)"
            )
        _sqrt_empirical_charge = torch.tensor(sqrt_empirical_charge)
        self.register_buffer(
            "cutoff_radii",
            torch.sqrt(3 * torch.outer(_sqrt_empirical_charge, _sqrt_empirical_charge)),
        )
        # Sanity check
        assert self.cutoff_radii.shape == (len(symbols), len(symbols))
        self._a1 = a1
        self._a2 = a2

    @classmethod
    def from_functional(
        cls,
        symbols: tp.Sequence[str],
        functional: str,
    ) -> tpx.Self:
        d = FUNCTIONAL_D3BJ_CONSTANTS[functional.lower()]
        return cls(symbols, a1=d["a1"], a2=d["a2"])

    def forward(
        self,
        species12: Tensor,
        distances: Tensor,
        order: int,
    ) -> Tensor:
        cutoff_radii = self.cutoff_radii[species12[0], species12[1]]
        damp_term = (self._a1 * cutoff_radii + self._a2).pow(order)
        return distances.pow(order) + damp_term


class TwoBodyDispersionD3(BasePairPotential):
    r"""Calculates the DFT-D3 dispersion corrections

    Only calculates the 2-body part of the dispersion corrections. Requires a
    damping function for the order-6 and order-8 potential terms.
    """

    _s6: Final[float]
    _s8: Final[float]
    _k1: Final[int]
    _k2: Final[float]
    _k3: Final[int]

    # Needed for bw compatibility
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        old_keys = list(state_dict.keys())
        for k in old_keys:
            if "damp_fn_6" in k:
                state_dict.pop(k)
                continue
            if "damp_fn_8" in k:
                new_key = k.replace("damp_fn_8", "damp_fn")
            else:
                new_key = k
            state_dict[new_key] = state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def __init__(
        self,
        symbols: tp.Sequence[str],
        # Potential
        s6: float,
        s8: float,
        damp_a1: float,
        damp_a2: float,
        sqrt_empirical_charge: tp.Sequence[float] = (),
        covalent_radii: tp.Sequence[float] = (),
        *,  # Cutoff
        cutoff_fn: CutoffArg = "smooth",
        cutoff: float = math.inf,
    ):
        super().__init__(symbols, cutoff=cutoff, cutoff_fn=cutoff_fn)
        sqrt_empirical_charge = self._validate_elem_seq(
            "sqrt_empirical_charge", sqrt_empirical_charge, SQRT_EMPIRICAL_CHARGE
        )

        covalent_radii = self._validate_elem_seq(
            "covalent_radii", covalent_radii, COVALENT_RADIUS
        )
        # Convert to Bohr since they are expected to be in angstrom
        covalent_radii = [self.ANGSTROM_TO_BOHR * r for r in covalent_radii]

        self._damp_fn = BeckeJohnsonDamp(
            symbols, damp_a1, damp_a2, sqrt_empirical_charge
        )
        self._s6 = s6
        self._s8 = s8

        # Hardcoded values from Grimme et. al.
        self._k1 = 16
        self._k2 = 4 / 3
        self._k3 = 4

        # Needed to get around numerical issues
        self._eps = 1e-35

        order6_constants, coordnums_a, coordnums_b = _load_c6_constants()
        self.register_buffer(
            "precalc_coeff6",
            order6_constants[self.atomic_numbers, :][:, self.atomic_numbers],
        )
        self.register_buffer(
            "precalc_coordnums_a",
            coordnums_a[self.atomic_numbers, :][:, self.atomic_numbers],
        )
        self.register_buffer(
            "precalc_coordnums_b",
            coordnums_b[self.atomic_numbers, :][:, self.atomic_numbers],
        )
        # The product of the sqrt of the empirical q's is stored directly
        _sqrt_empirical_charge = torch.tensor(sqrt_empirical_charge)
        self.register_buffer(
            "sqrt_charge_ab",
            torch.outer(_sqrt_empirical_charge, _sqrt_empirical_charge),
        )

        self.register_buffer("covalent_radii", torch.tensor(covalent_radii))

    @classmethod
    def from_functional(
        cls,
        symbols: tp.Sequence[str],
        functional: str,
        *,
        cutoff_fn: CutoffArg = "smooth",
        cutoff: float = math.inf,
    ) -> tpx.Self:
        d = FUNCTIONAL_D3BJ_CONSTANTS[functional.lower()]
        return cls(
            s6=d["s6"],
            s8=d["s8"],
            damp_a1=d["a1"],
            damp_a2=d["a2"],
            symbols=symbols,
            cutoff_fn=cutoff_fn,
            cutoff=cutoff,
        )

    def pair_energies(
        self,
        element_idxs: Tensor,
        neighbors: Neighbors,
    ) -> Tensor:
        # Internally this module works in AU
        distances = self.ANGSTROM_TO_BOHR * neighbors.distances

        species12 = element_idxs.view(-1)[neighbors.indices]
        molecs, atoms = element_idxs.shape

        # Shape is num_molecules * num_atoms
        coordnums = self._coordnums(
            molecs, atoms, species12, neighbors.indices, distances
        )

        # Order 6 and 8 coeff
        order6_coeff = self._interpolate_coeff6(species12, coordnums, neighbors.indices)
        order8_coeff = (
            3 * order6_coeff * self.sqrt_charge_ab[species12[0], species12[1]]
        )

        # Order 6 and 8 energies
        order6_energy = self._s6 * order6_coeff / self._damp_fn(species12, distances, 6)
        order8_energy = self._s8 * order8_coeff / self._damp_fn(species12, distances, 8)
        return -(order6_energy + order8_energy)

    # Use the coordination numbers and the internal precalc C6's and
    # CNa's/CNb's to get interpolated C6 coeffs, C8 coeffs are obtained from C6
    # coeffs directly Output shape is (A,)
    def _coordnums(
        self,
        num_molecules: int,
        num_atoms: int,
        species12: Tensor,
        atom_index12: Tensor,
        distances: Tensor,
    ) -> Tensor:
        # For coordination numbers "covalent radii" are used, not "cutoff radii"
        covalent_radii_sum = (
            self.covalent_radii[species12[0]] + self.covalent_radii[species12[1]]
        )

        count_fn = 1 / (
            1 + torch.exp(-self._k1 * (self._k2 * covalent_radii_sum / distances - 1))
        )

        # Add terms corresponding to all neighbors
        coordnums = distances.new_zeros((num_molecules * num_atoms))
        coordnums.index_add_(0, atom_index12[0], count_fn)
        coordnums.index_add_(0, atom_index12[1], count_fn)
        return coordnums

    def _interpolate_coeff6(
        self, species12: Tensor, coordnums: Tensor, atom_index12: Tensor
    ) -> Tensor:
        assert coordnums.ndim == 1, "coordnums must be one dimensional"
        assert species12.ndim == 2, "species12 must be 2 dimensional"

        # Find pre-computed values for every species pair, and flatten over all
        # references shape is (num_pairs, 5, 5) flat-> (num_pairs, 25)
        precalc_coeff6 = self.precalc_coeff6[
            species12[0],
            species12[1],
        ].flatten(1, 2)
        precalc_cn_a = self.precalc_coordnums_a[
            species12[0],
            species12[1],
        ].flatten(1, 2)
        precalc_cn_b = self.precalc_coordnums_b[
            species12[0],
            species12[1],
        ].flatten(1, 2)

        gauss_dist = (coordnums[atom_index12[0]].view(-1, 1) - precalc_cn_a) ** 2 + (
            coordnums[atom_index12[1]].view(-1, 1) - precalc_cn_b
        ) ** 2
        # Extra factor of gauss_dist.mean() and + 20 needed for numerical stability
        gauss_dist = torch.exp(-self._k3 * gauss_dist)
        # only consider C6 coefficients strictly greater than zero,
        # don't include -1 and 0.0 terms in the sums,
        # all missing parameters (with -1.0 values) are guaranteed to be the
        # same for precalc_cn_a/b and precalc_order6
        gauss_dist = gauss_dist.masked_fill(precalc_coeff6 <= 0.0, 0.0)
        # sum over references for w factor and z factor
        # This is needed for numerical stability, it will give 1 if W or Z are not
        # >> 1e-35 but those situations are rare in practice, and it avoids all
        # issues with NaN and exploding numbers / vanishing quantities
        w_factor = gauss_dist.sum(-1) + self._eps
        z_factor = (precalc_coeff6 * gauss_dist).sum(-1) + self._eps
        return z_factor / w_factor
