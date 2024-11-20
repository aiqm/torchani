import math
import typing as tp
import torch
from torch import Tensor
from torchani.cutoffs import CutoffArg
from torchani.potentials.core import BasePairPotential
from torchani.neighbors import Neighbors


# Ziegler-Biersack-Littman (ZBL) repulsion potential, which
# models screened nuclear repulsion
#  Chapter 2, "Universal screening fn" etc
# TODO: trainable
class RepulsionZBL(BasePairPotential):
    def __init__(
        self,
        symbols: tp.Sequence[str],
        # Potential
        k: float = 0.8853,  # LAMMPS uses 0.46850 (internally uses angstrom)
        screen_coeffs: tp.Sequence[float] = (),
        screen_exponents: tp.Sequence[float] = (),
        eff_exponent: float = 0.23,
        eff_atomic_nums: tp.Sequence[float] = (),
        trainable: tp.Sequence[str] = (),
        *,  # Cutoff
        cutoff: float = math.inf,
        cutoff_fn: CutoffArg = "smooth",
    ):
        super().__init__(symbols, cutoff=cutoff, cutoff_fn=cutoff_fn)
        eff_atomic_nums = self._validate_elem_seq(
            "eff_atomic_nums",
            eff_atomic_nums,
            torch.arange(118, dtype=torch.float).tolist(),
        )

        if not len(screen_exponents) == len(screen_coeffs):
            raise ValueError(
                "screen_exponents and screen_coeffs must have the same len"
            )
        # Defaults from Ziegler et. al.
        # Note: In Ziegler et.al. the parameters are:
        # coeffs: 0.1818, 0.5099 0.2802 0.02817 (typo in a part of the book)
        # exponents: 3.2, 0.9423, 0.4028, 0.2016
        # These parameters come directly from the LAMMPS code
        if not screen_coeffs:
            # Lammps uses 0.02817 for the last coeff, and that is needed to make
            # the sum be 1. There is probably a typo in the SRIM book
            screen_coeffs = [0.18175, 0.50986, 0.28022, 0.02817]
        if not screen_exponents:
            screen_exponents = [3.19980, 0.94229, 0.40290, 0.20162]

        if not math.isclose(sum(screen_coeffs), 1.0):
            raise ValueError("Screen coeffs must sum to 1")

        # Params and buffers
        self.register_buffer(
            "_eff_atomic_nums", torch.tensor(eff_atomic_nums), persistent=False
        )
        self.register_buffer(
            "_coeffs", torch.tensor(screen_coeffs).view(1, -1), persistent=False
        )
        self.register_buffer(
            "_exponents", torch.tensor(screen_exponents).view(1, -1), persistent=False
        )
        self._k = k
        self._kz = eff_exponent

    def pair_energies(self, elem_idxs: Tensor, neighbors: Neighbors) -> Tensor:
        # Clamp distances to prevent singularities when dividing by zero
        # All internal calcs use atomic units, so convert to Bohr
        dists = self.clamp(neighbors.distances) * self.ANGSTROM_TO_BOHR
        elem_pairs = elem_idxs.view(-1)[neighbors.indices]
        eff_za = self._eff_atomic_nums[elem_pairs[0]]
        eff_zb = self._eff_atomic_nums[elem_pairs[1]]
        eff_coulomb_term = eff_za * eff_zb / dists
        reduced_dists = dists * (eff_za**self._kz + eff_zb**self._kz) / self._k
        screen_factor = self.screen_fn(reduced_dists)
        return eff_coulomb_term * screen_factor

    def screen_fn(self, dists: Tensor) -> Tensor:  # Output shape is same as input
        return (self._coeffs * torch.exp(-self._exponents * dists.view(-1, 1))).sum(-1)
