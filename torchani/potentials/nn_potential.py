import typing as tp

import torch
from torch import Tensor

from torchani.electro import ChargeNormalizer
from torchani.tuples import EnergiesAtomicCharges
from torchani.neighbors import NeighborData
from torchani.atomics import AtomicContainer
from torchani.constants import PERIODIC_TABLE
from torchani.aev.computer import AEVComputer
from torchani.potentials.core import Potential


# Adaptor to use the aev computer as a three body potential
class NNPotential(Potential):
    def __init__(self, aev_computer: AEVComputer, neural_networks: AtomicContainer):
        # Fetch the symbols or "?" if they are not actually elements
        # NOTE: symbols that are not elements is supported for backwards
        # compatibility, since ANIModel supports arbitrary ordered dicts
        # as inputs.
        symbols = tuple(
            k if k in PERIODIC_TABLE else "?" for k in neural_networks.member(0).atomics
        )
        super().__init__(
            cutoff=aev_computer.radial_terms.cutoff, is_trainable=True, symbols=symbols
        )
        self.aev_computer = aev_computer
        self.neural_networks = neural_networks

    def forward(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData,
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> Tensor:
        aevs = self.aev_computer._compute_aev(element_idxs, neighbors)
        return self.neural_networks((element_idxs, aevs))[1]

    def atomic_energies(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData,
        ghost_flags: tp.Optional[Tensor] = None,
        average: bool = False,
    ) -> Tensor:
        aevs = self.aev_computer._compute_aev(element_idxs, neighbors)
        atomic_energies = self.neural_networks._atomic_energies((element_idxs, aevs))
        if atomic_energies.dim() == 2:
            atomic_energies = atomic_energies.unsqueeze(0)
        if average:
            return atomic_energies.sum(0)
        return atomic_energies


class SeparateChargesNNPotential(NNPotential):
    def __init__(
        self,
        aev_computer: AEVComputer,
        neural_networks: AtomicContainer,
        charge_networks: AtomicContainer,
        charge_normalizer: tp.Optional[ChargeNormalizer] = None,
    ):
        super().__init__(aev_computer, neural_networks)
        if charge_normalizer is None:
            charge_normalizer = ChargeNormalizer(self.get_chemical_symbols())
        self.charge_networks = charge_networks
        self.charge_normalizer = charge_normalizer

    @torch.jit.export
    def energies_and_atomic_charges(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData,
        ghost_flags: tp.Optional[Tensor] = None,
        total_charge: float = 0.0,
    ) -> EnergiesAtomicCharges:
        aevs = self.aev_computer._compute_aev(element_idxs, neighbors)
        energies = self.neural_networks((element_idxs, aevs))[1]
        raw_atomic_charges = self.charge_networks((element_idxs, aevs))[1]
        atomic_charges = self.charge_normalizer(
            element_idxs, raw_atomic_charges, total_charge
        )
        return EnergiesAtomicCharges(energies, atomic_charges)
