import typing as tp

import torch
from torch import Tensor

from torchani.nn import AtomicContainer
from torchani.aev import AEVComputer
from torchani.electro import ChargeNormalizer
from torchani.tuples import EnergiesAtomicCharges
from torchani.neighbors import NeighborData
from torchani.potentials.core import Potential


# Adaptor to use the aev computer as a three body potential
class NNPotential(Potential):
    def __init__(self, aev_computer: AEVComputer, neural_networks: AtomicContainer):
        symbols = tuple(k for k in neural_networks.member(0).atomics)
        super().__init__(symbols=symbols, cutoff=aev_computer.radial_terms.cutoff)
        self.aev_computer = aev_computer
        self.neural_networks = neural_networks

    def forward(
        self,
        elem_idxs: Tensor,
        neighbors: NeighborData,
        _coordinates: tp.Optional[Tensor] = None,
        ghost_flags: tp.Optional[Tensor] = None,
        atomic: bool = False,
    ) -> Tensor:
        aevs = self.aev_computer.compute_from_neighbors(
            elem_idxs, neighbors, _coordinates
        )
        return self.neural_networks(elem_idxs, aevs, atomic=atomic)

    def ensemble_values(
        self,
        elem_idxs: Tensor,
        neighbors: NeighborData,
        _coordinates: tp.Optional[Tensor] = None,
        ghost_flags: tp.Optional[Tensor] = None,
        atomic: bool = False,
    ) -> Tensor:
        if hasattr(self.neural_networks, "ensemble_values"):
            aevs = self.aev_computer.compute_from_neighbors(
                elem_idxs, neighbors, _coordinates
            )
            out = self.neural_networks.ensemble_values(elem_idxs, aevs, atomic=atomic)
            return out
        out = self(elem_idxs, neighbors, _coordinates, ghost_flags, atomic).unsqueeze(0)
        return out


# Output of NN is assumed to be of shape (molecules, 2) with
# out[:, 0] = energies
# out[:, 1] = charges
class MergedChargesNNPotential(NNPotential):
    def __init__(
        self,
        aev_computer: AEVComputer,
        neural_networks: AtomicContainer,
        charge_normalizer: tp.Optional[ChargeNormalizer] = None,
    ):
        super().__init__(aev_computer, neural_networks)
        if charge_normalizer is None:
            charge_normalizer = ChargeNormalizer(self.symbols)
        self.charge_normalizer = charge_normalizer

    @torch.jit.export
    def energies_and_atomic_charges(
        self,
        elem_idxs: Tensor,
        neighbors: NeighborData,
        _coordinates: tp.Optional[Tensor] = None,
        ghost_flags: tp.Optional[Tensor] = None,
        total_charge: int = 0,
        atomic: bool = False,
    ) -> EnergiesAtomicCharges:
        aevs = self.aev_computer.compute_from_neighbors(
            elem_idxs, neighbors, _coordinates
        )
        energies_qs = self.neural_networks(elem_idxs, aevs, atomic=True)
        energies = energies_qs[:, :, 0]
        if not atomic:
            energies = energies.sum(dim=-1)
        qs = self.charge_normalizer(elem_idxs, energies_qs[:, :, 1], total_charge)
        return EnergiesAtomicCharges(energies, qs)


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
            charge_normalizer = ChargeNormalizer(self.symbols)
        self.charge_networks = charge_networks
        self.charge_normalizer = charge_normalizer

    @torch.jit.export
    def energies_and_atomic_charges(
        self,
        elem_idxs: Tensor,
        neighbors: NeighborData,
        _coordinates: tp.Optional[Tensor] = None,
        ghost_flags: tp.Optional[Tensor] = None,
        total_charge: int = 0,
        atomic: bool = False,
    ) -> EnergiesAtomicCharges:
        aevs = self.aev_computer.compute_from_neighbors(
            elem_idxs, neighbors, _coordinates
        )
        energies = self.neural_networks(elem_idxs, aevs, atomic=atomic)
        qs = self.charge_networks(elem_idxs, aevs, atomic=True)
        qs = self.charge_normalizer(elem_idxs, qs, total_charge)
        return EnergiesAtomicCharges(energies, qs)
