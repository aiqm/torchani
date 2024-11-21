import typing as tp

import torch
from torch import Tensor

from torchani.nn import AtomicContainer
from torchani.aev import AEVComputer
from torchani.electro import ChargeNormalizer
from torchani.tuples import EnergiesAtomicCharges
from torchani.neighbors import Neighbors
from torchani.potentials.core import Potential


# Adaptor to use the aev computer as a three body potential
class NNPotential(Potential):
    def __init__(self, aev_computer: AEVComputer, neural_networks: AtomicContainer):
        symbols = tuple(k for k in neural_networks.member(0).atomics)
        super().__init__(symbols, cutoff=aev_computer.radial.cutoff)
        self.aev_computer = aev_computer
        self.neural_networks = neural_networks

    def compute_from_neighbors(
        self,
        elem_idxs: Tensor,
        coords: Tensor,
        neighbors: Neighbors,
        atomic: bool = False,
        ensemble_values: bool = False,
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> Tensor:
        aevs = self.aev_computer.compute_from_neighbors(elem_idxs, coords, neighbors)
        return self.neural_networks(elem_idxs, aevs, atomic, ensemble_values)


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
        raise ValueError("This class is currently under development")

    @torch.jit.export
    def energies_and_atomic_charges(
        self,
        elem_idxs: Tensor,
        coords: Tensor,
        neighbors: Neighbors,
        charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> EnergiesAtomicCharges:
        assert not ensemble_values, "Unsupported"
        aevs = self.aev_computer.compute_from_neighbors(elem_idxs, coords, neighbors)
        energies_qs = self.neural_networks(elem_idxs, aevs, atomic=True)
        energies = energies_qs[:, :, 0]
        if not atomic:
            energies = energies.sum(dim=-1)
        qs = self.charge_normalizer(elem_idxs, energies_qs[:, :, 1], charge)
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
        coords: Tensor,
        neighbors: Neighbors,
        charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> EnergiesAtomicCharges:
        assert not ensemble_values, "Unsupported"
        aevs = self.aev_computer.compute_from_neighbors(elem_idxs, coords, neighbors)
        energies = self.neural_networks(elem_idxs, aevs, atomic=atomic)
        qs = self.charge_networks(elem_idxs, aevs, atomic=True)
        qs = self.charge_normalizer(elem_idxs, qs, charge)
        return EnergiesAtomicCharges(energies, qs)
