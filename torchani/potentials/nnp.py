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
            symbols=symbols,
            cutoff=aev_computer.radial_terms.cutoff,
            is_trainable=True,
        )
        self.aev_computer = aev_computer
        self.neural_networks = neural_networks

    # TODO: Wrapper that executes the correct _compute_aev call, very dirty
    def _execute_aev_computer(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData,
        _coordinates: tp.Optional[Tensor] = None,
    ) -> Tensor:
        if (
            self.aev_computer.use_cuda_extension
            and self.aev_computer.use_cuaev_interface
        ):

            if not self.aev_computer.cuaev_is_initialized:
                self.aev_computer._init_cuaev_computer()
                self.aev_computer.cuaev_is_initialized = True
            assert _coordinates is not None
            return self.aev_computer._compute_cuaev_with_half_nbrlist(
                element_idxs, _coordinates, neighbors
            )
        return self.aev_computer._compute_aev(element_idxs, neighbors)

    def forward(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData,
        _coordinates: tp.Optional[Tensor] = None,
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> Tensor:
        aevs = self._execute_aev_computer(element_idxs, neighbors, _coordinates)
        return self.neural_networks((element_idxs, aevs))[1]

    @torch.jit.export
    def atomic_energies(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData,
        _coordinates: tp.Optional[Tensor] = None,
        ghost_flags: tp.Optional[Tensor] = None,
        ensemble_average: bool = True,
    ) -> Tensor:
        aevs = self._execute_aev_computer(element_idxs, neighbors, _coordinates)
        atomic_energies = self.neural_networks.members_atomic_energies(
            (element_idxs, aevs)
        )
        if atomic_energies.dim() == 2:
            atomic_energies = atomic_energies.unsqueeze(0)
        if ensemble_average:
            return atomic_energies.sum(0)
        return atomic_energies


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
            charge_normalizer = ChargeNormalizer(self.get_chemical_symbols())
        self.charge_normalizer = charge_normalizer

    @torch.jit.export
    def energies_and_atomic_charges(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData,
        _coordinates: tp.Optional[Tensor] = None,
        ghost_flags: tp.Optional[Tensor] = None,
        total_charge: int = 0,
    ) -> EnergiesAtomicCharges:
        aevs = self._execute_aev_computer(element_idxs, neighbors, _coordinates)
        atomic_energies_and_raw_atomic_charges = torch.sum(
            self.neural_networks.members_atomic_energies((element_idxs, aevs)), dim=0
        )  # shape is (M, C, A, 2)
        energies = torch.sum(atomic_energies_and_raw_atomic_charges[:, :, 0], dim=-1)
        raw_atomic_charges = atomic_energies_and_raw_atomic_charges[:, :, 1]
        atomic_charges = self.charge_normalizer(
            element_idxs, raw_atomic_charges, total_charge
        )
        return EnergiesAtomicCharges(energies, atomic_charges)


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
        _coordinates: tp.Optional[Tensor] = None,
        ghost_flags: tp.Optional[Tensor] = None,
        total_charge: int = 0,
    ) -> EnergiesAtomicCharges:
        aevs = self._execute_aev_computer(element_idxs, neighbors, _coordinates)
        energies = self.neural_networks((element_idxs, aevs))[1]
        raw_atomic_charges = torch.sum(
            self.charge_networks.members_atomic_energies((element_idxs, aevs)), dim=0
        )  # shape (M, C, A)
        atomic_charges = self.charge_normalizer(
            element_idxs, raw_atomic_charges, total_charge
        )
        return EnergiesAtomicCharges(energies, atomic_charges)
