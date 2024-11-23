import typing as tp

from torch import Tensor

from torchani.tuples import EnergiesScalars
from torchani.nn import AtomicContainer
from torchani.aev import AEVComputer
from torchani.electro import ChargeNormalizer
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
        charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> EnergiesScalars:
        aevs = self.aev_computer.compute_from_neighbors(elem_idxs, coords, neighbors)
        energies = self.neural_networks(elem_idxs, aevs, atomic, ensemble_values)
        return EnergiesScalars(energies)


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

    def compute_from_neighbors(
        self,
        elem_idxs: Tensor,
        coords: Tensor,
        neighbors: Neighbors,
        charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> EnergiesScalars:
        aevs = self.aev_computer.compute_from_neighbors(elem_idxs, coords, neighbors)
        # AtomicContainer is assumed to output a tensor with a final dimension "2"
        # which holds energies and charges
        energies_qs = self.neural_networks(elem_idxs, aevs, True, ensemble_values)
        assert energies_qs.shape[1:] == (
            elem_idxs.shape[0],
            elem_idxs.shape[2],
            2,
        ), "Incorrect shape for merged charge networks"
        energies, qs = energies_qs.unbind(-1)
        if not atomic:
            energies = energies.sum(dim=-1)
        return EnergiesScalars(energies, self.charge_normalizer(elem_idxs, qs, charge))


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

    def compute_from_neighbors(
        self,
        elem_idxs: Tensor,
        coords: Tensor,
        neighbors: Neighbors,
        charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> EnergiesScalars:
        aevs = self.aev_computer.compute_from_neighbors(elem_idxs, coords, neighbors)
        energies = self.neural_networks(elem_idxs, aevs, atomic, ensemble_values)
        qs = self.charge_networks(elem_idxs, aevs, atomic=True)
        return EnergiesScalars(energies, self.charge_normalizer(elem_idxs, qs, charge))
