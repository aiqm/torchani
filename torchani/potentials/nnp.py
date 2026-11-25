import torch
from torch import Tensor

from torchani.nn import AtomicContainer
from torchani.aev import AEVComputer
from torchani.electro import ChargeNormalizer, BaseChargeNormalizer
from torchani.neighbors import Neighbors
from torchani.potentials.core import Potential
from torchani.sae import SelfEnergy


# Adaptor to use the aev computer as a three body potential
class NNPotential(Potential):
    def __init__(self, aev_computer: AEVComputer, neural_networks: AtomicContainer):
        super().__init__(neural_networks.symbols, cutoff=aev_computer.radial.cutoff)
        self.aev_computer = aev_computer
        self.neural_networks = neural_networks

    def compute_from_neighbors(
        self,
        elem_idxs: Tensor,
        neighbors: Neighbors,
        features: Tensor | None = None,
        charge: Tensor | None = None,
        spin_mult: Tensor | None = None,
        atomic: bool = False,
        ensemble_values: bool = False,
        ghost_flags: Tensor | None = None,
    ) -> dict[str, Tensor]:
        assert features is None
        aevs = self.aev_computer.compute_from_neighbors(elem_idxs, neighbors)
        energies = self.neural_networks(elem_idxs, aevs, atomic, ensemble_values)
        return {"energies": energies}


# Output of NN is assumed to be of shape (molecules, 2) with
# out[:, 0] = energies
# out[:, 1] = charges
class MergedChargesNNPotential(NNPotential):
    def __init__(
        self,
        aev_computer: AEVComputer,
        neural_networks: AtomicContainer,
        charge_normalizer: BaseChargeNormalizer | None = None,
    ):
        super().__init__(aev_computer, neural_networks)
        if charge_normalizer is None:
            charge_normalizer = ChargeNormalizer(self.symbols)
        self.charge_normalizer = charge_normalizer

    def compute_from_neighbors(
        self,
        elem_idxs: Tensor,
        neighbors: Neighbors,
        features: Tensor | None = None,
        charge: Tensor | None = None,
        spin_mult: Tensor | None = None,
        atomic: bool = False,
        ensemble_values: bool = False,
        ghost_flags: Tensor | None = None,
    ) -> dict[str, Tensor]:
        assert features is None
        aevs = self.aev_computer.compute_from_neighbors(elem_idxs, neighbors)
        # AtomicContainer is assumed to output a tensor with a final dimension "2"
        # which holds energies and charges
        energies_qs = self.neural_networks(elem_idxs, aevs, True, ensemble_values)
        assert energies_qs.shape == (
            elem_idxs.shape[0],
            elem_idxs.shape[1],
            2,
        ), "Incorrect shape for merged charge networks"
        energies, qs = energies_qs.unbind(-1)
        if not atomic:
            energies = energies.sum(dim=-1)
        return {
            "energies": energies,
            "atomic_charges": self.charge_normalizer(elem_idxs, qs, charge),
        }


class SeparateChargesNNPotential(NNPotential):
    def __init__(
        self,
        aev_computer: AEVComputer,
        neural_networks: AtomicContainer,
        charge_networks: AtomicContainer,
        charge_normalizer: BaseChargeNormalizer | None = None,
    ):
        super().__init__(aev_computer, neural_networks)
        if charge_normalizer is None:
            charge_normalizer = ChargeNormalizer(self.symbols)
        self.charge_networks = charge_networks
        self.charge_normalizer = charge_normalizer

    def compute_from_neighbors(
        self,
        elem_idxs: Tensor,
        neighbors: Neighbors,
        features: Tensor | None = None,
        charge: Tensor | None = None,
        spin_mult: Tensor | None = None,
        atomic: bool = False,
        ensemble_values: bool = False,
        ghost_flags: Tensor | None = None,
    ) -> dict[str, Tensor]:
        assert features is None
        aevs = self.aev_computer.compute_from_neighbors(elem_idxs, neighbors)
        energies = self.neural_networks(elem_idxs, aevs, atomic, ensemble_values)
        qs = self.charge_networks(elem_idxs, aevs, atomic=True)
        return {
            "energies": energies,
            "atomic_charges": self.charge_normalizer(elem_idxs, qs, charge),
        }


class SeparateHeadsNNPotential(NNPotential):
    r"""Potential that also outputs arbitrary named atomic properties

    Each property is predicted by its own head of atomic networks, which share the
    input features with the energy networks. Each head may have an output activation
    (e.g. ``Softplus`` to enforce positive values), applied after its shifter if
    ``shift_before_out_activation`` is ``True``, and before it otherwise. The
    output named ``"atomic_charges"`` is normalized with the charge normalizer.
    """

    def __init__(
        self,
        aev_computer: AEVComputer,
        neural_networks: AtomicContainer,
        extra_networks: dict[str, AtomicContainer],
        extra_networks_shifters: dict[str, SelfEnergy],
        charge_normalizer: BaseChargeNormalizer | None = None,
        extra_networks_out_activations: dict[str, torch.nn.Module] | None = None,
        shift_before_out_activation: bool = True,
    ):
        super().__init__(aev_computer, neural_networks)
        if set(extra_networks) != set(extra_networks_shifters):
            raise ValueError(
                "extra_networks and extra_networks_shifters must have equal keys"
            )
        if extra_networks_out_activations is None:
            extra_networks_out_activations = {}
        if not set(extra_networks_out_activations).issubset(extra_networks):
            raise ValueError(
                "extra_networks_out_activations has keys not in extra_networks"
            )
        if charge_normalizer is None:
            charge_normalizer = ChargeNormalizer(self.symbols)
        self.extra_networks = torch.nn.ModuleDict(extra_networks)
        self.extra_networks_shifters = torch.nn.ModuleDict(extra_networks_shifters)
        self.extra_networks_out_activations = torch.nn.ModuleDict(
            extra_networks_out_activations
        )
        self._shift_before_out_activation = shift_before_out_activation
        self.charge_normalizer = charge_normalizer

    def compute_from_neighbors(
        self,
        elem_idxs: Tensor,
        neighbors: Neighbors,
        features: Tensor | None = None,
        charge: Tensor | None = None,
        spin_mult: Tensor | None = None,
        atomic: bool = False,
        ensemble_values: bool = False,
        ghost_flags: Tensor | None = None,
    ) -> dict[str, Tensor]:
        assert features is None
        aevs = self.aev_computer.compute_from_neighbors(elem_idxs, neighbors)
        energies = self.neural_networks(elem_idxs, aevs, atomic, ensemble_values)
        out = {"energies": energies}
        for k, networks in self.extra_networks.items():
            out[k] = networks(elem_idxs, aevs, atomic=True)
        if not self._shift_before_out_activation:
            for k, out_activation in self.extra_networks_out_activations.items():
                out[k] = out_activation(out[k])
        for k, shifter in self.extra_networks_shifters.items():
            out[k] = out[k] + shifter(elem_idxs, atomic=True)
        if self._shift_before_out_activation:
            for k, out_activation in self.extra_networks_out_activations.items():
                out[k] = out_activation(out[k])
        is_padding = elem_idxs == -1
        for k in self.extra_networks_out_activations.keys():
            out[k] = out[k].masked_fill(is_padding, 0.0)
        if "atomic_charges" in out:
            qs = out["atomic_charges"]
            out["atomic_charges"] = self.charge_normalizer(elem_idxs, qs, charge)
        return out
