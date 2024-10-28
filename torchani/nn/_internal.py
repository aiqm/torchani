r"""
This module is *internal* and considered an implementation detail. Functions and classes
are subject to change at any point. It contains legacy API and workarounds.
"""
import typing as tp
import warnings

import torch
from torch import Tensor

from torchani.tuples import SpeciesEnergies
from torchani.nn._core import AtomicContainer
from torchani.nn._containers import ANINetworks, ANIEnsemble


# Hack: ANINetworks that return zeros
class _ZeroANINetworks(ANINetworks):
    def forward(self, elem_idxs: Tensor, aevs: Tensor, atomic: bool = False) -> Tensor:
        if atomic:
            return aevs.new_zeros(elem_idxs.shape)
        return aevs.new_zeros(elem_idxs.shape[0])


# Hack: Wrapper that Grabs a network with "bad first scalar", and discards it
class _ANINetworksDiscardFirstScalar(ANINetworks):
    def forward(self, elem_idxs: Tensor, aevs: Tensor, atomic: bool = False) -> Tensor:
        assert elem_idxs.shape == aevs.shape[:-1]
        flat_elem_idxs = elem_idxs.flatten()
        aev = aevs.flatten(0, 1)
        scalars = aev.new_zeros(flat_elem_idxs.shape)
        for i, m in enumerate(self.atomics.values()):
            selected_idxs = (flat_elem_idxs == i).nonzero().view(-1)
            if selected_idxs.shape[0] > 0:
                input_ = aev.index_select(0, selected_idxs)
                scalars.index_add_(0, selected_idxs, m(input_)[:, 1].view(-1))
        scalars = scalars.view_as(elem_idxs)
        if atomic:
            return scalars
        return scalars.sum(dim=1)

    @torch.jit.unused
    def to_infer_model(self, use_mnp: bool = False) -> AtomicContainer:
        return self


# Legacy API
class Sequential(torch.nn.ModuleList):
    """Modified Sequential module that accept Tuple type as input"""

    def __init__(self, *modules):
        warnings.warn(
            "Use of `torchani.nn.Sequential` is strongly discouraged."
            "Please use the Assembler, or write your own torch.nn.Module."
            "For more information consult the 'migrating to 3' documentation"
        )
        super().__init__(modules)

    def forward(
        self,
        input_: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ):
        for module in self:
            if hasattr(module, "call"):
                input_ = module.call(input_, cell=cell, pbc=pbc)
            else:
                input_ = module(input_, cell=cell, pbc=pbc)
        return input_


class ANIModel(ANINetworks):
    def __init__(self, modules: tp.Any) -> None:
        warnings.warn(
            "torchani.ANIModel is deprecated, its use is discouraged."
            " Please use torchani.nn.ANINetworks instead."
        )
        super().__init__(modules, alias=True)

    # Signature is incompatible since this class is legacy
    def forward(  # type: ignore
        self,
        species_aevs: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> SpeciesEnergies:
        return self.call(species_aevs, cell, pbc)


class Ensemble(ANIEnsemble):
    def __init__(self, modules: tp.Any) -> None:
        warnings.warn(
            "torchani.Ensemble is deprecated, its use is discouraged."
            " Please use torchani.nn.ANIEnsemble instead."
        )
        super().__init__(modules, repeats=True)

    # Signature is incompatible since this class is legacy
    def forward(  # type: ignore
        self,
        species_aevs: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> SpeciesEnergies:
        return self.call(species_aevs, cell, pbc)
