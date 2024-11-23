# Legacy classes and hacks
import typing as tp
import warnings

import torch
from torch import Tensor

from torchani.nn._core import AtomicContainer
from torchani.nn._containers import ANINetworks


# Legacy API
class ANIModel(ANINetworks):
    def __init__(self, modules: tp.Any) -> None:
        warnings.warn(
            "`torchani.nn.ANIModel` is deprecated, its use is discouraged."
            " Please use `torchani.nn.ANINetworks` instead, which is equivalent."
        )
        super().__init__(modules, alias=True)


# Legacy API
# Modified Sequential module that accepts Tuple type as input
class Sequential(torch.nn.ModuleList):
    r"""Create a pipeline of modules, like `torch.nn.Sequential`

    Deprecated:
        Use of `torchani.nn.Sequential` is strongly discouraged. Please use
        `torchani.arch.Assembler`, or write a `torch.nn.Module`. For more info
        consult `the migration guide <torchani-migrating>`
    """

    def __init__(self, *modules):
        warnings.warn(
            "Use of `torchani.nn.Sequential` is strongly discouraged."
            "Please use `torchani.arch.Assembler`, or write a `torch.nn.Module`."
            " For more info consult 'Migrating to TorchANI 3' in the user guide."
        )
        super().__init__(modules)

    def forward(
        self,
        input_: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ):
        r"""Return the result of chaining together the calculation of the modules"""
        for module in self:
            input_ = module(input_, cell, pbc)
        return input_


# Hack: ANINetworks that return zeros
class _ZeroANINetworks(ANINetworks):
    def forward(
        self,
        elem_idxs: Tensor,
        aevs: Tensor,
        atomic: bool = False,
        ensemble_values: bool = False,
    ) -> Tensor:
        if atomic:
            return aevs.new_zeros(elem_idxs.shape)
        return aevs.new_zeros(elem_idxs.shape[0])


# Hack: Wrapper that Grabs a network with "bad first scalar", and discards it
class _ANINetworksDiscardFirstScalar(ANINetworks):
    def forward(
        self,
        elem_idxs: Tensor,
        aevs: Tensor,
        atomic: bool = False,
        ensemble_values: bool = False,
    ) -> Tensor:
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
