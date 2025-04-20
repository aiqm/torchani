import typing as tp

import torch
from torch import Tensor
from torchani._core import _ChemModule
from torchani.utils import PERIODIC_TABLE


class _Embedding(_ChemModule):
    def forward(self, elem_idxs: Tensor) -> Tensor:
        raise NotImplementedError("Must be implemented by subclasses")


class AtomicOneHot(_Embedding):
    r"""Embed a sequence of atoms into one-hot vectors

    Padding atoms are set to zeros. As an example:

    .. code-block:: python

        symbols = ("H", "C", "N")
        one_hot = AtomicOneHot(symbols)
        encoded = one_hot(torch.tensor([1, 0, 2, -1]))
        # encoded == torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]])

    """
    def __init__(self, symbols: tp.Sequence[str]) -> None:
        super().__init__(symbols)
        num = len(self.symbols)
        one_hot = torch.zeros(num + 1, num)
        one_hot[torch.arange(num), torch.arange(num)] = 1
        self.register_buffer("one_hot", one_hot)

    def forward(self, elem_idxs: Tensor) -> Tensor:
        return self.one_hot[elem_idxs]


class AtomicEmbedding(_Embedding):
    r"""Embed a sequence of atoms into a continuous vector space

    This module is a thin wrapper over `torch.nn.Embedding`. Padding
    atoms are set to zero. As an example:

    .. code-block:: python

        symbols = ("H", "C", "N")
        embed = AtomicEmbedding(symbols, 2)
        encoded = embed(torch.tensor([1, 0, 2, -1]))
        # `encoded` depends on the random init, but it could be for instance:
        # torch.tensor([[1.2, .1], [-.5, .8], [.3, -.4], [0, 0]])
    """
    def __init__(self, symbols: tp.Sequence[str], dim: int = 10) -> None:
        super().__init__(symbols)
        num = len(self.symbols)
        self.embed = torch.nn.Embedding(num + 1, dim, padding_idx=num)

    def forward(self, elem_idxs: Tensor) -> Tensor:
        padding_idx = self.embed.padding_idx
        assert padding_idx is not None  # mypy
        _elem_idxs = elem_idxs.clone()
        _elem_idxs[elem_idxs == -1] = padding_idx
        return self.embed(_elem_idxs)


class AtomicContainer(torch.nn.Module):
    r"""Base class for ANI modules that contain Atomic Neural Networks"""

    num_species: int
    total_members_num: int
    active_members_idxs: tp.List[int]

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        super().__init__()
        self.total_members_num = 1
        self.active_members_idxs = [0]
        self.num_species = 0
        atomic_numbers = torch.tensor([0], dtype=torch.long)
        self.register_buffer("atomic_numbers", atomic_numbers, persistent=False)

    def forward(
        self,
        elem_idxs: Tensor,
        aevs: tp.Optional[Tensor] = None,
        atomic: bool = False,
        ensemble_values: bool = False,
    ) -> Tensor:
        assert aevs is not None
        if atomic:
            return aevs.new_zeros(elem_idxs.shape)
        return aevs.new_zeros(elem_idxs.shape[0])

    @property
    @torch.jit.unused
    def symbols(self) -> tp.Tuple[str, ...]:
        return tuple(PERIODIC_TABLE[z] for z in self.atomic_numbers)

    @torch.jit.export
    def get_active_members_num(self) -> int:
        return len(self.active_members_idxs)

    @torch.jit.export
    def set_active_members(self, idxs: tp.List[int]) -> None:
        for idx in idxs:
            if not (0 <= idx < self.total_members_num):
                raise IndexError(
                    f"Idx {idx} should be 0 <= idx < {self.total_members_num}"
                )
        self.active_members_idxs = idxs

    @torch.jit.unused
    def to_infer_model(self, use_mnp: bool = False) -> "AtomicContainer":
        return self


class AtomicNetwork(torch.nn.Module):
    def __init__(
        self,
        layer_dims: tp.Sequence[int],
        activation: tp.Union[str, torch.nn.Module] = "gelu",
        bias: bool = False,
    ) -> None:
        super().__init__()
        if any(d <= 0 for d in layer_dims):
            raise ValueError("Layer dims must be strict positive integers")

        dims = tuple(layer_dims)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(_in, _out, bias=bias, dtype=torch.float)
                for _in, _out in zip(dims[:-2], dims[1:-1])
            ]
        )
        self.final_layer = torch.nn.Linear(dims[-2], dims[-1], bias=bias)
        self.activation = parse_activation(activation)
        self.has_biases = bias

    def __getitem__(self, idx: int) -> torch.nn.Module:
        if idx in [-1, len(self.layers)]:
            return self.final_layer
        if idx < -1:
            idx += 1
        return self.layers[idx]

    def forward(self, features: Tensor) -> Tensor:
        for layer in self.layers:
            features = self.activation(layer(features))
        return self.final_layer(features)

    def extra_repr(self) -> str:
        r""":meta private:"""
        layer_dims = [layer.in_features for layer in self.layers]
        layer_dims.extend([self.final_layer.in_features, self.final_layer.out_features])
        parts = [
            f"layer_dims={tuple(layer_dims)},",
            f"activation={self.activation},",
            f"bias={self.has_biases},",
        ]
        return " \n".join(parts)


class TightCELU(torch.nn.Module):
    r"""CELU activation function with alpha=0.1"""
    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.celu(x, alpha=0.1)


def parse_activation(module: tp.Union[str, torch.nn.Module]) -> torch.nn.Module:
    if module == "gelu":
        return torch.nn.GELU()
    if module == "celu":
        return TightCELU()
    assert isinstance(module, torch.nn.Module)  # mypy
    return module
