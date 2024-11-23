import typing as tp

import torch
from torch import Tensor


class AtomicContainer(torch.nn.Module):
    r"""Base class for ANI modules that contain Atomic Neural Networks"""

    num_species: int
    total_members_num: int
    active_members_idxs: tp.List[int]

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        super().__init__()
        self.total_members_num = 0
        self.active_members_idxs = []
        self.num_species = 0

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
    def member(self, idx: int) -> "AtomicContainer":
        if idx == 0:
            return self
        raise IndexError("Only idx=0 supported")

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
