r"""
Atomic Networks, Atomic Network Containers and factory methods with useful
defaults
"""
import typing as tp

import torch
from torch import Tensor

from torchani.utils import TightCELU


class AtomicContainer(torch.nn.Module):
    r"""Base class for ANI modules that contain Atomic Neural Networks"""
    num_networks: int
    num_species: int

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        super().__init__()
        self.num_networks = 0
        self.num_species = 0

    def forward(
        self,
        species_aev: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> tp.Tuple[Tensor, Tensor]:
        raise NotImplementedError()

    def member(self, idx: int) -> "AtomicContainer":
        if idx == 0:
            return self
        raise IndexError("Only idx=0 supported")

    @torch.jit.export
    def _atomic_energies(
        self,
        species_aev: tp.Tuple[Tensor, Tensor],
    ) -> Tensor:
        raise NotImplementedError()

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

    def extra_repr(self) -> str:
        layer_dims = [layer.in_features for layer in self.layers]
        layer_dims.extend([self.final_layer.in_features, self.final_layer.out_features])
        parts = [
            f"layer_dims={tuple(layer_dims)},",
            f"activation={self.activation},",
            f"bias={self.has_biases},",
        ]
        return " \n".join(parts)

    def forward(self, features: Tensor) -> Tensor:
        for layer in self.layers:
            features = self.activation(layer(features))
        return self.final_layer(features)


def parse_activation(module: tp.Union[str, torch.nn.Module]) -> torch.nn.Module:
    if module == "gelu":
        return torch.nn.GELU()
    if module == "celu":
        return TightCELU()
    assert isinstance(module, torch.nn.Module)  # mypy
    return module


def like_1x(
    symbol: str,
    in_dim: int = 384,
    out_dim: int = 1,
    activation: tp.Union[str, torch.nn.Module] = "celu",
    bias: bool = True,
) -> AtomicNetwork:
    r"""Makes an atomic network. Defaults are the ones in the ANI-1x (and 1ccx)
    model"""
    dims = {
        "H": (160, 128, 96),
        "C": (144, 112, 96),
        "N": (128, 112, 96),
        "O": (128, 112, 96),
    }
    layer_dims = (in_dim,) + dims[symbol] + (out_dim,)
    return AtomicNetwork(
        layer_dims=layer_dims,
        activation=activation,
        bias=bias,
    )


def like_2x(
    symbol: str,
    in_dim: int = 1008,
    out_dim: int = 1,
    activation: tp.Union[str, torch.nn.Module] = "celu",
    bias: bool = True,
) -> AtomicNetwork:
    r"""Makes an atomic network. Defaults are the ones in the ANI-2x model"""
    dims = {
        "H": (256, 192, 160),
        "C": (224, 192, 160),
        "N": (192, 160, 128),
        "O": (192, 160, 128),
        "S": (160, 128, 96),
        "F": (160, 128, 96),
        "Cl": (160, 128, 96),
    }
    layer_dims = (in_dim,) + dims[symbol] + (out_dim,)
    return AtomicNetwork(
        layer_dims=layer_dims,
        activation=activation,
        bias=bias,
    )


# NOTE: Same as 2x, but C layer has 196 instead of 192 nodes
def like_ala(
    symbol: str,
    in_dim: int = 1008,
    out_dim: int = 1,
    activation: tp.Union[str, torch.nn.Module] = "celu",
    bias: bool = True,
) -> AtomicNetwork:
    r"""Makes an atomic network. Defaults are the ones in the ANI-ala  model"""
    dims = {
        "H": (256, 192, 160),
        "C": (224, 196, 160),
        "N": (192, 160, 128),
        "O": (192, 160, 128),
        "S": (160, 128, 96),
        "F": (160, 128, 96),
        "Cl": (160, 128, 96),
    }
    layer_dims = (in_dim,) + dims[symbol] + (out_dim,)
    return AtomicNetwork(
        layer_dims=layer_dims,
        activation=activation,
        bias=bias,
    )


def like_dr(
    symbol: str,
    in_dim: int = 1008,
    out_dim: int = 1,
    activation: tp.Union[str, torch.nn.Module] = "gelu",
    bias: bool = False,
) -> AtomicNetwork:
    r"""Makes an atomic network. Defaults are the ones in the ANI-dr model"""
    dims = {
        "H": (256, 192, 160),
        "C": (256, 192, 160),
        "N": (192, 160, 128),
        "O": (192, 160, 128),
        "S": (160, 128, 96),
        "F": (160, 128, 96),
        "Cl": (160, 128, 96),
    }
    layer_dims = (in_dim,) + dims[symbol] + (out_dim,)
    return AtomicNetwork(
        layer_dims=layer_dims,
        activation=activation,
        bias=bias,
    )


def like_mbis_charges(
    symbol: str,
    in_dim: int = 1008,
    out_dim: int = 2,
    activation: tp.Union[str, torch.nn.Module] = "gelu",
    bias: bool = False,
) -> AtomicNetwork:
    r"""
    Makes an atomic network.
    The defaults are the ones used in the charge network of the ANI-mbis model
    """
    return like_2x(
        symbol,
        in_dim=in_dim,
        out_dim=out_dim,
        bias=False,
        activation=activation,
    )


AtomicMaker = tp.Callable[[str, int], AtomicNetwork]
AtomicMakerArg = tp.Union[str, AtomicMaker]


def parse_atomics(module: AtomicMakerArg) -> AtomicMaker:
    if module in ["ani1x", "ani1ccx"]:
        return like_1x
    elif module == "ani2x":
        return like_2x
    elif module == "aniala":
        return like_ala
    elif module == "anidr":
        return like_dr
    assert not isinstance(module, str)  # mypy
    return module
