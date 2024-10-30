import torch

import typing as tp
from torchani.nn._core import AtomicNetwork


# Protocol used by all factories
class AtomicMaker(tp.Protocol):
    def __call__(
        self,
        symbol: str,
        in_dim: int,
        # The following are dummy defaults.
        # As long as the protocol has something of the correct type as a kwarg,
        # it will type-check correctly. In reality different AtomicMaker
        # have different defaults
        out_dim: int = 1,
        activation: tp.Union[str, torch.nn.Module] = "celu",
        bias: bool = True,
    ) -> AtomicNetwork:
        pass


def make_1x_network(
    symbol: str,
    in_dim: int = 384,
    out_dim: int = 1,
    activation: tp.Union[str, torch.nn.Module] = "celu",
    bias: bool = True,
) -> AtomicNetwork:
    r"""
    Makes an atomic network. Defaults are the ones in the ANI-1x (and 1ccx) model
    """
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


def make_2x_network(
    symbol: str,
    in_dim: int = 1008,
    out_dim: int = 1,
    activation: tp.Union[str, torch.nn.Module] = "celu",
    bias: bool = True,
) -> AtomicNetwork:
    r"""
    Makes an atomic network. Defaults are the ones in the ANI-2x model
    """
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
def make_ala_network(
    symbol: str,
    in_dim: int = 1008,
    out_dim: int = 1,
    activation: tp.Union[str, torch.nn.Module] = "celu",
    bias: bool = True,
) -> AtomicNetwork:
    r"""
    Makes an atomic network. Defaults are the ones in the ANI-ala model
    """
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


def make_dr_network(
    symbol: str,
    in_dim: int = 1008,
    out_dim: int = 1,
    activation: tp.Union[str, torch.nn.Module] = "gelu",
    bias: bool = False,
) -> AtomicNetwork:
    r"""
    Makes an atomic network. Defaults are the ones in the ANI-dr model
    """
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


AtomicMakerArg = tp.Union[
    tp.Literal["ani1x", "ani1ccx", "ani2x", "aniala", "anidr"], AtomicMaker
]


def _parse_network_maker(module: AtomicMakerArg) -> AtomicMaker:
    if module in ["ani1x", "ani1ccx"]:
        return make_1x_network
    elif module == "ani2x":
        return make_2x_network
    elif module == "aniala":
        return make_ala_network
    elif module == "anidr":
        return make_dr_network
    assert not isinstance(module, str)  # mypy
    return module
