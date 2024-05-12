r"""Factory methods that create atomic networks of different kinds"""
import typing as tp
from copy import deepcopy

import torch


def _parse_activation(module: tp.Union[str, torch.nn.Module]) -> torch.nn.Module:
    if module == "gelu":
        return torch.nn.GELU()
    if module == "celu":
        return torch.nn.CELU(0.1)
    assert not isinstance(module, str)  # mypy
    return module


def _parse_atomics(
    module: tp.Union[str, tp.Callable[[str, int], torch.nn.Module]],
) -> tp.Callable[[str, int], torch.nn.Module]:
    if module == "ani1x":
        return like_1x
    elif module == "ani2x":
        return like_2x
    elif module == "anidr":
        return like_dr
    elif module == "aniala":
        return like_ala
    elif module == "ani1ccx":
        return like_1ccx
    assert not isinstance(module, str)  # mypy
    return module


def standard(
    dims: tp.Sequence[int],
    activation: tp.Optional[torch.nn.Module] = None,
    bias: bool = False,
    classifier_out: int = 1,
):
    r"""Makes a standard ANI style atomic network"""
    if activation is None:
        activation = torch.nn.GELU()

    dims = list(deepcopy(dims))
    layers = []
    for dim_in, dim_out in zip(dims[:-1], dims[1:]):
        layers.extend([torch.nn.Linear(dim_in, dim_out, bias=bias), activation])
    # final layer is a linear classifier that is always appended
    layers.append(torch.nn.Linear(dims[-1], classifier_out, bias=bias))

    assert len(layers) == (len(dims) - 1) * 2 + 1
    return torch.nn.Sequential(*layers)


def like_1x(
    atom: str = "H",
    feat_dim: int = 384,
    activation: tp.Optional[torch.nn.Module] = None,
    bias: bool = True,
    classifier_out: int = 1,
):
    r"""Makes an atomic network. Defaults are the ones in the ANI-1x (and 1ccx) model"""
    if activation is None:
        activation = torch.nn.CELU(0.1)
    dims_for_atoms = {
        "H": (feat_dim, 160, 128, 96),
        "C": (feat_dim, 144, 112, 96),
        "N": (feat_dim, 128, 112, 96),
        "O": (feat_dim, 128, 112, 96),
    }
    return standard(
        dims_for_atoms[atom],
        activation=activation,
        bias=bias,
        classifier_out=classifier_out,
    )


def like_ala(
    atom: str = "H",
    feat_dim: int = 1008,
    activation: tp.Optional[torch.nn.Module] = None,
    bias: bool = True,
    classifier_out: int = 1,
):
    r"""Makes an atomic network. The defaults are the ones used in the ANI-2x model"""
    if activation is None:
        activation = torch.nn.CELU(0.1)
    dims_for_atoms = {
        "H": (feat_dim, 256, 192, 160),
        "C": (feat_dim, 224, 196, 160),
        "N": (feat_dim, 192, 160, 128),
        "O": (feat_dim, 192, 160, 128),
        "S": (feat_dim, 160, 128, 96),
        "F": (feat_dim, 160, 128, 96),
        "Cl": (feat_dim, 160, 128, 96),
    }
    return standard(
        dims_for_atoms[atom],
        activation=activation,
        bias=bias,
        classifier_out=classifier_out,
    )


def like_2x(
    atom: str = "H",
    feat_dim: int = 1008,
    activation: tp.Optional[torch.nn.Module] = None,
    bias: bool = True,
    classifier_out: int = 1,
):
    r"""Makes an atomic network. The defaults are the ones used in the ANI-2x model"""
    if activation is None:
        activation = torch.nn.CELU(0.1)
    dims_for_atoms = {
        "H": (feat_dim, 256, 192, 160),
        "C": (feat_dim, 224, 192, 160),
        "N": (feat_dim, 192, 160, 128),
        "O": (feat_dim, 192, 160, 128),
        "S": (feat_dim, 160, 128, 96),
        "F": (feat_dim, 160, 128, 96),
        "Cl": (feat_dim, 160, 128, 96),
    }
    return standard(
        dims_for_atoms[atom],
        activation=activation,
        bias=bias,
        classifier_out=classifier_out,
    )


def like_dr(
    atom: str = "H",
    feat_dim: int = 1008,
    activation: tp.Optional[torch.nn.Module] = None,
    bias: bool = False,
    classifier_out: int = 1,
):
    r"""Makes an atomic network. The defaults are the ones used in the ANI-dr model"""
    dims_for_atoms = {
        "H": (feat_dim, 256, 192, 160),
        "C": (feat_dim, 256, 192, 160),
        "N": (feat_dim, 192, 160, 128),
        "O": (feat_dim, 192, 160, 128),
        "S": (feat_dim, 160, 128, 96),
        "F": (feat_dim, 160, 128, 96),
        "Cl": (feat_dim, 160, 128, 96),
    }
    return standard(
        dims_for_atoms[atom],
        activation=activation,
        bias=bias,
        classifier_out=classifier_out,
    )


like_1ccx = like_1x
