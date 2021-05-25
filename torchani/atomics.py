r"""Factory methods that create atomic networks of different kinds"""
from copy import deepcopy
from typing import Sequence, Optional

import torch
from torch.nn import Module


def standard(dims: Sequence[int],
             activation: Optional[Module] = None,
             bias: bool = True,
             classifier_out: int = 1):
    r"""Makes a standard ANI style atomic network"""
    if activation is None:
        activation = torch.nn.CELU(0.1)
    else:
        activation = activation

    dims = list(deepcopy(dims))
    layers = []
    for dim_in, dim_out in zip(dims[:-1], dims[1:]):
        layers.extend([torch.nn.Linear(dim_in, dim_out, bias=bias), activation])
    # final layer is a linear classifier that is always appended
    layers.append(torch.nn.Linear(dims[-1], classifier_out, bias=bias))

    assert len(layers) == (len(dims) - 1) * 2 + 1
    return torch.nn.Sequential(*layers)


def like_1x(atom: str = 'H', **kwargs):
    r"""Makes a sequential atomic network like the one used in the ANI-1x model"""
    dims_for_atoms = {'H': (384, 160, 128, 96),
                      'C': (384, 144, 112, 96),
                      'N': (384, 128, 112, 96),
                      'O': (384, 128, 112, 96)}
    return standard(dims_for_atoms[atom], **kwargs)


def like_1ccx(atom: str = 'H', **kwargs):
    r"""Makes a sequential atomic network like the one used in the ANI-1ccx model"""
    return like_1x(atom=atom, **kwargs)


def like_2x(atom: str = 'H', **kwargs):
    r"""Makes a sequential atomic network like the one used in the ANI-2x model"""
    dims_for_atoms = {'H': (1008, 256, 192, 160),
                      'C': (1008, 224, 192, 160),
                      'N': (1008, 192, 160, 128),
                      'O': (1008, 192, 160, 128),
                      'S': (1008, 160, 128, 96),
                      'F': (1008, 160, 128, 96),
                      'Cl': (1008, 160, 128, 96)}
    return standard(dims_for_atoms[atom], **kwargs)
