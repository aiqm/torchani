import typing as tp
import math

import torch
from torch import Tensor


# All cutoffs assume the elements in "distances" are smaller than "cutoff"
# Cutoff modules must have no parameters
class Cutoff(torch.nn.Module):
    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        raise NotImplementedError


class CutoffDummy(Cutoff):
    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        return torch.ones_like(distances)


class CutoffCosine(Cutoff):
    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        return 0.5 * torch.cos(distances * (math.pi / cutoff)) + 0.5


class CutoffSmooth(Cutoff):
    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        order = 2
        eps = 1e-10
        e = 1 - 1 / (1 - (distances / cutoff) ** order).clamp(min=eps)
        return torch.exp(e)


class CutoffSmooth4(Cutoff):
    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        order = 4
        eps = 1e-10
        e = 1 - 1 / (1 - (distances / cutoff) ** order).clamp(min=eps)
        return torch.exp(e)


CutoffArg = tp.Union[str, Cutoff]


def parse_cutoff_fn(
    cutoff_fn: CutoffArg,
    global_cutoff: tp.Optional[Cutoff] = None,
) -> Cutoff:
    if cutoff_fn == "global":
        assert global_cutoff is not None
        cutoff_fn = global_cutoff
    if cutoff_fn == 'dummy':
        cutoff_fn = CutoffDummy()
    elif cutoff_fn == 'cosine':
        cutoff_fn = CutoffCosine()
    elif cutoff_fn in ('smooth', 'smooth2'):
        cutoff_fn = CutoffSmooth()
    elif cutoff_fn == 'smooth4':
        cutoff_fn = CutoffSmooth4()
    else:
        assert isinstance(cutoff_fn, Cutoff)
    return cutoff_fn
