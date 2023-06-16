import math
from typing import Union

import torch
from torch import Tensor
from torch.jit import Final


# All cutoffs assume the elements in "distances" are smaller than "cutoff"
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
    order: Final[int]
    eps: Final[float]

    def __init__(self, eps: float = 1e-10, order: int = 2):
        super().__init__()
        # Higher orders make the cutoff more similar to 1
        # for a wider range of distances, before the cutoff.
        # lower orders distort the underlying function more
        assert order > 0, "order must be a positive integer greater than zero"
        assert order % 2 == 0, "Order must be even"

        self.order = order
        self.eps = eps

    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        e = 1 - 1 / (1 - (distances / cutoff) ** self.order).clamp(min=self.eps)
        return torch.exp(e)


def _parse_cutoff_fn(cutoff_fn: Union[str, Cutoff]) -> torch.nn.Module:
    if cutoff_fn == 'dummy':
        cutoff_fn = CutoffDummy()
    elif cutoff_fn == 'cosine':
        cutoff_fn = CutoffCosine()
    elif cutoff_fn == 'smooth':
        cutoff_fn = CutoffSmooth()
    else:
        assert isinstance(cutoff_fn, Cutoff)
    return cutoff_fn
