r"""Collection of Cutoff functions

These can be used to envelope the outputs of `torchani.potentials.PairPotential` and
`torchani.aev.AEVComputer`
"""

import typing as tp
import math

import torch
from torch import Tensor


# All cutoffs assume the elements in "distances" are smaller than "cutoff"
# all parameters of a Cutoff **must be passed to init of the superclass**
# If cuaev supports the cutoff _cuaev_name must be defined to be a unique string
class Cutoff(torch.nn.Module):
    r"""Base class for cutoff functions"""

    _cuaev_name: str

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        super().__init__()
        self._fn_params = args + tuple(kwargs.values())
        self._cuaev_name = ""

    @torch.jit.unused
    def is_same(self, other: object) -> bool:
        # This is a hack since __eq__ can't be safely overriden in torch.nn.Module
        if not isinstance(other, Cutoff):
            return False
        if type(self) is not type(other):
            return False
        if not self._fn_params == other._fn_params:
            return False
        return True

    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        r"""Calculates factors that modify a scalar function, using pair distances"""
        raise NotImplementedError


class CutoffDummy(Cutoff):
    r"""Dummy cutoff that returns ones as factors"""

    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        return torch.ones_like(distances)


class CutoffCosine(Cutoff):
    r"""Use a cosine function as a cutoff"""

    def __init__(self) -> None:
        super().__init__()
        self._cuaev_name = "cosine"

    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        return 0.5 * torch.cos(distances * (math.pi / cutoff)) + 0.5


class CutoffSmooth(Cutoff):
    r"""Use an infinitely differentiable exponential cutoff"""

    def __init__(self, order: int = 2, eps: float = 1.0e-10) -> None:
        super().__init__(order, eps)
        if order == 2 and eps == 1.0e-10:
            self._cuaev_name = "smooth"
        self.order = order
        self.eps = eps

    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        e = 1 - 1 / (1 - (distances / cutoff) ** self.order).clamp(min=self.eps)
        return torch.exp(e)

    def extra_repr(self) -> str:
        r""":meta private:"""
        return f"order={self.order}, eps={self.eps:.1e}"


CutoffArg = tp.Union[
    tp.Literal["global", "dummy", "cosine", "smooth"],
    Cutoff,
]


def _parse_cutoff_fn(
    cutoff_fn: CutoffArg,
    global_cutoff: tp.Optional[Cutoff] = None,
) -> Cutoff:
    if cutoff_fn == "global":
        assert global_cutoff is not None
        cutoff_fn = global_cutoff
    if cutoff_fn == "dummy":
        cutoff_fn = CutoffDummy()
    elif cutoff_fn == "cosine":
        cutoff_fn = CutoffCosine()
    elif cutoff_fn == "smooth":
        cutoff_fn = CutoffSmooth()
    elif not isinstance(cutoff_fn, Cutoff):
        raise ValueError(f"Unsupported cutoff fn: {cutoff_fn}")
    return tp.cast(Cutoff, cutoff_fn)
