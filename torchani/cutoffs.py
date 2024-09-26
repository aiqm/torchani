r"""
Collection of Cutoff functions, which can be used to smoothly limit the range
of PairPotential and AEVComputer.
"""
import typing as tp
import math

import torch
from torch import Tensor


# All cutoffs assume the elements in "distances" are smaller than "cutoff"
# all parameters of a Cutoff **must be passed to init of the superclass**
# If cuaev supports the cutoff _cuaev_name must be defined to be a unique string
class Cutoff(torch.nn.Module):
    _cuaev_name: str

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        super().__init__()
        self.fn_params = args + tuple(kwargs.values())
        self._cuaev_name = ""

    @torch.jit.unused
    def is_same(self, other: object) -> bool:
        # This is a hack since __eq__ can't be safely overriden in torch.nn.Module
        if not isinstance(other, Cutoff):
            return False
        if type(self) is not type(other):
            return False
        if not self.fn_params == other.fn_params:
            return False
        return True

    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        raise NotImplementedError


class CutoffDummy(Cutoff):
    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        return torch.ones_like(distances)


class CutoffCosine(Cutoff):
    def __init__(self) -> None:
        super().__init__()
        self._cuaev_name = "cosine"

    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        return 0.5 * torch.cos(distances * (math.pi / cutoff)) + 0.5


class CutoffSmooth(Cutoff):
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
        return f"order={self.order}, eps={self.eps:.1e}"


CutoffArg = tp.Union[
    tp.Literal[
        "global",
        "dummy",
        "cosine",
        "smooth",
        "smooth2",
        "smooth4",
    ],
    Cutoff,
]


def parse_cutoff_fn(
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
    elif cutoff_fn in ("smooth", "smooth2"):
        cutoff_fn = CutoffSmooth(order=2)
    elif cutoff_fn == "smooth4":
        cutoff_fn = CutoffSmooth(order=4)
    elif not isinstance(cutoff_fn, Cutoff):
        raise ValueError(f"Unsupported cutoff fn: {cutoff_fn}")
    return tp.cast(Cutoff, cutoff_fn)
