r"""This module is *internal* and considered an implementation detail.

Functions and classes here are subject to change at any point.
"""

import typing as tp
import unittest
from torch import Tensor
from itertools import product

import torch
from parameterized import parameterized_class
from torch.testing._internal.common_utils import TestCase, make_tensor

from torchani.constants import ATOMIC_NUMBER
from torchani.annotations import Device, DType
from torchani.neighbors import adaptive_list, Neighbors


def _get_cls_name(cls: type, idx: int, params: tp.Dict[str, tp.Any]) -> str:
    return f"{cls.__name__}_{params['_device'].type}{'_jit' if params['_jit'] else ''}"


# As an exception, device = None means "both" here
def expand(
    device: tp.Literal["cpu", "cuda", "both"] = "both",
    jit: tp.Optional[bool] = None,
):
    devices: tp.Tuple[torch.device, ...]
    if device == "both":
        devices = (torch.device("cpu"), torch.device("cuda"))
    else:
        devices = (torch.device(device),)
    _jit = (False, True) if jit is None else (jit,)
    decorator = parameterized_class(
        ("_device", "_jit"),
        product(devices, _jit),
        class_name_func=_get_cls_name,
    )
    return decorator


# If you want a class to run in all possible contexts (all combinations of
# cuda, cpu, jit and nojit) then use @expand()

# If you want to, for example, only test in a cuda device then use
# @expand(device='cuda')

# Note that CUDA tests are automatically skipped if torch.cuda.is_available()
# is False

# By default self.device is CPU, and self.jit is False


_T = tp.TypeVar("_T", bound=torch.nn.Module)


class ANITestCase(TestCase):
    _device: torch.device
    _jit: bool

    @property
    def device(self) -> torch.device:
        return getattr(self, "_device", torch.device("cpu"))

    @property
    def jit(self) -> bool:
        return getattr(self, "_jit", False)

    @classmethod
    def setUpClass(cls):
        if (
            getattr(cls, "_device", torch.device("cpu")).type == "cuda"
        ) and not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    # jit-scripting should for the most part be transparent to users, so we
    # assume the input and output types of this function are actually the same
    def _setup(self, model: _T) -> _T:
        model.to(self.device)
        if self.jit:
            return tp.cast(_T, torch.jit.script(model))
        return model


# A group of molecules
class Molecs(tp.NamedTuple):
    coords: Tensor
    atomic_nums: Tensor
    cell: tp.Optional[Tensor]
    pbc: tp.Optional[Tensor]


def make_molecs(
    molecs_num: int,
    atoms_num: int,
    cell_size: float = 10.0,
    pbc: bool = False,
    symbols: tp.Sequence[str] = ("H", "C", "N", "O"),
    seed: tp.Optional[int] = None,
    device: Device = None,
    dtype: DType = None,
) -> Molecs:
    # CUDA rng is strange and maybe non deterministic even with seeds?
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed) if seed is not None else rng.seed()
    if seed is not None:
        torch.manual_seed(seed)
    coords = torch.rand(
        (molecs_num, atoms_num, 3),
        generator=rng,
        device="cpu", dtype=dtype,
    ) * cell_size + 1.0e-3
    idxs = torch.randint(
        low=0,
        high=len(symbols),
        size=(molecs_num * atoms_num,),
        generator=rng,
        device="cpu",
        dtype=torch.long,
    )
    idxs = idxs.to(device)
    coords = coords.to(device)
    atomic_num_kinds = torch.tensor(
        list(map(ATOMIC_NUMBER.get, symbols)), device=device, dtype=torch.long
    )
    atomic_nums = atomic_num_kinds[idxs].view(molecs_num, atoms_num)
    if pbc:
        _pbc = torch.tensor([True, True, True], device=device, dtype=torch.bool)
        _cell = torch.eye(3, device=device, dtype=dtype) * (cell_size + 2.0e-3)
    else:
        _pbc = None
        _cell = None
    return Molecs(coords, atomic_nums, _cell, _pbc)


def make_molec(
    atoms: int,
    cell_size: float = 10.0,
    pbc: bool = False,
    symbols: tp.Sequence[str] = ("H", "C", "N", "O"),
    seed: tp.Optional[int] = None,
    device: Device = None,
    dtype: DType = None,
) -> Molecs:
    return make_molecs(1, atoms, cell_size, pbc, symbols, seed, device, dtype)


def make_neighbors(
    atoms: int,
    cutoff: float = 5.2,
    symbols: tp.Sequence[str] = ("H", "C", "N", "O"),
    seed: tp.Optional[int] = None,
    device: Device = None,
    dtype: DType = None,
) -> Neighbors:
    molec = make_molec(atoms, 10.0, False, symbols, seed, device, dtype)
    return adaptive_list(cutoff, molec.atomic_nums, molec.coords)


__all__ = ["make_tensor", "TestCase", "ANITestCase", "expand"]
