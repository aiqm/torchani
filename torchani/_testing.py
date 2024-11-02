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
from torchani.annotations import Device
from torchani.neighbors import all_pairs, Neighbors


def _get_cls_name(cls: type, idx: int, params: tp.Dict[str, tp.Any]) -> str:
    return f"{cls.__name__}_{params['_device'].type}{'_jit' if params['_jit'] else ''}"


def expand(
    device: tp.Optional[Device] = None,
    jit: tp.Optional[bool] = None,
):
    _device: tp.Tuple[torch.device, ...]
    if device is None:
        _device = (torch.device("cpu"), torch.device("cuda"))
    else:
        _device = (torch.device(device),)
    _jit = (False, True) if jit is None else (jit,)
    decorator = parameterized_class(
        ("_device", "_jit"),
        product(_device, _jit),
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
        model = model.to(self.device)
        if self.jit:
            return tp.cast(_T, torch.jit.script(model))
        return model


# A group of molecules
class Molecs(tp.NamedTuple):
    coords: Tensor
    atomic_nums: Tensor
    cell: Tensor
    pbc: Tensor


def make_molecs(
    molecs_num: int,
    atoms_num: int,
    cell_size: float = 10.0,
    pbc: bool = False,
    symbols: tp.Sequence[str] = ("H", "C", "N", "O"),
    seed: tp.Optional[int] = None,
    dtype: torch.dtype = torch.float,
    device: tp.Literal["cpu", "cuda"] = "cpu",
) -> Molecs:
    rng = torch.Generator(device=device)
    rng.manual_seed(seed) if seed is not None else rng.seed()
    coords = torch.rand(
        (molecs_num, atoms_num, 3), generator=rng, device=device, dtype=dtype
    )
    idxs = torch.randint(
        low=0,
        high=len(symbols),
        size=(molecs_num * atoms_num,),
        generator=rng,
        device=device,
        dtype=torch.long,
    )
    atomic_num_kinds = torch.tensor(
        list(map(ATOMIC_NUMBER.get, symbols)), device=device, dtype=torch.long
    )
    atomic_nums = atomic_num_kinds[idxs].view(molecs_num, atoms_num)
    cell = torch.eye(3, device=device, dtype=dtype) * cell_size
    _pbc = torch.full((3,), fill_value=int(pbc), device=device, dtype=torch.bool)
    return Molecs(coords, atomic_nums, cell, _pbc)


def make_molec(
    atoms: int,
    cell_size: float = 10.0,
    pbc: bool = False,
    symbols: tp.Sequence[str] = ("H", "C", "N", "O"),
    seed: tp.Optional[int] = None,
    dtype: torch.dtype = torch.float,
    device: tp.Literal["cpu", "cuda"] = "cpu",
) -> Molecs:
    return make_molecs(1, atoms, cell_size, pbc, symbols, seed, dtype, device)


def make_neighbors(
    atoms: int,
    cutoff: float = 5.2,
    symbols: tp.Sequence[str] = ("H", "C", "N", "O"),
    seed: tp.Optional[int] = None,
    dtype: torch.dtype = torch.float,
    device: tp.Literal["cpu", "cuda"] = "cpu",
) -> Neighbors:
    molec = make_molec(atoms, 10.0, False, symbols, seed, dtype, device)
    return all_pairs(molec.atomic_nums, molec.coords, cutoff)


__all__ = ["make_tensor", "TestCase", "ANITestCase", "expand"]
