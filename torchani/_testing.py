r"""
This module is *internal* and considered an implementation detail. Functions and classes
are subject to change at any point.
"""
import typing as tp
import unittest
from itertools import product

import torch
from parameterized import parameterized_class
from torch.testing._internal.common_utils import TestCase, make_tensor

from torchani.annotations import Device


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


__all__ = ["make_tensor", "TestCase", "ANITestCase", "expand"]
