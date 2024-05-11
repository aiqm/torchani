import typing as tp
import unittest
from itertools import product

import torch
from parameterized import parameterized_class
from torch.testing._internal.common_utils import TestCase, make_tensor  # noqa: F401


def _get_cls_name(cls: type, idx: int, params: tp.Dict[str, tp.Any]) -> str:
    return f"{cls.__name__}_{params['_device']}{'_jit' if params['_jit'] else ''}"


def expand(
    device: tp.Union[tp.Literal["cpu"], tp.Literal["cuda"], None] = None,
    jit: tp.Optional[bool] = None,
):
    if device not in (None, "cpu", "cuda"):
        raise ValueError("Device must be None or one of 'cpu', 'cuda'")
    _device = ("cpu", "cuda") if device is None else (device,)
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


@expand()
class ANITest(TestCase):
    _device: str
    _jit: bool

    @property
    def device(self) -> str:
        return getattr(self, "_device", "cpu")

    @property
    def jit(self) -> bool:
        return getattr(self, "_jit", False)

    @classmethod
    def setUpClass(cls):
        if (getattr(cls, "_device", "cpu") == "cuda") and not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    # jit-scripting should for the most part be transparent to users, so we assume the
    # input and output types of this function are actually the same
    def _setup(self, model: _T) -> _T:
        model = model.to(self.device)
        if self.jit:
            return torch.jit.script(model)
        return model


__all__ = ["make_tensor", "TestCase", "ANITest", "expand"]
