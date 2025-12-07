r"""
Default values for different architectures, optimizers and lr schedulers
"""

import sys
import typing as tp
from dataclasses import dataclass, asdict
from torchani.annotations import PyScalar


@dataclass
class Options:
    def as_dict(self) -> tp.Dict[str, PyScalar]:
        return asdict(self)


@dataclass
class ReduceLROnPlateau(Options):
    factor: float = 0.5
    patience: int = 100
    threshold: float = 1e-4


@dataclass
class StepLR(Options):
    gamma: float = 0.5  # gamma
    step_size: int = 150  # step_size in epochs


@dataclass
class ExponentialLR(Options):
    gamma: float = 0.99


# max_epochs is set to scheduler.max_epochs automatically
@dataclass
class CosineAnnealingLR(Options):
    total_epochs: int = 500
    eta_min: float = 0.0  # eta_min


@dataclass
class OptimizerArgs(Options):
    lr: float = 5e-4
    weight_decay: float = 1e-6


# eps for Adamax, Adam, AdamW is not shown, and it is always 1e-8
@dataclass
class Adam(OptimizerArgs):
    amsgrad: bool = False


@dataclass
class AdamW(OptimizerArgs):
    amsgrad: bool = False


@dataclass
class RMSprop(OptimizerArgs):
    momentum: float = 0.0
    alpha: float = 0.99
    centered: bool = True


@dataclass
class Adamax(OptimizerArgs):
    pass


@dataclass
class SGD(OptimizerArgs):
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    # Has no fused impl


def resolve_options(
    _options: tp.Sequence[str] = (),
    cls: str = "",
) -> tp.Dict[str, PyScalar]:
    r"""
    Take a sequence of 'key=val' strings and convert it into a dict of python
    objects.

    This function relies on ``eval`` which makes it unsafe
    """
    try:
        default_options: tp.Dict[str, PyScalar] = getattr(
            sys.modules[__name__], cls
        )().as_dict()
    except AttributeError:
        default_options = {}

    options: tp.Dict[str, PyScalar] = dict()

    for kv in _options:
        if "=" not in kv:
            raise RuntimeError(f"Bad format for option {kv}, should be key=value")

        k, _v = kv.split("=")
        try:
            v = eval(_v.replace("true", "True").replace("false", "False"))
        except Exception:
            # Assume string
            v = _v
        options.update({k: v})
    default_options.update(options)
    return default_options


def parse_scheduler_str(scheduler: str) -> str:
    scheduler = scheduler.capitalize()
    if "LR" not in scheduler:
        scheduler = "".join((scheduler, "LR"))
    if scheduler == "PlateauLR":
        scheduler = "ReduceLROnPlateau"
    elif scheduler in ["CosineLR", "CosLR"]:
        scheduler = "CosineAnnealingLR"
    elif scheduler in ["ExponentialLR", "ExpLR"]:
        scheduler = "ExponentialLR"
    elif scheduler == "StepLR":
        pass
    else:
        raise ValueError(f"Unsupported scheduler {scheduler}")
    return scheduler


def parse_optimizer_str(optimizer: str) -> str:
    return {"rmsprop": "RMSprop", "sgd": "SGD", "adamw": "AdamW"}.get(
        optimizer.lower(), optimizer.capitalize()
    )
