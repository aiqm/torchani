r"""Locations used by TorchANI to cache various resources"""

import typing as tp
import os
from pathlib import Path
from torchani.annotations import StrPath

_RESOURCES = Path(__file__).resolve().parent / "resources"


def set_data_dir(data_dir: tp.Optional[StrPath] = None) -> None:
    r"""Manually set the root location of resources"""
    if data_dir is None:
        os.environ["TORCHANI_DATA_DIR"] = ""
    else:
        os.environ["TORCHANI_DATA_DIR"] = str(data_dir)


def datasets_dir() -> Path:
    r"""Directory where datasets are stored"""
    dir = data_dir() / "Datasets"
    dir.mkdir(exist_ok=True, parents=True)
    return dir


def neurochem_dir() -> Path:
    r"""Directory where neurochem files"""
    dir = data_dir() / "Neurochem"
    dir.mkdir(exist_ok=True, parents=True)
    return dir


def state_dicts_dir() -> Path:
    r"""Directory where the state-dicts of built-in models are stored"""
    dir = data_dir() / "StateDicts"
    dir.mkdir(exist_ok=True, parents=True)
    return dir


def data_dir() -> Path:
    r"""Root location for resources"""
    ENV_DATA_DIR = os.getenv("TORCHANI_DATA_DIR")
    if ENV_DATA_DIR:
        return Path(ENV_DATA_DIR)
    return Path(Path.home(), ".local", "share", "Torchani")


def _resources_dir() -> Path:
    return _RESOURCES
