r"""
Default location for various TorchANI resources
"""

import typing as tp
import os
from pathlib import Path
from torchani.annotations import StrPath

_RESOURCES = Path(__file__).resolve().parent / "resources"


def set_data_dir(data_dir: tp.Optional[StrPath] = None) -> None:
    if data_dir is None:
        os.environ["TORCHANI_DATA_DIR"] = ""
    else:
        os.environ["TORCHANI_DATA_DIR"] = str(data_dir)


def datasets_dir() -> Path:
    dir = data_dir() / "Datasets"
    dir.mkdir(exist_ok=True, parents=True)
    return dir


def neurochem_dir() -> Path:
    dir = data_dir() / "Neurochem"
    dir.mkdir(exist_ok=True, parents=True)
    return dir


def state_dicts_dir() -> Path:
    dir = data_dir() / "StateDicts"
    dir.mkdir(exist_ok=True, parents=True)
    return dir


def data_dir() -> Path:
    ENV_DATA_DIR = os.getenv("TORCHANI_DATA_DIR")
    if ENV_DATA_DIR:
        return Path(ENV_DATA_DIR)
    return Path(Path.home(), ".local", "share", "torchani")


def resources_dir() -> Path:
    return _RESOURCES
