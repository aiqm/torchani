r"""Locations used by TorchANI to cache various resources"""

from enum import Enum
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


def custom_models_dir() -> Path:
    r"""Directory where custom models are stored"""
    dir = data_dir() / "Models"
    dir.mkdir(exist_ok=True, parents=True)
    return dir


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


def batched_data_dir() -> Path:
    r"""Directory where pre-batched datasets are stored"""
    dir = data_dir() / "Batched"
    dir.mkdir(exist_ok=True, parents=True)
    return dir


def train_dir() -> Path:
    r"""Directory where training runs are stored"""
    dir = data_dir() / "Train"
    dir.mkdir(exist_ok=True, parents=True)
    return dir


def ftune_dir() -> Path:
    r"""Directory where finetuning runs are stored"""
    dir = data_dir() / "Train"
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


class DisambiguationError(RuntimeError):
    pass


class DataKind(Enum):
    TRAIN = "train"
    FTUNE = "ftune"
    BATCH = "batch"
    MODELS = "models"


def select_subdirs(
    names_or_idxs: tp.Iterable[str],
    kind: DataKind = DataKind.TRAIN,
) -> tp.List[Path]:
    root = {
        DataKind.TRAIN: train_dir(),
        DataKind.FTUNE: ftune_dir(),
        DataKind.BATCH: batched_data_dir(),
        DataKind.MODELS: custom_models_dir(),
    }[kind]

    sorted_paths = sorted(root.iterdir())
    paths_len = len(sorted_paths)
    selected_paths = []
    for name_or_idx in names_or_idxs:
        if name_or_idx.isdigit():
            idx = int(name_or_idx)
            if idx > paths_len or idx < 0:
                raise RuntimeError(f"Index {idx} invalid")
            selected_paths.append(sorted_paths[idx])
        else:
            paths = [p for p in sorted_paths if p.name.startswith(name_or_idx)]
            if not paths:
                raise RuntimeError(
                    f"No paths starting with name {name_or_idx} found."
                    f"Present paths are: {sorted_paths}"
                ) from None
            elif len(paths) > 1:
                raise DisambiguationError(
                    f"More than one path starts with {name_or_idx}: {paths}"
                ) from None
            else:
                selected_paths.append(paths[0])
    return selected_paths
