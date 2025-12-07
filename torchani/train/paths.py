r"""
Default location for various ANITune resources
"""

import typing as tp
from pathlib import Path
from enum import Enum
from torchani.paths import data_dir

# TODO fix this hack
DATA_DIR = data_dir()
MODELS_PATH = DATA_DIR / "Models"
BATCH_PATH = DATA_DIR / "Batched"
TRAIN_PATH = DATA_DIR / "Train"
FTUNE_PATH = DATA_DIR / "Ftune"

MODELS_PATH.mkdir(exist_ok=True, parents=True)
BATCH_PATH.mkdir(exist_ok=True, parents=True)
TRAIN_PATH.mkdir(exist_ok=True, parents=True)
FTUNE_PATH.mkdir(exist_ok=True, parents=True)


class DataKind(Enum):
    TRAIN = "train"
    FTUNE = "ftune"
    BATCH = "batch"
    MODELS = "models"


class DisambiguationError(RuntimeError):
    pass


def select_subdirs(
    names_or_idxs: tp.Iterable[str],
    kind: DataKind = DataKind.TRAIN,
) -> tp.List[Path]:
    root = {
        DataKind.TRAIN: TRAIN_PATH,
        DataKind.FTUNE: FTUNE_PATH,
        DataKind.BATCH: BATCH_PATH,
        DataKind.MODELS: MODELS_PATH,
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
