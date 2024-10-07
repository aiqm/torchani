r"""
Default location for various TorchANI resources
"""
import os
from pathlib import Path

ENV_DATA_DIR = os.getenv("TORCHANI_DATA_DIR")

if ENV_DATA_DIR:
    DATA_DIR = Path(ENV_DATA_DIR)
else:
    DATA_DIR = Path(Path.home(), ".local", "share", "torchani")

STATE_DICTS = DATA_DIR / "StateDicts"
DATASETS = DATA_DIR / "Datasets"
NEUROCHEM = DATA_DIR / "Neurochem"

DATA_DIR.mkdir(exist_ok=True, parents=True)
STATE_DICTS.mkdir(exist_ok=True, parents=True)
DATASETS.mkdir(exist_ok=True, parents=True)
NEUROCHEM.mkdir(exist_ok=True, parents=True)
RESOURCES = Path(__file__).resolve().parent / "resources"
