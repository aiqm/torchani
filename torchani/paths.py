r"""
Default location for various TorchANI resources
"""
from pathlib import Path

LOCAL_DIR = (Path.home() / ".local") / "torchani"
STATE_DICTS = LOCAL_DIR / "StateDicts"
DATASETS = LOCAL_DIR / "Datasets"
NEUROCHEM = LOCAL_DIR / "Neurochem"

LOCAL_DIR.mkdir(exist_ok=True, parents=True)
STATE_DICTS.mkdir(exist_ok=True, parents=True)
DATASETS.mkdir(exist_ok=True, parents=True)
NEUROCHEM.mkdir(exist_ok=True, parents=True)
RESOURCES = Path(__file__).resolve().parent / "resources"
