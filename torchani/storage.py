r"""
Default storage location for various torchani resources
"""
from pathlib import Path

LOCAL_DIR = (Path.home() / ".local") / "torchani"
STATE_DICTS_DIR = LOCAL_DIR / "StateDicts"
DATASETS_DIR = LOCAL_DIR / "Datasets"
NEUROCHEM_DIR = LOCAL_DIR / "Neurochem"

LOCAL_DIR.mkdir(exist_ok=True, parents=True)
STATE_DICTS_DIR.mkdir(exist_ok=True, parents=True)
DATASETS_DIR.mkdir(exist_ok=True, parents=True)
NEUROCHEM_DIR.mkdir(exist_ok=True, parents=True)
RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
