import torch
import sys
sys.path.append("/home/jolmos/programas/torchani_sandbox/")
from pathlib import Path
from torchani.datasets._utils import _fetch_and_create_builtin_dataset
from torchani.paths import datasets_dir

#descargar el ds

_fetch_and_create_builtin_dataset(
        root=Path("/home/jolmos/ani_training/datasets"),
        ds_name = "ANI2x",
        lot = "wb97x-def2tzvpp",
        verbose = True,
        download=True,
        dummy_properties = None,
        skip_check=False,
        suffix = ".h5")


#ds = torchani.datasets.ANI2x(lot="wb97x-def2tzvpp")
