import torch
import sys
sys.path.append("/home/jolmos/programas/torchani_sandbox/")
from pathlib import Path
import torchani

#descargar el ds

batcher = torchani.datasets.batching.Batcher(
        dest_root = Path("/home/jolmos/ani_training/datasets/ani2x_batched/"),
        )

batcher.divide_and_batch(
    src=[Path("datasets/ANI2x-wb97x-def2tzvpp/ANI-1x-wb97x-def2tzvpp.h5"),
        Path("datasets/ANI2x-wb97x-def2tzvpp/ANI-2x_subset-wb97x-def2tzvpp.h5")],
    dest_dir = "",
    splits={"training":0.8,"validation":0.2},
    )

