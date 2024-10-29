r"""The TorchANI neural network potential library main namespace

Most of the functions and classes of the library are accessible from their specific
modules. For convenience, some useful classes are accessible *also* from here. These
are:

- `ANI`
- `SpeciesConverter`
- `AEVComputer`
- `ANINetworks`
- `EnergyAdder`
"""

import os
import warnings
from importlib.metadata import version, PackageNotFoundError

import torch

from torchani import (
    nn,
    aev,
    assembly,
    utils,
    models,
    units,
    datasets,
    legacy_data,
    transforms,
    cli,
    electro,
    neighbors,
    cutoffs,
    sae,
    constants,
    grad,
    io,
    neurochem,
    annotations,
    paths,
)

# Legacy API, don't document
from torchani.utils import EnergyShifter
from torchani.nn import ANIModel, Ensemble

# Dump into global namespace for convenience
from torchani.aev import AEVComputer
from torchani.nn import ANINetworks, ANIEnsemble, SpeciesConverter
from torchani.potentials import EnergyAdder

# NOTE: ase is an optional dependency so don't import here

try:
    __version__ = version("torchani")
except PackageNotFoundError:
    pass  # package is not installed

__all__ = [
    "neighbors",
    "aev",
    "nn",
    "grad",
    "models",
    "potentials",
    "cutoffs",
    "datasets",
    "transforms",
    "io",
    "electro",
    "assembly",
    "constants",
    "utils",
    "units",
    "sae",
    "cli",
    "paths",
    "annotations",
    # Legacy API
    "neurochem",
    "legacy_data",
    "EnergyShifter",
    "Ensemble",
    "ANIModel",
    # In global namespace for convenience
    "SpeciesConverter",
    "AEVComputer",
    "ANINetworks",
    "ANIEnsemble",
    "EnergyAdder",
]

# Disable TF32 since it catastrophically degrades accuracy of ANI models
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Warn about disabling TF3 only if an Ampere (or newer) GPU is detected
# (suppressed by setting TORCHANI_NO_WARN_TF32)
if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    max_sm_major = max(
        [torch.cuda.get_device_capability(i)[0] for i in range(num_devices)]
    )
    if (max_sm_major >= 8) and os.getenv("TORCHANI_NO_WARN_TF32") != "1":
        warnings.warn(
            "TorchANI disables TF32 (supported by your GPU) to prevent accuracy loss."
            " To suppress warn set the env var TORCHANI_NO_WARN_TF32=1"
            " For example, if using bash,"
            " you may add 'export TORCHANI_NO_WARN_TF32=1' to your .bashrc"
        )

# Optional submodule, depends on ase being available
try:
    from torchani import ase  # noqa: F401

    __all__.append("ase")
    ASE_IS_AVAILABLE = True
except ImportError:
    ASE_IS_AVAILABLE = False
