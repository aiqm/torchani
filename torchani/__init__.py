r"""The TorchANI neural network potential library main namespace

Most of the functions and classes of the library are accessible from their specific
modules. For convenience, some useful classes are accessible *also* from here. These
are:

- `torchani.SpeciesConverter <torchani.nn.SpeciesConverter>`
- `torchani.AEVComputer <torchani.aev.AEVComputer>`
- `torchani.ANINetworks <torchani.nn.ANINetworks>`
- `torchani.SelfEnergy <torchani.sae.SelfEnergy>`
"""

from importlib.metadata import version, PackageNotFoundError

from torchani import (
    nn,
    aev,
    arch,
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
    sae_estimation,
    constants,
    grad,
    io,
    neurochem,
    annotations,
    paths,
)

# Legacy API, don't document
from torchani.utils import EnergyShifter
from torchani.nn import ANIModel

# Dump into global namespace for convenience
from torchani.aev import AEVComputer
from torchani.nn import ANINetworks, Ensemble, SpeciesConverter
from torchani.sae import SelfEnergy
from torchani.grad import single_point

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
    "arch",
    "constants",
    "utils",
    "units",
    "sae",
    "sae_estimation",
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
    "Ensemble",
    "SelfEnergy",
    "single_point",
]

# Optional submodule, depends on ase being available
try:
    from torchani import ase  # noqa: F401

    __all__.insert(19, "ase")  # Insert in a nice location for docs
    ASE_IS_AVAILABLE = True
except ImportError:
    ASE_IS_AVAILABLE = False
