r"""The TorchANI neural network potential library main namespace

Most of the functions and classes of the library are accessible from their specific
modules. For convenience, some useful classes are accessible *also* from here. These
are:

- `torchani.SpeciesConverter <torchani.nn.SpeciesConverter>`
- `torchani.AEVComputer <torchani.aev.AEVComputer>`
- `torchani.ANINetworks <torchani.nn.ANINetworks>`
- `torchani.SelfEnergy <torchani.sae.SelfEnergy>`
"""

import importlib
from importlib.metadata import version, PackageNotFoundError

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
    "ase",
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
    "SelfEnergy",
    "single_point",
]

# modules that are lazily imported
_lazy_modules = {
    "nn",
    "aev",
    "arch",
    "ase",
    "utils",
    "models",
    "units",
    "datasets",
    "legacy_data",
    "transforms",
    "cli",
    "electro",
    "neighbors",
    "cutoffs",
    "sae",
    "sae_estimation",
    "constants",
    "grad",
    "io",
    "neurochem",
    "annotations",
    "paths",
    "potentials",
}

_lazy_attributes = {
    "EnergyShifter": "utils",
    "ANIModel": "nn",
    "AEVComputer": "aev",
    "ANINetworks": "nn",
    "Ensemble": "nn",
    "SpeciesConverter": "nn",
    "SelfEnergy": "sae",
    "single_point": "grad",
}


def __getattr__(name):
    if name in _lazy_modules:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _lazy_attributes:
        module_name = _lazy_attributes[name]
        module = importlib.import_module(f".{module_name}", __name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:
    __version__ = version("torchani")
except PackageNotFoundError:
    pass  # package is not installed
