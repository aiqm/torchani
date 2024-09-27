r"""
`TorchANI`_ is a PyTorch library for training, development and research of
`ANI`_ style neural networks, maintained by the `Roitberg group`_.  TorchANI
contains classes like :class:`AEVComputer`, :class:`ANIModel`, and
:class:`EnergyShifter` that can be pipelined to compute molecular energies from
the 3D coordinates of molecules.  It also include tools to: deal with ANI
datasets (e.g. `ANI-1`_, `ANI-1x`_, `ANI-1ccx`_, `ANI-2x`_) at
:attr:`torchani.datasets`, import various file formats of NeuroChem at

In order to use TorchANI

.. _TorchANI:
    https://doi.org/10.26434/chemrxiv.12218294.v1

.. _ANI:
    http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract

.. _Roitberg group:
    https://roitberg.chem.ufl.edu/

.. _ANI-1:
    https://www.nature.com/articles/sdata2017193

.. _ANI-1x:
    https://aip.scitation.org/doi/abs/10.1063/1.5023802

.. _ANI-1ccx:
    https://doi.org/10.26434/chemrxiv.6744440.v1

.. _ANI-2x:
    https://doi.org/10.26434/chemrxiv.11819268.v1
"""
import os
import warnings
from importlib.metadata import version, PackageNotFoundError

import torch

from torchani.utils import EnergyShifter
from torchani.nn import ANIModel, Ensemble, SpeciesConverter
from torchani.aev import AEVComputer
from torchani import (
    assembler,
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
    infer,
    constants,
    grad,
    io,
    neurochem,
    annotations,
    paths,
)
# NOTE: ase is an optional dependency so don't import here

try:
    __version__ = version("torchani")
except PackageNotFoundError:
    pass  # package is not installed

__all__ = [
    'AEVComputer',
    'EnergyShifter',
    'ANIModel',
    'Ensemble',
    'grad',
    'SpeciesConverter',
    'utils',
    'annotations',
    'paths',
    'models',
    'units',
    'potentials',
    'neighbors',
    'cutoffs',
    'datasets',
    'legacy_data',
    'transforms',
    'cli',
    'io',
    'electro',
    'assembler',
    "sae",
    "infer",
    "constants",
    "neurochem",
]

# Disable TF32 since it catastrophically degrades accuracy
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Warn about disabling TF3 only if an Ampere (or newer) GPU is detected
# (suppressed by setting TORCHANI_NO_WARN_TF32)
if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    max_sm_major = max(
        [torch.cuda.get_device_capability(i)[0] for i in range(num_devices)]
    )
    if (max_sm_major >= 8) and ("TORCHANI_NO_WARN_TF32" not in os.environ):
        warnings.warn(
            "Torchani disables TF32 (supported by your GPU) to prevent accuracy loss."
            " To suppress warning set the env var TORCHANI_NO_WARN_TF32 to any value"
        )

# Optional submodule, depends on 'ase' being available
try:
    from . import ase  # noqa: F401
    __all__.append('ase')
    ASE_IS_AVAILABLE = True
except ImportError:
    ASE_IS_AVAILABLE = False
