"""`TorchANI`_ is a PyTorch implementation of `ANI`_, created and maintained by
the `Roitberg group`_.  TorchANI contains classes like
:class:`AEVComputer`, :class:`ANIModel`, and :class:`EnergyShifter` that can
be pipelined to compute molecular energies from the 3D coordinates of
molecules.  It also include tools to: deal with ANI datasets(e.g. `ANI-1`_,
`ANI-1x`_, `ANI-1ccx`_, `ANI-2x`_) at :attr:`torchani.data`, import various file
formats of NeuroChem at :attr:`torchani.neurochem`, and more at :attr:`torchani.utils`.

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
    transforms,
    cli,
    geometry,
    calc,
    neighbors,
    cutoffs,
    sae,
    infer,
    neurochem,  # TODO: Get rid of this
    data,  # TODO: Get rid of this
)

try:
    __version__ = version("torchani")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = [
    'AEVComputer',
    'EnergyShifter',
    'ANIModel',
    'Ensemble',
    'SpeciesConverter',
    'utils',
    'models',
    'units',
    'potentials',
    'neighbors',
    'cutoffs',
    'datasets',
    'transforms',
    'cli',
    'geometry',
    'calc',
    'assembler',
    "sae",
    "infer",
    'neurochem',  # TODO: Get rid of this
    'data',  # TODO: Get rid of this
]

# TF32 catastrophically degrades accuracy so we disable it
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# We warn about this only if an Ampere GPU (or newer) is detected
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

# We warn about ASE not being available since it is an optional dependency, but
# many users will probably want to enable it
# TODO: Maybe just make this a hard dependency
try:
    from . import ase  # noqa: F401
    __all__.append('ase')
except ImportError:
    if "TORCHANI_NO_WARN_ASE" not in os.environ:
        warnings.warn(
            "ASE could not be found."
            " The torchani.ase module and model's *.ase() methods won't be available\n"
            " To use them install ASE ('pip install ase' or'conda install ase')"
            " To suppress warning set the env var TORCHANI_NO_WARN_ASE to any value"
        )
