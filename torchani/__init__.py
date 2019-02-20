# -*- coding: utf-8 -*-
"""TorchANI is a PyTorch implementation of `ANI`_, created and maintained by
the `Roitberg group`_.  TorchANI contains classes like
:class:`AEVComputer`, :class:`ANIModel`, and :class:`EnergyShifter` that can
be pipelined to compute molecular energies from the 3D coordinates of
molecules.  It also include tools to: deal with ANI datasets(e.g. `ANI-1`_,
`ANI-1x`_, `ANI-1ccx`_, etc.) at :attr:`torchani.data`, import various file
formats of NeuroChem at :attr:`torchani.neurochem`, help working with ignite
at :attr:`torchani.ignite`, and more at :attr:`torchani.utils`.

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
"""

from .utils import EnergyShifter
from .nn import ANIModel, Ensemble
from .aev import AEVComputer
from . import ignite
from . import utils
from . import neurochem
from . import data
from . import models
from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

__all__ = ['AEVComputer', 'EnergyShifter', 'ANIModel', 'Ensemble',
           'ignite', 'utils', 'neurochem', 'data', 'models']

try:
    from . import ase  # noqa: F401
    __all__.append('ase')
except ImportError:
    pass
