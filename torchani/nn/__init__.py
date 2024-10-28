r"""
This submodule contains classes that represent atomic neural networks
(``AtomicNetwork``), and groups of element-specific neural networks (``ANINetworks``,
``ANIEnsemble``), and perform different reduction operations over them.

It also contains useful factory methods to instantiate neural networks for different
elements.

Inference-optimized versions of ``ANIEnsemble`` and ``AtomicNetwork``, recommended for
single-point calculations of single molecules, molecular dynamics and geometry
optimizations, are also provided.
"""

from torchani.nn._core import AtomicNetwork, AtomicContainer, parse_activation
from torchani.nn._factories import (
    parse_network_maker,
    AtomicMaker,
    AtomicMakerArg,
    make_1x_network,
    make_2x_network,
    make_ala_network,
    make_dr_network,
)
from torchani.nn._infer import BmmAtomicNetwork, BmmEnsemble, BmmLinear, MNPNetworks
from torchani.nn._containers import ANINetworks, ANIEnsemble, SpeciesConverter
from torchani.nn._internal import (
    ANIModel,
    Ensemble,
    Sequential,
)

__all__ = [
    # Core
    "AtomicContainer",
    "AtomicNetwork",
    "parse_activation",
    # Factories
    "AtomicMaker",
    "AtomicMakerArg",
    "parse_network_maker",
    "make_1x_network",
    "make_2x_network",
    "make_ala_network",
    "make_dr_network",
    # Containers
    "ANINetworks",
    "ANIEnsemble",
    "SpeciesConverter",
    # Inference optimization
    "MNPNetworks",
    "BmmLinear",
    "BmmEnsemble",
    "BmmAtomicNetwork",
    # Legacy
    "ANIModel",
    "Ensemble",
    "Sequential",
]
