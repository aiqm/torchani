r"""Classes that represent atomic (and groups of element-specific) neural networks

The most important classes in this module are `AtomicNetwork`, which represents a
callable that computes scalars from local atomic features, `ANINetworks`, and
`Ensemble`, which collect groups of element-specific neural networks and perform
different reduction operations over them.

It also contains useful factory methods to instantiate neural networks for different
elements.

Inference-optimized versions of `Ensemble` and `AtomicNetwork`, recommended for
calculations of single molecules, molecular dynamics and geometry optimizations, are
also provided.
"""

from torchani.nn._core import (
    AtomicNetwork,
    AtomicContainer,
    AtomicOneHot,
    AtomicEmbedding,
    parse_activation,
    TightCELU,
)
from torchani.nn._infer import BmmAtomicNetwork, BmmEnsemble, BmmLinear, MNPNetworks
from torchani.nn._containers import (
    ANINetworks,
    ANISharedNetworks,
    SingleNN,
    Ensemble,
    SpeciesConverter,
)
from torchani.nn._internal import Sequential, ANIModel

__all__ = [
    # Core
    "AtomicOneHot",
    "AtomicEmbedding",
    "AtomicContainer",
    "AtomicNetwork",
    "parse_activation",
    # Containers
    "ANINetworks",
    "ANISharedNetworks",
    "SingleNN",
    "Ensemble",
    "SpeciesConverter",
    # Inference optimization
    "MNPNetworks",
    "BmmLinear",
    "BmmEnsemble",
    "BmmAtomicNetwork",
    # Activation functions
    "TightCELU",
    # Legacy
    "ANIModel",
    "Sequential",
]
