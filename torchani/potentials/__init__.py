r"""
Potentials are callable objects that can be attached to TorchANI models. They
calculate explicit analytical energy expressions that modify the final output
of the models, usually to correct it in some regions of chemical space where
they don't behave in a physically appropriate manner.

Potentials may be many-body potentials or pairwise potentials. Subclasses must override
'forward'.

New potentials can be added to the TorchANI library by simple subclassing. For
a tutorial on how to do this please refer to the corresponding documentation in
examples.

Many potentials correspond to functions implemented in specific articles. If
you use any of these potentials in your work, please cite the corresponding
article(s).
"""

from torchani.potentials.core import Potential, PairPotential
from torchani.potentials.nnp import (
    NNPotential,
    SeparateChargesNNPotential,
    MergedChargesNNPotential,
)
from torchani.potentials.repulsion import RepulsionXTB
from torchani.potentials.dispersion import TwoBodyDispersionD3
from torchani.potentials.shifter import EnergyAdder


__all__ = [
    "Potential",
    "PairPotential",
    "EnergyAdder",
    "RepulsionXTB",
    "TwoBodyDispersionD3",
    "NNPotential",
    "SeparateChargesNNPotential",
    "MergedChargesNNPotential",
]
