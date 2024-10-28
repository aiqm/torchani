r"""
Potentials are callable objects that can be attached to TorchANI models. They
calculate explicit analytical energy expressions that modify the final output
of the models, usually to correct it in some regions of chemical space where
they don't behave in a physically appropriate manner.

New potentials can be added to the TorchANI library by simple subclassing. For
a tutorial on how to do this please refer to the corresponding documentation in
examples.

``Potential`` is a torch ``Module``. It can represent many-body potentials.
``PairPotential`` is a more specific subclass, which represents pair potentials.
Subclasses of ``Potential`` should override ``forward``, and subclasses of
``PairPotential`` should override ``pair_energies``.

Many potentials correspond to functions implemented in specific scientific articles. If
you use any of these potentials in your work, please cite the corresponding article(s).
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
