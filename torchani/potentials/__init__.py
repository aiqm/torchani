r"""Callable objects that can be attached to TorchANI models.

Potentials calculate explicit analytical energy expressions that modify the final output
of the models, usually to correct it in some regions of chemical space where they don't
behave in a physically appropriate manner.

New potentials can be added to the TorchANI library by simple subclassing. For a
tutorial on how to do this please refer to the corresponding documentation in examples.

``Potential`` is a torch ``Module``. It can represent many-body potentials.
``PairPotential`` is a more specific subclass, which represents pair potentials.
Subclasses of ``Potential`` should override ``compute``, and subclasses of
``PairPotential`` should override ``pair_energies``.

Many potentials correspond to functions implemented in specific scientific articles or
books. If you use any of these potentials in your work, please include the corresponding
citations.
"""

from torchani.potentials.core import (
    Potential,
    BasePairPotential,
    PairPotential,
    DummyPotential,
)
from torchani.potentials.nnp import (
    NNPotential,
    SeparateChargesNNPotential,
    MergedChargesNNPotential,
)
from torchani.potentials.xtb import RepulsionXTB
from torchani.potentials.dftd3 import TwoBodyDispersionD3
from torchani.potentials.lj import DispersionLJ, RepulsionLJ, LennardJones
from torchani.potentials.zbl import RepulsionZBL
from torchani.potentials.fixed_coulomb import FixedCoulomb, FixedMNOK


__all__ = [
    "Potential",
    "DummyPotential",
    "BasePairPotential",
    "PairPotential",
    "RepulsionXTB",
    "TwoBodyDispersionD3",
    "NNPotential",
    "SeparateChargesNNPotential",
    "MergedChargesNNPotential",
    "RepulsionZBL",
    "RepulsionLJ",
    "DispersionLJ",
    "LennardJones",
    "FixedCoulomb",
    "FixedMNOK",
]
