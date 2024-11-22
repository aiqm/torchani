r"""Composable functions that modify to properties when training or batching a dataset.

The API for the transforms in this module is modelled after the ``torchvision``
transforms, which you may be familiar with already. An example of their usage:

.. code-block:: python

    from torchani.transforms import (
        Compose,
        SubtractSAE,
        SubtractRepulsionXTB,
        SubtractTwoBodyDispersionD3,
    )
    from torchani.datasets import ANIBatchedDataset

    symbols = ("H", "N", "O")
    sae = SubtractSAE(symbols=symbols, (-0.5, -54.6, -75))
    rep = SubtractRepulsionXTB(symbols=symbols)
    disp = SubtractTwoBodyDispersionD3(symbols=symbols, functional='wB97X')
    t = Compose([sae, rep, disp])

    # Transforms will be applied automatically when iterating over the datasets
    train = ANIBatchedDataset('/dataset/path/', transform=t, split='training')
    valid = ANIBatchedDataset('/dataset/path/', transform=t, split='validation')
"""

import typing as tp

import torch
from torch import Tensor

from torchani.grad import energies_and_forces
from torchani.nn import SpeciesConverter
from torchani.constants import ATOMIC_NUMBER
from torchani.potentials import (
    Potential,
    RepulsionXTB,
    TwoBodyDispersionD3,
)
from torchani.sae import SelfEnergy


class Transform(torch.nn.Module):
    r"""Base class for callables that modify mappings of molecule properties

    If the callable supports only a limited number of atomic numbers (in a given order)
    then the atomic_numbers tensor should be defined, otherwise it should be None
    """

    atomic_numbers: tp.Optional[Tensor]

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        super().__init__()

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        r"""
        Transform a batch of properties

        Args:
            properties: Input properties
        Returns:
            Transformed properties
        """
        raise NotImplementedError("Must be overriden by subclasses")


class Identity(Transform):
    r"""Pass-through transform"""

    def __init__(self) -> None:
        super().__init__()
        self.atomic_numbers = None

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        return properties


identity = Identity()


class SubtractEnergyAndForce(Transform):
    r"""Subtract the energies (and optionally forces) of a potential

    Args:
        potential: The potential to use for calculating energies and forces
        subtract_force: Whether to subtract forces
    """

    def __init__(self, potential: Potential, subtract_force: bool = True):
        super().__init__()
        self.potential = potential
        self.atomic_numbers = potential.atomic_numbers
        self.subtract_force = subtract_force

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        if torch.jit.is_scripting():
            raise RuntimeError("SubtractEnergyAndForce doesn't support JIT")
        species = properties["species"]
        coordinates = properties["coordinates"]
        if self.subtract_force:
            energies, forces = energies_and_forces(self.potential, species, coordinates)
            properties["energies"] -= energies
            properties["forces"] -= forces
        else:
            properties["energies"] -= self.potential(species, coordinates)
        return properties


class SubtractRepulsionXTB(Transform):
    r"""Subtract xTB repulsion energies (and optionally forces)

    Takes same arguments as `torchani.potentials.RepulsionXTB`
    """

    def __init__(
        self,
        *args,
        subtract_force: bool = True,
        **kwargs,
    ):
        super().__init__()
        self._transform = SubtractEnergyAndForce(
            RepulsionXTB(*args, **kwargs), subtract_force=subtract_force
        )
        self.atomic_numbers = self._transform.atomic_numbers

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        return self._transform(properties)


class SubtractTwoBodyDispersionD3(Transform):
    r"""Subtract two-body DFT-D3 energies (and optionally forces)

    Takes same arguments as `torchani.potentials.TwoBodyDispersionD3`
    """

    def __init__(
        self,
        *args,
        subtract_force: bool = True,
        **kwargs,
    ):
        super().__init__()
        self._transform = SubtractEnergyAndForce(
            TwoBodyDispersionD3.from_functional(*args, **kwargs),
            subtract_force=subtract_force,
        )
        self.atomic_numbers = self._transform.atomic_numbers

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        return self._transform(properties)


class SubtractSAE(Transform):
    r"""Subtract self atomic energies.

    Takes same arguments as `torchani.sae.SelfEnergy`
    """

    def __init__(self, symbols: tp.Sequence[str], self_energies: tp.Sequence[float]):
        super().__init__()
        self._shifter = SelfEnergy(symbols, self_energies)
        self.atomic_numbers = self._shifter.atomic_numbers

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        elem_idxs = self._shifter._conv_tensor[properties["species"]]
        properties["energies"] -= self._shifter(elem_idxs)
        return properties


class AtomicNumbersToIndices(Transform):
    r"""Converts atomic numbers to indices

    Note:
        Provided for backwards compatibility, if added to a transform pipeline, it
        should in general be the *last* transform.
    Args:
        symbols: |symbols|
    """

    def __init__(self, symbols: tp.Sequence[str]):
        super().__init__()
        self.atomic_numbers = torch.tensor(
            [ATOMIC_NUMBER[s] for s in symbols], dtype=torch.long
        )
        self.converter = SpeciesConverter(symbols)

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        properties["species"] = self.converter(properties["species"])
        return properties


# Similar to torchvision.transforms.Compose, but JIT scriptable
class Compose(Transform):
    r"""Compose several `torchani.transforms.Transform` into a pipeline

    Args:
        transforms: Transforms to compose, in order.
    """

    def __init__(self, transforms: tp.Sequence[Transform]):
        super().__init__()

        # Validate that all transforms use the same atomic numbers
        atomic_numbers: tp.List[Tensor] = [
            t.atomic_numbers for t in transforms if t.atomic_numbers is not None
        ]
        if atomic_numbers:
            if not all((a == atomic_numbers[0]).all() for a in atomic_numbers):
                raise ValueError(
                    "All composed transforms must support the same atomic numbers"
                )
            self.atomic_numbers = atomic_numbers[0]
        else:
            self.atomic_numbers = None

        self.transforms = torch.nn.ModuleList(transforms)

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        for t in self.transforms:
            properties = t(properties)
        return properties

    def __repr__(self) -> str:
        parts = [self.__class__.__name__, "("]
        for t in self.transforms:
            parts.append("\n")
            parts.append(f"    {t}")
        parts.append("\n)")
        return "".join(parts)
