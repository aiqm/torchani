r"""
Transforms are composable functions that can be  applied to properties when
training or batching a dataset. They are modelled after ``torchvision``
transforms. An example of their usage:

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

    transform = Compose([sae, rep, disp])

    # Transforms will be applied automatically when iterating over the datasets
    training = ANIBatchedDataset(
        '/path/to/batched-dataset/',
        transform=transform,
        split='training',
    )
    validation = ANIBatchedDataset(
        '/path/to/batched-dataset/',
        transform=transform,
        split='validation',
    )
"""

import typing as tp

import torch
from torch import Tensor

from torchani.grad import energies_and_forces
from torchani.nn import SpeciesConverter
from torchani.constants import ATOMIC_NUMBER
from torchani.potentials import (
    Potential,
    EnergyAdder,
    RepulsionXTB,
    TwoBodyDispersionD3,
)


class Transform(torch.nn.Module):
    r"""
    Base class for callables that modify conformer properties on the fly

    If the callable supports only a limited number of atomic numbers (in a
    given order) then the atomic_numbers tensor should be defined, otherwise it
    should be None
    """

    atomic_numbers: tp.Optional[Tensor]

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        super().__init__()

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
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
    r"""
    Subtract the energies (and optionally forces) from an arbitrary Potential
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
            properties["energies"] -= self.potential.calc(species, coordinates)
        return properties


class SubtractRepulsionXTB(Transform):
    r"""
    Convenience class that subtracts repulsion terms.

    Takes same arguments as :class:``torchani.potentials.RepulsionXTB``
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
    r"""
    Convenience class that subtracts dispersion terms.

    Takes same arguments as ``torchani.potentials.TwoBodyDispersionD3.from_functional``
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
    r"""
    Convenience class that subtracts self atomic energies.

    Takes same arguments as :class:``torchani.potentials.EnergyAdder``
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._transform = SubtractEnergyAndForce(
            EnergyAdder(*args, **kwargs), subtract_force=False
        )
        self.atomic_numbers = self._transform.atomic_numbers

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        return self._transform(properties)


class AtomicNumbersToIndices(Transform):
    r"""
    Converts atomic numbers to arbitrary indices

    Provided for legacy support, if added to a transform pipeline, it should in
    general be the *last* transform.
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
    r"""Composes several ``Transform`` together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
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
