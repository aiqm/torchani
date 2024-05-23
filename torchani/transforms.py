r"""
Transforms to be applied to properties when training or batching a dataset. The
usage is the same as transforms in torchvision.

Example::

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
        '/path/to/database/',
        transform=transform,
        split='training',
    )
    validation = ANIBatchedDataset(
        '/path/to/database/',
        transform=transform,
        split='validation',
    )
"""
import typing as tp

import torch
from torch import Tensor

from torchani.grad import energies_and_forces
from torchani.utils import ATOMIC_NUMBERS
from torchani.nn import SpeciesConverter
from torchani.potentials import (
    PotentialWrapper,
    StandaloneEnergyAdder,
    StandaloneTwoBodyDispersionD3,
    StandaloneRepulsionXTB,
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
    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        super().__init__()
        self.atomic_numbers = None

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        return properties


class SubtractEnergyAndForce(Transform):
    r"""
    Subtract the energies (and optionally forces) from an arbitrary Wrapper
    module. This can be coupled with, e.g., an arbitrary pairwise potential.
    """

    def __init__(self, wrapper: PotentialWrapper, subtract_force: bool = True):
        super().__init__()
        if not wrapper.periodic_table_index:
            raise ValueError("Wrapper module should have periodic_table_index=True")
        self.wrapper = wrapper
        self.atomic_numbers = wrapper.potential.atomic_numbers
        self.subtract_force = subtract_force

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        species = properties["species"]
        coordinates = properties["coordinates"]
        if self.subtract_force:
            if torch.jit.is_scripting():
                raise RuntimeError(
                    "It is not possible to JIT compile transforms that calculate forces"
                )
            energies, forces = energies_and_forces(self.wrapper, species, coordinates)
            properties["forces"] -= forces
            properties["energies"] -= energies
        else:
            properties["energies"] -= self.wrapper((species, coordinates)).energies
        return properties


class SubtractRepulsionXTB(Transform):
    r"""
    Convenience class that subtracts repulsion terms.

    Takes same arguments as :class:``torchani.potentials.StandaloneRepulsionXTB``
    """

    def __init__(
        self,
        *args,
        subtract_force: bool = True,
        **kwargs,
    ):
        super().__init__()
        self._transform = SubtractEnergyAndForce(
            StandaloneRepulsionXTB(*args, **kwargs), subtract_force=subtract_force
        )
        self.atomic_numbers = self._transform.atomic_numbers

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        return self._transform(properties)


class SubtractTwoBodyDispersionD3(Transform):
    r"""
    Convenience class that subtracts dispersion terms.

    Takes same arguments as :class:``torchani.potentials.StandaloneTwoBodyDispersionD3``
    """

    def __init__(
        self,
        *args,
        subtract_force: bool = True,
        **kwargs,
    ):
        super().__init__()
        self._transform = SubtractEnergyAndForce(
            StandaloneTwoBodyDispersionD3(*args, **kwargs),
            subtract_force=subtract_force,
        )
        self.atomic_numbers = self._transform.atomic_numbers

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        return self._transform(properties)


class SubtractSAE(Transform):
    r"""
    Convenience class that subtracts self atomic energies.

    Takes same arguments as :class:``torchani.potentials.StandaloneEnergyAdder``
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._transform = SubtractEnergyAndForce(
            StandaloneEnergyAdder(*args, **kwargs), subtract_force=False
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
            [ATOMIC_NUMBERS[s] for s in symbols],
            dtype=torch.long,
        )
        self.converter = SpeciesConverter(symbols)

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        species = self.converter(
            (properties["species"], properties["coordinates"]),
        ).species
        properties["species"] = species
        return properties


# The Compose code is mostly copied from torchvision, but made JIT scriptable
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
