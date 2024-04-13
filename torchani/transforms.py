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
import math
import warnings

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from torchani.utils import ATOMIC_NUMBERS
from torchani.nn import SpeciesConverter
from torchani.datasets import ANIBatchedDataset
from torchani.wrappers import Wrapper
from torchani.potentials import (
    StandaloneEnergyAdder,
    StandaloneTwoBodyDispersionD3,
    StandaloneRepulsionXTB,
)


class Transform(torch.nn.Module):
    r"""
    Base class for callables that modify conformer properties on the fly

    If the callable supports only a limited number of atomic numbers (in a given order)
    then the atomic_numbers tensor should be defined, otherwise it should be None
    """
    atomic_numbers: tp.Optional[Tensor]

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        super().__init__()

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        raise NotImplementedError("Must be overriden by subclasses")


class SubtractEnergy(Transform):
    r"""
    Subtract the energy calculated from an arbitrary Wrapper module This
    can be coupled with, e.g., an arbitrary pairwise potential in order to
    subtract analytic energies before training.
    """
    def __init__(self, wrapper: Wrapper):
        super().__init__()
        if not wrapper.periodic_table_index:
            raise ValueError("Wrapper module should have periodic_table_index=True")
        self.wrapper = wrapper
        if hasattr(self.wrapper.module, "atomic_numbers"):
            self.atomic_numbers = tp.cast(Tensor, self.wrapper.module.atomic_numbers)
        else:
            self.atomic_numbers = None

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        properties['energies'] -= self.wrapper(
            (properties["species"], properties["coordinates"]),
        ).energies
        return properties


class SubtractForce(Transform):
    r"""
    Subtract the force calculated from an arbitrary Wrapper module. This
    can be coupled with, e.g., an arbitrary pairwise potential in order to
    subtract analytic forces before training.
    """
    def __init__(self, wrapper: Wrapper):
        super().__init__()
        if not wrapper.periodic_table_index:
            raise ValueError("Wrapper module should have periodic_table_index=True")
        self.wrapper = wrapper
        if hasattr(self.wrapper.module, "atomic_numbers"):
            self.atomic_numbers = tp.cast(Tensor, self.wrapper.module.atomic_numbers)
        else:
            self.atomic_numbers = None

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        coords = properties["coordinates"]
        coords.requires_grad_(True)
        energies = self.wrapper(
            (properties["species"], properties["coordinates"]),
        ).energies
        forces = -torch.autograd.grad(energies.sum(), coords)[0]
        coords.requires_grad_(False)
        properties["forces"] -= forces
        return properties


class SubtractRepulsionXTB(Transform):
    r"""
    Convenience class that subtracts repulsion terms.

    Takes same arguments as :class:``torchani.potentials.StandaloneRepulsionXTB``
    """
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()
        self._transform = SubtractEnergy(
            StandaloneRepulsionXTB(*args, **kwargs)
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
        **kwargs,
    ):
        super().__init__()
        self._transform = SubtractEnergy(
            StandaloneTwoBodyDispersionD3(*args, **kwargs)
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
        self._transform = SubtractEnergy(StandaloneEnergyAdder(*args, **kwargs))
        self.atomic_numbers = self._transform.atomic_numbers

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        return self._transform(properties)


class AtomicNumbersToIndices(Transform):
    r"""
    WARNING: Using this class is very error prone and not recommended

    Converts atomic numbers to arbitrary indices, provided for legacy support

    (If added to a transform pipeline, it should in general be the *last*
    transform)
    """
    def __init__(self, symbols: tp.Sequence[str]):
        super().__init__()
        warnings.warn(
            "It is not recommended convert atomic numbers to indices, this is "
            " very error prone and can generate multiple issues"
        )
        self.atomic_numbers = torch.tensor([ATOMIC_NUMBERS[s] for s in symbols], dtype=torch.long)
        self.converter = SpeciesConverter(symbols)

    def forward(self, properties: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
        species = self.converter((properties['species'], properties['coordinates'])).species
        properties['species'] = species
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
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def calculate_saes(dataset: tp.Union[DataLoader, ANIBatchedDataset],
                         elements: tp.Sequence[str],
                         mode: str = 'sgd',
                         fraction: float = 1.0,
                         fit_intercept: bool = False,
                         device: str = 'cpu',
                         max_epochs: int = 1,
                         lr: float = 0.01) -> tp.Tuple[Tensor, tp.Optional[Tensor]]:
    if mode == 'exact':
        if lr != 0.01:
            raise ValueError("lr is only used with mode=sgd")
        if max_epochs != 1:
            raise ValueError("max_epochs is only used with mode=sgd")

    assert mode in ['sgd', 'exact']
    if isinstance(dataset, DataLoader):
        assert isinstance(dataset.dataset, ANIBatchedDataset)
        old_transform = dataset.dataset.transform
        dataset.dataset.transform = AtomicNumbersToIndices(elements)
    else:
        assert isinstance(dataset, ANIBatchedDataset)
        old_transform = dataset.transform
        dataset.transform = AtomicNumbersToIndices(elements)

    num_species = len(elements)
    num_batches_to_use = math.ceil(len(dataset) * fraction)
    print(f'Using {num_batches_to_use} of a total of {len(dataset)} batches to estimate SAE')

    if mode == 'exact':
        print('Calculating SAE using exact OLS method...')
        m_out, b_out = _calculate_saes_exact(dataset, num_species, num_batches_to_use,
                                             device=device, fit_intercept=fit_intercept)
    elif mode == 'sgd':
        print("Estimating SAE using stochastic gradient descent...")
        m_out, b_out = _calculate_saes_sgd(dataset, num_species, num_batches_to_use,
                                           device=device,
                                           fit_intercept=fit_intercept,
                                           max_epochs=max_epochs, lr=lr)

    if isinstance(dataset, DataLoader):
        assert isinstance(dataset.dataset, ANIBatchedDataset)
        dataset.dataset.transform = old_transform
    else:
        assert isinstance(dataset, ANIBatchedDataset)
        dataset.transform = old_transform
    return m_out, b_out


def _calculate_saes_sgd(dataset, num_species: int, num_batches_to_use: int,
                        device: str = 'cpu',
                        fit_intercept: bool = False,
                        max_epochs: int = 1,
                        lr: float = 0.01) -> tp.Tuple[Tensor, tp.Optional[Tensor]]:

    class LinearModel(torch.nn.Module):

        m: torch.nn.Parameter
        b: tp.Optional[torch.nn.Parameter]

        def __init__(self, num_species: int, fit_intercept: bool = False):
            super().__init__()
            self.register_parameter('m', torch.nn.Parameter(torch.ones(num_species, dtype=torch.float)))
            if fit_intercept:
                self.register_parameter('b', torch.nn.Parameter(torch.zeros(1, dtype=torch.float)))
            else:
                self.register_parameter('b', None)

        def forward(self, x: Tensor) -> Tensor:
            x *= self.m
            if self.b is not None:
                x += self.b
            return x.sum(-1)

    model = LinearModel(num_species, fit_intercept).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(max_epochs):
        for j, properties in enumerate(dataset):
            species = properties['species'].to(device)
            species_counts = torch.zeros((species.shape[0], num_species), dtype=torch.float, device=device)
            for n in range(num_species):
                species_counts[:, n] = (species == n).sum(-1).float()
            true_energies = properties['energies'].float().to(device)
            predicted_energies = model(species_counts)
            loss = (true_energies - predicted_energies).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if j == num_batches_to_use - 1:
                break
    model.m.requires_grad_(False)
    m_out = model.m.data.cpu()

    b_out: tp.Optional[Tensor]
    if model.b is not None:
        model.b.requires_grad_(False)
        b_out = model.b.data.cpu()
    else:
        b_out = None
    return m_out, b_out


def _calculate_saes_exact(dataset, num_species: int, num_batches_to_use: int,
                          device: str = 'cpu',
                          fit_intercept: bool = False) -> tp.Tuple[Tensor, tp.Optional[Tensor]]:

    if num_batches_to_use == len(dataset):
        warnings.warn("Using all batches to estimate SAE, this may take up a lot of memory.")
    list_species_counts = []
    list_true_energies = []
    for j, properties in enumerate(dataset):
        species = properties['species'].to(device)
        true_energies = properties['energies'].float().to(device)
        species_counts = torch.zeros((species.shape[0], num_species), dtype=torch.float, device=device)
        for n in range(num_species):
            species_counts[:, n] = (species == n).sum(-1).float()
        list_species_counts.append(species_counts)
        list_true_energies.append(true_energies)
        if j == num_batches_to_use - 1:
            break

    if fit_intercept:
        list_species_counts.append(torch.ones(num_species, device=device, dtype=torch.float))
    total_true_energies = torch.cat(list_true_energies, dim=0)
    total_species_counts = torch.cat(list_species_counts, dim=0)

    # here total_true_energies is of shape m x 1 and total_species counts is m x n
    # n = num_species if we don't fit an intercept, and is equal to num_species + 1
    # if we fit an intercept. See the torch documentation for linalg.lstsq for more info
    x = torch.linalg.lstsq(total_species_counts, total_true_energies.unsqueeze(-1), driver='gels').solution
    m_out = x.T.squeeze()

    b_out: tp.Optional[Tensor]
    if fit_intercept:
        b_out = x[num_species]
    else:
        b_out = None
    return m_out, b_out
