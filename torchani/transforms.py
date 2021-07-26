r"""Transforms to be applied to properties when training. The usage is the same
as transforms in torchvision.

Example:
Most TorchANI modules take atomic indices ("species") as input as a
default, so if you want to convert atomic numbers to indices and then subtract
the self atomic energies (SAE) when iterating from a batched dataset you can
call

from torchani.transforms import AtomicNumbersToIndices, SubtractSAE, Compose
from torchani.datasets import ANIBatchedDataset

transform = Compose([AtomicNumbersToIndices(('H', 'C', 'N'), SubtractSAE([-0.57, -0.0045, -0.0035])])
training = ANIBatchedDataset('/path/to/database/', transform=transform, split='training')
validation = ANIBatchedDataset('/path/to/database/', transform=transform, split='validation')
"""
from typing import Dict, Sequence, Union, Tuple, Optional, List
import math
import warnings

import torch
from torch import Tensor

from .utils import EnergyShifter, PERIODIC_TABLE, ATOMIC_NUMBERS
from .nn import SpeciesConverter
from .datasets import ANIBatchedDataset
from torch.utils.data import DataLoader


class SubtractRepulsion(torch.nn.Module):

    def __init__(self, elements: Union[Sequence[str], Sequence[int]]):
        super().__init__()

    def forward(self, properties: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError("Not yet implemented")
        return properties


class SubtractSAE(torch.nn.Module):

    atomic_numbers: Tensor

    def __init__(self, elements: Union[Sequence[str], Sequence[int]], self_energies: List[float], intercept: float = 0.0):
        super().__init__()
        symbols, atomic_numbers = _parse_elements(elements)
        if len(self_energies) != len(atomic_numbers):
            raise ValueError("There should be one self energy per element")
        self.register_buffer('supported_atomic_numbers', torch.tensor(atomic_numbers, dtype=torch.long))
        if intercept != 0.0:
            self_energies.append(intercept)
            # for some reason energy_shifter is defaulted as double, so I make
            # it float here
            self.energy_shifter = EnergyShifter(self_energies, fit_intercept=True).float()
        else:
            self.energy_shifter = EnergyShifter(self_energies).float()

    def forward(self, properties: Dict[str, Tensor]) -> Dict[str, Tensor]:
        properties['energies'] -= self.energy_shifter.sae(properties['species'])
        return properties


class AtomicNumbersToIndices(torch.nn.Module):

    atomic_numbers: Tensor

    def __init__(self, elements: Union[Sequence[str], Sequence[int]]):
        super().__init__()
        symbols, atomic_numbers = _parse_elements(elements)

        self.register_buffer('supported_atomic_numbers', torch.tensor(atomic_numbers, dtype=torch.long))
        self.converter = SpeciesConverter(symbols)

    def forward(self, properties: Dict[str, Tensor]) -> Dict[str, Tensor]:
        species = self.converter((properties['species'], properties['coordinates'])).species
        properties['species'] = species
        return properties


def _parse_elements(elements: Union[Sequence[str], Sequence[int]]) -> Tuple[Sequence[str], Sequence[int]]:
    assert len(elements) == len(set(elements)), 'Elements should not be duplicated'
    symbols: List[str] = []
    atomic_numbers: List[int] = []
    if isinstance(elements[0], int):
        for e in elements:
            assert isinstance(e, int) and e > 0, f"Encountered an atomic number that is <= 0 {elements}"
            symbols.append(PERIODIC_TABLE[e])
            atomic_numbers.append(e)
    else:
        for e in elements:
            assert isinstance(e, str), "Input sequence must consist of chemical symbols or atomic numbers"
            symbols.append(e)
            atomic_numbers.append(ATOMIC_NUMBERS[e])
    return symbols, atomic_numbers


class Compose(torch.nn.Module):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    # This code is mostly copied from torchvision, but made JIT scriptable

    def __init__(self, transforms: Sequence[torch.nn.Module]):
        super().__init__()
        if not isinstance(transforms[0], AtomicNumbersToIndices):
            warnings.warn("The first transform in your pipeline is not AtomicNumbersToIndices, make sure that this is your intent")
        transform_names = [type(t).__name__ for t in transforms]

        if len(transform_names) != len(set(transform_names)):
            warnings.warn("Your transform pipeline seems to have duplicate transforms, make sure that this is your intent")

        all_atomic_numbers: List[Tensor] = []
        for t in transforms:
            if hasattr(t, 'atomic_numbers') and isinstance(t.atomic_numbers, Tensor):
                all_atomic_numbers.append(t.atomic_numbers)

        for z in all_atomic_numbers:
            if not z == all_atomic_numbers[0]:
                raise ValueError("Two or more of your transforms use different atomic numbers, this is incorrect")

        self.transforms = torch.nn.ModuleList(transforms)

    def forward(self, properties: Dict[str, Tensor]) -> Dict[str, Tensor]:
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


def calculate_saes(dataset: Union[DataLoader, ANIBatchedDataset],
                         elements: Sequence[str],
                         mode: str = 'sgd',
                         fraction: float = 1.0,
                         fit_intercept: bool = False,
                         device: str = 'cpu',
                         max_epochs: int = 1,
                         lr: float = 0.01) -> Tuple[Tensor, Optional[Tensor]]:
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
                        lr: float = 0.01) -> Tuple[Tensor, Optional[Tensor]]:

    class LinearModel(torch.nn.Module):

        m: torch.nn.Parameter
        b: Optional[torch.nn.Parameter]

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

    b_out: Union[Tensor, None]
    if model.b is not None:
        model.b.requires_grad_(False)
        b_out = model.b.data.cpu()
    else:
        b_out = None
    return m_out, b_out


def _calculate_saes_exact(dataset, num_species: int, num_batches_to_use: int,
                          device: str = 'cpu',
                          fit_intercept: bool = False) -> Tuple[Tensor, Optional[Tensor]]:

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
    x = torch.linalg.lstsq(total_species_counts, total_true_energies, driver='gels').solution
    m_out = x.T.squeeze()

    b_out: Union[Tensor, None]
    if fit_intercept:
        b_out = x[num_species]
    else:
        b_out = None
    return m_out, b_out
