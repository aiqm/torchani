r"""Functions to calculate self atomic energies (SAEs) via linear regression.

It is recommended to use GSAEs (Ground State Atomic Energies) for new models instead, so
that models predict atomization energies.
"""

import typing as tp
import math

import torch
from torch import Tensor

from torchani.annotations import Device
from torchani.datasets import BatchedDataset
from torchani.transforms import AtomicNumbersToIndices

__all__ = ["exact_saes", "approx_saes"]


def exact_saes(
    dataset: BatchedDataset,
    symbols: tp.Sequence[str],
    fraction: float = 1.0,
    fit_intercept: bool = False,
    device: Device = None,
) -> tp.Tuple[Tensor, tp.Optional[Tensor]]:
    r"""Calculate SAEs of a dataset

    Given a `torchani.datasets.BatchedDataset` class, this function calculates
    the associated SAEs.

    Args:
        dataset: Batched dataset to use
        symbols: |symbols|
        fraction: Fraction of the dataset to use.
        fit_intercept: Whether to let the multilinear regression not go through zero.
        device: Device to use for tensors
    """
    old_transform = dataset.transform
    dataset.transform = AtomicNumbersToIndices(symbols)
    num_species = len(symbols)
    num_batches_to_use = math.ceil(len(dataset) * fraction)
    list_species_counts = []
    list_true_energies = []
    for j, properties in enumerate(dataset):
        species = properties["species"].to(device)
        true_energies = properties["energies"].to(dtype=torch.float, device=device)
        species_counts = torch.zeros(
            (species.shape[0], num_species), dtype=torch.float, device=device
        )
        for n in range(num_species):
            species_counts[:, n] = (species == n).sum(-1).float()
        list_species_counts.append(species_counts)
        list_true_energies.append(true_energies)
        if j == num_batches_to_use - 1:
            break
    dataset.transform = old_transform

    if fit_intercept:
        list_species_counts.append(
            torch.ones(num_species, device=device, dtype=torch.float)
        )
    total_true_energies = torch.cat(list_true_energies, dim=0)
    total_species_counts = torch.cat(list_species_counts, dim=0)

    # here total_true_energies is of shape m x 1 and total_species counts is m x n. n =
    # num_species if we don't fit an intercept, and is equal to num_species + 1 if we
    # fit an intercept. See the torch documentation for linalg.lstsq for more info
    x = torch.linalg.lstsq(
        total_species_counts, total_true_energies.unsqueeze(-1), driver="gels"
    ).solution
    m_out = x.T.squeeze()
    if fit_intercept:
        return m_out, x[num_species]
    return m_out, None


def approx_saes(
    dataset: BatchedDataset,
    symbols: tp.Sequence[str],
    fraction: float = 1.0,
    fit_intercept: bool = False,
    device: Device = None,
    max_epochs: int = 1,
    lr: float = 0.01,
) -> tp.Tuple[Tensor, tp.Optional[Tensor]]:
    r"""Calculate SAEs of a dataset in an approximate manner, using SGD

    Given a `torchani.datasets.BatchedDataset` class, this function calculates
    the associated SAEs using stochastic gradient descent.

    Args:
        dataset: Batched dataset to use
        symbols: |symbols|
        fraction: Fraction of the dataset to use.
        fit_intercept: Whether to let the multilinear regression not go through zero.
        device: Device to use for tensors
        max_epochs: Maximum number of epochs
        lr: Learning rate
        verbose: Whether to print detailed info to ``stdout``.
    """
    old_transform = dataset.transform
    dataset.transform = AtomicNumbersToIndices(symbols)
    num_species = len(symbols)
    num_batches_to_use = math.ceil(len(dataset) * fraction)
    model = _LinearModel(num_species, fit_intercept).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(max_epochs):
        for j, properties in enumerate(dataset):
            species = properties["species"].to(device)
            species_counts = torch.zeros(
                (species.shape[0], num_species), dtype=torch.float, device=device
            )
            for n in range(num_species):
                species_counts[:, n] = (species == n).sum(-1).float()
            true_energies = properties["energies"].to(dtype=torch.float, device=device)
            predicted_energies = model(species_counts)
            loss = (true_energies - predicted_energies).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if j == num_batches_to_use - 1:
                break
    dataset.transform = old_transform
    model.m.requires_grad_(False)
    m_out = model.m.data.cpu()
    if model.b is not None:
        model.b.requires_grad_(False)
        return m_out, model.b.data.cpu()
    return m_out, None


# Utility for approx_saes
class _LinearModel(torch.nn.Module):
    m: torch.nn.Parameter
    b: tp.Optional[torch.nn.Parameter]

    def __init__(self, num_species: int, fit_intercept: bool = False):
        super().__init__()
        self.register_parameter(
            "m", torch.nn.Parameter(torch.ones(num_species, dtype=torch.float))
        )
        self.register_parameter(
            "b",
            (
                torch.nn.Parameter(torch.zeros(1, dtype=torch.float))
                if fit_intercept
                else None
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        x *= self.m
        if self.b is not None:
            x += self.b
        return x.sum(-1)
