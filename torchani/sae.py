import typing as tp
import warnings
import math

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from torchani.datasets import ANIBatchedDataset
from torchani.transforms import AtomicNumbersToIndices


def calculate_saes(
    dataset: tp.Union[DataLoader, ANIBatchedDataset],
    elements: tp.Sequence[str],
    mode: str = "sgd",
    fraction: float = 1.0,
    fit_intercept: bool = False,
    device: str = "cpu",
    max_epochs: int = 1,
    lr: float = 0.01,
    verbose: bool = False,
) -> tp.Tuple[Tensor, tp.Optional[Tensor]]:
    if mode == "exact":
        if lr != 0.01:
            raise ValueError("lr is only used with mode=sgd")
        if max_epochs != 1:
            raise ValueError("max_epochs is only used with mode=sgd")

    if mode not in ["sgd", "exact"]:
        raise ValueError("'mode' must be one of ['sgd', 'exact']")
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
    if verbose:
        print(
            f"Using {num_batches_to_use} batches"
            f" out of a total of {len(dataset)} batches"
            " to estimate SAE"
        )

    if mode == "exact":
        if verbose:
            print("Calculating SAE using exact OLS method...")
        m_out, b_out = _calculate_saes_exact(
            dataset,
            num_species,
            num_batches_to_use,
            device=device,
            fit_intercept=fit_intercept,
        )
    elif mode == "sgd":
        if verbose:
            print("Estimating SAE using stochastic gradient descent...")
        m_out, b_out = _calculate_saes_sgd(
            dataset,
            num_species,
            num_batches_to_use,
            device=device,
            fit_intercept=fit_intercept,
            max_epochs=max_epochs,
            lr=lr,
        )

    if isinstance(dataset, DataLoader):
        assert isinstance(dataset.dataset, ANIBatchedDataset)
        dataset.dataset.transform = old_transform
    else:
        assert isinstance(dataset, ANIBatchedDataset)
        dataset.transform = old_transform
    return m_out, b_out


def _calculate_saes_sgd(
    dataset,
    num_species: int,
    num_batches_to_use: int,
    device: str = "cpu",
    fit_intercept: bool = False,
    max_epochs: int = 1,
    lr: float = 0.01,
) -> tp.Tuple[Tensor, tp.Optional[Tensor]]:
    class LinearModel(torch.nn.Module):
        m: torch.nn.Parameter
        b: tp.Optional[torch.nn.Parameter]

        def __init__(self, num_species: int, fit_intercept: bool = False):
            super().__init__()
            self.register_parameter(
                "m", torch.nn.Parameter(torch.ones(num_species, dtype=torch.float))
            )
            if fit_intercept:
                self.register_parameter(
                    "b", torch.nn.Parameter(torch.zeros(1, dtype=torch.float))
                )
            else:
                self.register_parameter("b", None)

        def forward(self, x: Tensor) -> Tensor:
            x *= self.m
            if self.b is not None:
                x += self.b
            return x.sum(-1)

    model = LinearModel(num_species, fit_intercept).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(max_epochs):
        for j, properties in enumerate(dataset):
            species = properties["species"].to(device)
            species_counts = torch.zeros(
                (species.shape[0], num_species), dtype=torch.float, device=device
            )
            for n in range(num_species):
                species_counts[:, n] = (species == n).sum(-1).float()
            true_energies = properties["energies"].float().to(device)
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


def _calculate_saes_exact(
    dataset,
    num_species: int,
    num_batches_to_use: int,
    device: str = "cpu",
    fit_intercept: bool = False,
) -> tp.Tuple[Tensor, tp.Optional[Tensor]]:
    if num_batches_to_use == len(dataset):
        warnings.warn(
            "Using all batches to estimate SAE, this may take up a lot of memory."
        )
    list_species_counts = []
    list_true_energies = []
    for j, properties in enumerate(dataset):
        species = properties["species"].to(device)
        true_energies = properties["energies"].float().to(device)
        species_counts = torch.zeros(
            (species.shape[0], num_species), dtype=torch.float, device=device
        )
        for n in range(num_species):
            species_counts[:, n] = (species == n).sum(-1).float()
        list_species_counts.append(species_counts)
        list_true_energies.append(true_energies)
        if j == num_batches_to_use - 1:
            break

    if fit_intercept:
        list_species_counts.append(
            torch.ones(num_species, device=device, dtype=torch.float)
        )
    total_true_energies = torch.cat(list_true_energies, dim=0)
    total_species_counts = torch.cat(list_species_counts, dim=0)

    # here total_true_energies is of shape m x 1 and total_species counts is m x n
    # n = num_species if we don't fit an intercept, and is equal to num_species + 1
    # if we fit an intercept. See the torch documentation for linalg.lstsq for more info
    x = torch.linalg.lstsq(
        total_species_counts, total_true_energies.unsqueeze(-1), driver="gels"
    ).solution
    m_out = x.T.squeeze()

    b_out: tp.Optional[Tensor]
    if fit_intercept:
        b_out = x[num_species]
    else:
        b_out = None
    return m_out, b_out
