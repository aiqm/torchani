r"""
Calculation and bookkeeping of self atomic energies (SAEs) and ground state
atomic energies (GSAEs).
"""
import typing as tp
import warnings
import math

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from torchani.datasets import BatchedDataset
from torchani.transforms import AtomicNumbersToIndices


# GSAES were calculating using the following splin multiplicities: H: 2, C: 3,
# N: 4, O: 3, S: 3, F: 2, Cl: 2 and using UKS in all cases, with tightscf, on
# orca 4.2.3 (except for the wB97X-631Gd energies, which were computed with
# Gaussian 09).
#
# The coupled cluster GSAE energies are calculated using DLPNO-CCSD def2-TZVPP
# def2-TZVPP/C which is not the exact same as CCSD(T)*/CBS but is close enough
# for atomic energies. For H I set the E to -0.5 since that is the exact
# nonrelativistic solution and I believe CC can't really converge for H.
GSAES: tp.Dict[str, tp.Dict[str, float]] = {
    "b973c-def2mtzvp": {
        "H": -0.506930113968,
        "C": -37.81441001258,
        "N": -54.556538547322,
        "O": -75.029181326588,
        "F": -99.688618987039,
        "S": -398.043159341582,
        "Cl": -460.082223445159,
    },
    "wb97x-631gd": {
        "C": -37.8338334,
        "Cl": -460.116700600,
        "F": -99.6949007,
        "H": -0.4993212,
        "N": -54.5732825,
        "O": -75.0424519,
        "S": -398.0814169,
    },
    "wB97md3bj-def2tzvpp": {
        "C": -37.870597534068,
        "Cl": -460.197921425433,
        "F": -99.784869113871,
        "H": -0.498639663159,
        "N": -54.621568655507,
        "O": -75.111870707635,
        "S": -398.158126819835,
    },
    "wb97mv-def2tzvpp": {
        "C": -37.844395699666,
        "Cl": -460.124987825603,
        "F": -99.745234404775,
        "H": -0.494111111003,
        "N": -54.590952163069,
        "O": -75.076760965132,
        "S": -398.089446664032,
    },
    "ccsd(t)star-cbs": {
        "C": -37.780724507998,
        "Cl": -459.664237510771,
        "F": -99.624864557142,
        "H": -0.5000000000000,
        "N": -54.515992576387,
        "O": -74.976148184192,
        "S": -397.646401989238,
    },
    "dsd_blyp_d3bj-def2tzvp": {
        "H": -0.4990340388250001,
        "C": -37.812711066967,
        "F": -99.795668645591,
        "Cl": -460.052391015914,
        "Br": -2573.595184605241,
        "I": -297.544092721991,
    },
    "wb97m_d3bj-def2tzvppd": {
        "H": -0.4987605100487531,
        "C": -37.87264507233593,
        "O": -75.11317840410095,
        "N": -54.62327513368922,
        "F": -99.78611622985483,
        "Cl": -460.1988762285739,
        "S": -398.1599636677874,
        "Br": -2574.1167240829964,
        "I": -297.76228914445625,
        "P": -341.3059197024934,
    },
    "revpbe_d3bj-def2tzvp": {
        "H": -0.504124985686,
        "C": -37.845615868613,
        "N": -54.587739850180995,
        "O": -75.071223222771,
        "S": -398.041639842051,
    },
    "wb97x-def2tzvpp": {
        "C": -37.8459781,
        "Cl": -460.1467777,
        "F": -99.7471707,
        "H": -0.5013925,
        "N": -54.5915914,
        "O": -75.0768759,
        "S": -398.1079973,
    },
}


def sorted_gsaes(
    elements: tp.Sequence[str], functional: str, basis_set: str
) -> tp.List[float]:
    r"""Return a sequence of GSAES sorted by element

    Example usage:
    gsaes = sorted_gsaes(('H', 'C', 'S'), 'wB97X', '631Gd')
    # gsaes = [-0.4993213, -37.8338334, -398.0814169]

    Functional and basis set are case insensitive
    """
    gsaes = GSAES[f"{functional.lower()}-{basis_set.lower()}"]
    return [gsaes[e] for e in elements]


def calculate_saes(
    dataset: tp.Union[DataLoader, BatchedDataset],
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
        assert isinstance(dataset.dataset, BatchedDataset)
        old_transform = dataset.dataset.transform
        dataset.dataset.transform = AtomicNumbersToIndices(elements)
    else:
        assert isinstance(dataset, BatchedDataset)
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
        assert isinstance(dataset.dataset, BatchedDataset)
        dataset.dataset.transform = old_transform
    else:
        assert isinstance(dataset, BatchedDataset)
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
