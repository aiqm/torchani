from collections import OrderedDict
import typing as tp
import math
import os
import warnings
import itertools
from pathlib import Path
from collections import Counter

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
import torch.utils.data

from torchani.constants import MASS, ATOMIC_NUMBER, PERIODIC_TABLE
from torchani.tuples import SpeciesEnergies


__all__ = [
    "pad_atomic_properties",
    "ChemicalSymbolsToInts",
    "ChemicalSymbolsToAtomicNumbers",
    "AtomicNumbersToMasses",
    "get_atomic_masses",
    "GSAES",
    "PERIODIC_TABLE",
    "ATOMIC_NUMBER",
    "TightCELU",
]


PADDING = {
    "species": -1,
    "numbers": -1,
    "atomic_numbers": -1,
    "coordinates": 0.0,
    "forces": 0.0,
    "energies": 0.0,
}

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


class TightCELU(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.celu(x, alpha=0.1)


def sorted_gsaes(
    elements: tp.Sequence[str], functional: str, basis_set: str
) -> tp.List[float]:
    r"""Return sorted GSAES by element

    Example usage:
    gsaes = sorted_gsaes(('H', 'C', 'S'), 'wB97X', '631Gd')
    # gsaes = [-0.4993213, -37.8338334, -398.0814169]

    Functional and basis set are case insensitive
    """
    gsaes = GSAES[f"{functional.lower()}-{basis_set.lower()}"]
    return [gsaes[e] for e in elements]


# Pure python linspace to ensure reproducibility
def linspace(start: float, stop: float, steps: int) -> tp.Tuple[float, ...]:
    return tuple(start + ((stop - start) / steps) * j for j in range(steps))


def check_openmp_threads(verbose: bool = True) -> None:
    if "OMP_NUM_THREADS" not in os.environ:
        warnings.warn(
            "OMP_NUM_THREADS not set."
            " MNP works best if OMP_NUM_THREADS >= 2."
            " You can set this variable by running 'export OMP_NUM_THREADS=4')"
            " or 'export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK' if using slurm"
        )
        return

    num_threads = int(os.environ["OMP_NUM_THREADS"])
    if num_threads <= 0:
        raise RuntimeError(f"OMP_NUM_THREADS set to an incorrect value: {num_threads}")
    if verbose:
        print(f"OMP_NUM_THREADS set to: {num_threads}")


def species_to_formula(species: NDArray[np.str_]) -> tp.List[str]:
    r"""Transforms an array of strings into the corresponding formula.  This
    function expects an array of shape (M, A) and returns a list of
    formulas of len M.
    sorts in alphabetical order e.g. [['H', 'H', 'C']] -> ['CH2']"""
    if species.ndim == 1:
        species = np.expand_dims(species, axis=0)
    elif species.ndim != 2:
        raise ValueError("Species needs to have two dims/axes")
    formulas = []
    for s in species:
        symbol_counts: tp.List[tp.Tuple[str, int]] = sorted(Counter(s).items())
        iterable = (
            str(i) if str(i) != "1" else ""
            for i in itertools.chain.from_iterable(symbol_counts)
        )
        formulas.append("".join(iterable))
    return formulas


def cumsum_from_zero(input_: Tensor) -> Tensor:
    r"""Cumulative sum just like pytorch's cumsum, but with the first element
    of the result being zero"""
    cumsum = torch.zeros_like(input_)
    torch.cumsum(input_[:-1], dim=0, out=cumsum[1:])
    return cumsum


def nonzero_in_chunks(tensor: Tensor, chunk_size: int = 2**31 - 1):
    r"""Flattens a tensor and applies nonzero in chunks of a given size

    This is a workaround for a limitation in PyTorch's nonzero function, which
    fails with a `RuntimeError` when applied to tensors with more than INT_MAX
    elements.

    The issue is documented in PyTorch's GitHub repository:
    https://github.com/pytorch/pytorch/issues/51871
    """
    tensor = tensor.view(-1)
    num_splits = math.ceil(tensor.numel() / chunk_size)

    if num_splits <= 1:
        return tensor.nonzero().view(-1)

    # Split tensor into chunks, and for each chunk find nonzero elements and
    # adjust the indices in each chunk to account for their original position
    # in the tensor. Finally collect the results
    offset = 0
    nonzero_chunks: tp.List[Tensor] = []
    for chunk in torch.chunk(tensor, num_splits):
        nonzero_chunks.append(chunk.nonzero() + offset)
        offset += chunk.shape[0]
    return torch.cat(nonzero_chunks).view(-1)


def fast_masked_select(x: Tensor, mask: Tensor, idx: int) -> Tensor:
    # x.index_select(0, tensor.view(-1).nonzero().view(-1)) is EQUIVALENT to:
    # torch.masked_select(x, tensor) but FASTER
    # nonzero_in_chunks calls tensor.view(-1).nonzero().view(-1)
    # but support very large tensors, with numel > INT_MAX
    return x.index_select(idx, nonzero_in_chunks(mask))


def pad_atomic_properties(
    properties: tp.Sequence[tp.Mapping[str, Tensor]],
    padding_values: tp.Optional[tp.Dict[str, float]] = None,
) -> tp.Dict[str, Tensor]:
    """Put a sequence of atomic properties together into single tensor.

    Inputs are `[{'species': ..., ...}, {'species': ..., ...}, ...]` and the outputs
    are `{'species': padded_tensor, ...}`

    Arguments:
        properties (:class:`collections.abc.Sequence`): sequence of properties.
        padding_values (dict): the value to fill to pad tensors to same size
    """
    if padding_values is None:
        padding_values = PADDING

    vectors = [k for k in properties[0].keys() if properties[0][k].dim() > 1]
    scalars = [k for k in properties[0].keys() if properties[0][k].dim() == 1]
    padded_sizes = {k: max(x[k].shape[1] for x in properties) for k in vectors}
    num_molecules = [x[vectors[0]].shape[0] for x in properties]
    total_num_molecules = sum(num_molecules)
    output = {}
    for k in scalars:
        output[k] = torch.cat([x[k] for x in properties])
    for k in vectors:
        tensor = properties[0][k]
        shape = list(tensor.shape)
        device = tensor.device
        dtype = tensor.dtype
        if dtype in [torch.uint8, torch.int8, torch.int16, torch.int32]:
            dtype = torch.long
        shape[0] = total_num_molecules
        shape[1] = padded_sizes[k]
        output[k] = torch.full(
            shape, padding_values.get(k, 0.0), device=device, dtype=dtype
        )
        index0 = 0
        for n, x in zip(num_molecules, properties):
            original_size = x[k].shape[1]
            # here x[k] is implicitly cast to long if it has another integer type
            output[k][index0: index0 + n, 0:original_size, ...] = x[k]
            index0 += n
    return output


def map_to_central(coordinates: Tensor, cell: Tensor, pbc: Tensor) -> Tensor:
    """Map atoms outside the unit cell into the cell using PBC.

    Arguments:

        coordinates (:class:`torch.Tensor`): Tensor of shape
            ``(molecules, atoms, 3)``.

        cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three
            vectors defining unit cell:

            .. code-block:: python

                tensor([[x1, y1, z1],
                        [x2, y2, z2],
                        [x3, y3, z3]])

        pbc (:class:`torch.Tensor`): boolean vector of size 3 storing
            if pbc is enabled for that direction.

    Returns:
        :class:`torch.Tensor`: coordinates of atoms mapped back to unit cell.
    """
    # Step 1: convert coordinates from standard cartesian coordinate to unit
    # cell coordinates
    inv_cell = torch.inverse(cell)
    coordinates_cell = torch.matmul(coordinates, inv_cell)
    # Step 2: wrap cell coordinates into [0, 1)
    coordinates_cell -= coordinates_cell.floor() * pbc
    # Step 3: convert from cell coordinates back to standard cartesian
    # coordinate
    return torch.matmul(coordinates_cell, cell)


class EnergyShifter(torch.nn.Module):
    """Helper class for adding and subtracting self atomic energies

    This is a subclass of :class:`torch.nn.Module`, so it can be used directly
    in a pipeline as ``[input->AEVComputer->ANIModel->EnergyShifter->output]``.

    Arguments:
        self_energies (:class:`collections.abc.Sequence`): Sequence of floating
            numbers for the self energy of each atom type. The numbers should
            be in order, i.e. ``self_energies[i]`` should be atom type ``i``.
        fit_intercept (bool): Whether to calculate the intercept during the LSTSQ
            fit. The intercept will also be taken into account to shift energies.
    """

    self_energies: Tensor

    def __init__(self, self_energies, fit_intercept=False):
        super().__init__()

        self.fit_intercept = fit_intercept
        if self_energies is not None:
            self_energies = torch.tensor(self_energies, dtype=torch.double)

        self.register_buffer("self_energies", self_energies)

    @classmethod
    def with_gsaes(cls, elements: tp.Sequence[str], functional: str, basis_set: str):
        r"""Instantiate an EnergyShifter with a given set of GSAES"""
        obj = cls(sorted_gsaes(elements, functional, basis_set), fit_intercept=False)
        return obj

    @torch.jit.export
    def _atomic_saes(self, species: Tensor) -> Tensor:
        # Compute atomic self energies for a set of species.
        self_atomic_energies = self.self_energies[species]
        self_atomic_energies = self_atomic_energies.masked_fill(species == -1, 0.0)
        return self_atomic_energies

    @torch.jit.export
    def sae(self, species: Tensor) -> Tensor:
        """Compute self energies for molecules.

        Padding atoms will be automatically excluded.

        Arguments:
            species (:class:`torch.Tensor`): Long tensor in shape
                ``(conformations, atoms)``.

        Returns:
            :class:`torch.Tensor`: 1D vector in shape ``(conformations,)``
            for molecular self energies.
        """
        sae = self._atomic_saes(species).sum(dim=1)
        if self.fit_intercept:
            sae += self.self_energies[-1]
        return sae

    def forward(
        self,
        species_energies: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> SpeciesEnergies:
        """(species, molecular energies)->(species, molecular energies + sae)"""
        species, energies = species_energies
        sae = self._atomic_saes(species).sum(dim=1)

        if self.fit_intercept:
            sae += self.self_energies[-1]
        return SpeciesEnergies(species, energies + sae)


class ChemicalSymbolsToAtomicNumbers(torch.nn.Module):
    r"""Converts a sequence of chemical symbols into a tensor of atomic numbers

    .. code-block:: python

       # We have a species list which we want to convert to atomic numbers
       symbols_to_numbers = ChemicalSymbolsToAtomicNumbers()
       atomic_numbers = symbols_to_numbers(['H', 'C', 'H', 'H', 'C', 'Cl', 'Fe'])

       # atomic_numbers is now torch.tensor([1, 6, 1, 1, 6, 17, 26])
    """
    _dummy: Tensor
    atomics_dict: tp.Dict[str, int]

    def __init__(self, atomic_numbers: tp.Optional[tp.Dict[str, int]] = None):
        super().__init__()
        if atomic_numbers is None:
            atomic_numbers = ATOMIC_NUMBER
        self.atomics_dict = atomic_numbers
        # dummy tensor to hold output device
        self.register_buffer("_dummy", torch.empty(0), persistent=False)

    def forward(self, symbols: tp.List[str]) -> Tensor:
        numbers = [self.atomics_dict[s] for s in symbols]
        return torch.tensor(numbers, dtype=torch.long, device=self._dummy.device)


class ChemicalSymbolsToInts(torch.nn.Module):
    r"""Helper that can be called to convert chemical symbol string to integers

    On initialization the class should be supplied with a :class:`list` (or in
    general :class:`collections.abc.Sequence`) of :class:`str`. The returned
    instance is a callable object, which can be called with an arbitrary list
    of the supported species that is converted into a tensor of dtype
    :class:`torch.long`. Usage example:

    .. code-block:: python

       from torchani.utils import ChemicalSymbolsToInts


       # We initialize ChemicalSymbolsToInts with the supported species
       symbols_to_idxs = ChemicalSymbolsToInts(['H', 'C', 'Fe', 'Cl'])

       # We have a species list which we want to convert to an index tensor
       index_tensor = symbols_to_idxs(['H', 'C', 'H', 'H', 'C', 'Cl', 'Fe'])

       # index_tensor is now [0 1 0 0 1 3 2]

    Arguments:
        all_species (:class:`collections.abc.Sequence` of :class:`str`):
        sequence of all supported species, in order (it is recommended to order
        according to atomic number).
    """
    _dummy: Tensor
    rev_species: tp.Dict[str, int]

    def __init__(self, all_species: tp.Sequence[str]):
        super().__init__()
        if isinstance(all_species, str):
            raise ValueError("Input must be a *sequence of str*, but it can't be *str*")
        self.rev_species = {s: i for i, s in enumerate(all_species)}
        # dummy tensor to hold output device
        self.register_buffer("_dummy", torch.empty(0), persistent=False)

    def forward(self, species: tp.List[str]) -> Tensor:
        r"""Convert species from sequence of strings to 1D tensor"""
        rev = [self.rev_species[s] for s in species]
        return torch.tensor(rev, dtype=torch.long, device=self._dummy.device)

    def __len__(self):
        return len(self.rev_species)


class AtomicNumbersToMasses(torch.nn.Module):
    r"""Convert a tensor of atomic numbers into a tensor of atomic masses


    Arguments:
        atomic_numbers (:class:`torch.Tensor`): tensor with atomic numbers

    Returns:
        :class:`torch.Tensor`: with, atomic masses, with the same shape as the input.
    """
    atomic_masses: Tensor

    def __init__(
        self,
        masses: tp.Iterable[float] = (),
        device: tp.Union[torch.device, tp.Literal["cpu", "cuda"]] = "cpu",
        dtype: torch.dtype = torch.float,
    ) -> None:
        super().__init__()
        if not masses:
            masses = MASS
        self.register_buffer(
            "atomic_masses",
            torch.tensor(masses, device=device, dtype=dtype),
        )

    def forward(self, atomic_numbers: Tensor) -> Tensor:
        assert not (atomic_numbers == 0).any(), "Input should be atomic numbers"
        mask = atomic_numbers == -1
        masses = self.atomic_masses[atomic_numbers]
        masses.masked_fill_(mask, 0.0)
        return masses


# Convenience fn around AtomicNumbersToMasses that is non-jittable
def atomic_numbers_to_masses(
    atomic_numbers: Tensor,
    dtype: torch.dtype = torch.float,
) -> Tensor:
    if torch.jit.is_scripting():
        raise RuntimeError(
            "'torchani.utils.atomic_numbers_to_masses' doesn't support JIT, "
            " consider using torchani.utils.AtomicNumbersToMasses instead"
        )
    device = atomic_numbers.device
    return AtomicNumbersToMasses(device=device, dtype=dtype)(atomic_numbers)


# Alias for bw compatibility
get_atomic_masses = atomic_numbers_to_masses


def sort_by_element(it: tp.Iterable[str]) -> tp.Tuple[str, ...]:
    r"""
    Sort an iterable of chemical symbols by element
    """
    if isinstance(it, str):
        it = (it,)
    return tuple(sorted(it, key=lambda x: ATOMIC_NUMBER[x]))


def merge_state_dicts(paths: tp.Iterable[Path]) -> tp.OrderedDict[str, Tensor]:
    r"""
    Merge multiple single-model state dicts into a state dict for an ensemble of models
    """
    if any(not path.is_file() for path in paths):
        raise ValueError("All passed paths must be existing files with state dicts")
    merged_dict: tp.Dict[str, Tensor] = {}
    for j, path in enumerate(sorted(paths)):
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        # Compatibility with lightning state dicts
        if "state_dict" in state_dict:
            state_dict = {
                k.replace("model.", ""): v
                for k, v in state_dict["state_dict"].items()
                if k.startswith("model")
            }
        keys = tuple(state_dict.keys())
        for k in keys:
            if "neural_networks" not in k:
                continue
            new_key = k.replace("neural_networks", f"neural_networks.{j}")
            value = state_dict.pop(k)
            state_dict[new_key] = value

        for k, v in merged_dict.items():
            if "neural_networks" not in k:
                if k not in state_dict:
                    raise ValueError(f"Missing key in state dict: {k}")
                if (v != state_dict[k]).any():
                    raise ValueError(f"Incompatible values for key {k}")
        merged_dict.update(state_dict)
    return OrderedDict(merged_dict)
