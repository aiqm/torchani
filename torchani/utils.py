r"""
Miscellaneous utilities used throughout the code
"""

from collections import OrderedDict
import typing as tp
import tarfile
import zipfile
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

from torchani.annotations import Device
from torchani.constants import MASS, ATOMIC_NUMBER, PERIODIC_TABLE, GSAES
from torchani.tuples import SpeciesEnergies


__all__ = [
    "download_and_extract",
    "strip_redundant_padding",
    "pad_atomic_properties",
    "AtomicNumbersToChemicalSymbols",
    "IntsToChemicalSymbols",
    "ChemicalSymbolsToInts",
    "ChemicalSymbolsToAtomicNumbers",
    "AtomicNumbersToMasses",
    "get_atomic_masses",
    "PERIODIC_TABLE",
    "ATOMIC_NUMBER",
    "TightCELU",
]

# The second dimension of these keys can be assumed to be "number of atoms"
ATOMIC_KEYS = (
    "species",
    "numbers",
    "atomic_numbers",
    "coordinates",
    "forces",
    "coefficients",  # atomic density coefficients
    "atomic_charges",
    "atomic_volumes_mbis",
    "atomic_charges_mbis",
    "atomic_dipole_magnitudes_mbis",
    "atomic_quadrupole_magnitudes_mbis",
    "atomic_octupole_magnitudes_mbis",
    "atomic_dipoles",
    "atomic_polarizabilities",
)

SYMBOLS_1X = ("H", "C", "N", "O")
SYMBOLS_2X = ("H", "C", "N", "O", "S", "F", "Cl")


PADDING = {
    "species": -1,
    "numbers": -1,
    "atomic_numbers": -1,
    "coordinates": 0.0,
    "forces": 0.0,
    "energies": 0.0,
}


class TightCELU(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.celu(x, alpha=0.1)


def download_and_extract(
    url: str,
    file_name: str,
    dest_dir: Path,
    verbose: bool = False,
) -> None:
    dest_dir.mkdir(exist_ok=True)
    r"""
    Download and extract a .tar.gz or .zip file form a given url
    """
    # Download
    dest_path = dest_dir / file_name
    if verbose:
        print(f"Downloading {file_name}...")
    torch.hub.download_url_to_file(url, str(dest_path), progress=verbose)
    # Extract
    if file_name.endswith(".tar.gz"):
        with tarfile.open(dest_path, "r:gz") as f:
            f.extractall(dest_dir)
    elif file_name.endswith(".zip"):
        with zipfile.ZipFile(dest_path) as zf:
            zf.extractall(dest_path.parent)
    # Delete
    dest_path.unlink()


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
    r"""
    Transform an array of strings into the corresponding formulas

    Take a ``numpy`` ndarray of shape ``(molecules, atoms)`` with chemical symbols and
    return a list of formulas with ``len(formulas) = molecules``. Sorting of symbols
    within formulas is alphabetical e.g. ``[['H', 'H', 'C']] -> ['CH2']``
    """
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
    r"""
    Cumulative sum just like ``torch``'s ``torch.cumsum``, but with the first element of
    the result being zero
    """
    cumsum = torch.zeros_like(input_)
    torch.cumsum(input_[:-1], dim=0, out=cumsum[1:])
    return cumsum


def nonzero_in_chunks(tensor: Tensor, chunk_size: int = 2**31 - 1):
    r"""
    Flatten a tensor and applies nonzero in chunks of a given size

    Workaround for a limitation in PyTorch's nonzero function, which fails with a
    ``RuntimeError`` when applied to tensors with more than ``INT_MAX`` elements.

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
    r"""
    Combine a sequence of properties together into single tensor.

    Inputs are ``[{'species': tensor, ...}, {'species': tensor, ...}, ...]`j` and the
    output is of the form ``{'species': padded_tensor, ...}``.

    Arguments:
        properties (list[dict[str, Tensor]]): Sequence of properties
        padding_values (Optional[list[dict[str, float]]]): Values to use for padding.

    Returns:
        dict[str, Tensor]: Padded tensors.
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
            output[k][index0:index0 + n, 0:original_size, ...] = x[k]
            index0 += n
    return output


def strip_redundant_padding(
    properties: tp.Dict[str, Tensor],
    atomic_properties: tp.Iterable[str] = ATOMIC_KEYS,
) -> tp.Dict[str, Tensor]:
    # NOTE: Assume that the padding value is -1
    species = properties["species"]
    non_padding = (species >= 0).any(dim=0).nonzero().squeeze()
    for k in atomic_properties:
        if k in properties:
            properties[k] = properties[k].index_select(1, non_padding)
    return properties


def map_to_central(coordinates: Tensor, cell: Tensor, pbc: Tensor) -> Tensor:
    r"""
    Map atoms outside the unit cell into the cell using PBC

    Arguments:

        coordinates (:class:`torch.Tensor`): Float Tensor,  ``(molecules, atoms, 3)``.
        cell (:class:`torch.Tensor`): Float tensor of shape ``(3, 3)`` of the three
            vectors defining unit cell:

            .. code-block:: python

                tensor([[x1, y1, z1],
                        [x2, y2, z2],
                        [x3, y3, z3]])
        pbc (:class:`torch.Tensor`): Boolean tensor of shape ``(3,)`` size 3 storing
            whether PBC is enabled for each direction.

    Returns:
        :class:`torch.Tensor`: Coords of atoms mapped to the unit cell.
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
    """
    Helper class for adding and subtracting self atomic energies

    Note: This class is *legacy*. Please use
    :class:`torchani.potentials.EnergyAddder`, which has equivalent
    functionality instead of this class.

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

    @staticmethod
    def _sorted_gsaes(
        elements: tp.Sequence[str], functional: str, basis_set: str
    ) -> tp.List[float]:
        gsaes = GSAES[f"{functional.lower()}-{basis_set.lower()}"]
        return [gsaes[e] for e in elements]

    @classmethod
    def with_gsaes(cls, elements: tp.Sequence[str], functional: str, basis_set: str):
        r"""Instantiate an EnergyShifter with a given set of GSAES"""
        return cls(
            cls._sorted_gsaes(elements, functional, basis_set), fit_intercept=False
        )

    @torch.jit.export
    def _atomic_saes(self, species: Tensor) -> Tensor:
        # Compute atomic self energies for a set of species.
        self_atomic_energies = self.self_energies[species]
        self_atomic_energies = self_atomic_energies.masked_fill(species == -1, 0.0)
        return self_atomic_energies

    @torch.jit.export
    def sae(self, species: Tensor) -> Tensor:
        """Compute self energies for molecules.

        Padding atoms are automatically excluded.

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
        species, energies = species_energies
        sae = self._atomic_saes(species).sum(dim=1)

        if self.fit_intercept:
            sae += self.self_energies[-1]
        return SpeciesEnergies(species, energies + sae)


class _NumbersConvert(torch.nn.Module):
    def __init__(self, symbol_dict: tp.Dict[int, str]):
        self.symbol_dict = symbol_dict
        super().__init__()

    def forward(self, species: Tensor) -> tp.List[str]:
        assert species.dim() == 1, "Only 1D tensors supported"
        species = species[species != -1]
        # This can't be an in-place loop to be jit-compilable
        out: tp.List[str] = []
        for x in species:
            out.append(self.symbol_dict[x.item()])
        return out

    def __len__(self) -> int:
        return len(self.symbol_dict)


class AtomicNumbersToChemicalSymbols(_NumbersConvert):
    r"""Converts tensor of atomic numbers to list of chemical symbols

    On initialization, it is optional to supply the class with a :class:'dict'
    containing custom numbers and symbols. This is not necessary, as the class
    is provided ATOMIC_NUMBER by default. Otherwise, the class should be
    supplied with a :class:`list` (or in general
    :class:`collections.abc.Sequence`) of :class:`str`.The returned instance is
    a callable object, which can be called an arbituary tensor of the supported
    atomic numbers that is converted into a list of strings.

    Usage example:
    .. code-block:: python

       # We intialize our class
       numbers_to_symbols = AtomicNumberstoChemicalSymbols()

       # We have atomic numbers which we want to convert to a species list
       atomic_numbers  = torch.tensor([6, 1, 1, 1])

       symbols = numbers_to_symbols(atomic_numbers)

       Output:
        ['C', 'H', 'H', 'H']

     Arguments:
        atomic_numbers: tensor of atomic number values you wish to convert (must be 1-D)
    """

    def __init__(self):
        super().__init__({v: k for k, v in ATOMIC_NUMBER.items()})


class IntsToChemicalSymbols(_NumbersConvert):
    r"""Convert tensor or list of integers to list[str] of chemical symbols

    On initialization, it is optional to supply the class with a :class:'dict'
    containing custom numbers and symbols. This is not necessary, as the class
    is provided ATOMIC_NUMBER by default. Otherwise, the class should be
    supplied with a :class:`list` (or in general
    :class:`collections.abc.Sequence`) of :class:`str`. The returned instance
    is a callable object, which can be called with an arbitrary list or tensor
    of the supported indicies that is converted into a list of strings.

    Usage example:

        #species list used for indexing
        elements = ['H','C','N','O','S','F', 'Cl']

        species_converter = IntsToChemicalSymbols(elements)

        species = torch.Tensor([3, 0, 0, -1, -1, -1])

        species_converter(species)

        Output:
            ['O', 'H', 'H']

    Arguments:
        elements: list of species in your model, used for indexing
        species: list or tensor of species integer values you wish to convert
        (must be 1-D)

    """

    def __init__(self, symbols: tp.Sequence[str]):
        if isinstance(symbols, str):
            raise ValueError("symbols must be a sequence of str, but it can't be a str")
        super().__init__({i: s for i, s in enumerate(symbols)})


class _ChemicalSymbolsConvert(torch.nn.Module):
    _dummy: Tensor

    def __init__(self, symbol_dict: tp.Dict[str, int], device: Device = "cpu"):
        super().__init__()
        self.symbol_dict = symbol_dict
        self.register_buffer("_dummy", torch.empty(0, device=device), persistent=False)

    def forward(self, species: tp.List[str]) -> Tensor:
        # This can't be an in-place loop to be jit-compilable
        numbers_list: tp.List[int] = []
        for x in species:
            numbers_list.append(self.symbol_dict[x])
        return torch.tensor(numbers_list, dtype=torch.long, device=self._dummy.device)

    def __len__(self) -> int:
        return len(self.symbol_dict)


class ChemicalSymbolsToAtomicNumbers(_ChemicalSymbolsConvert):
    r"""Converts a sequence of chemical symbols into a tensor of atomic numbers


    On initialization, it is optional to supply the class with a :class:'dict'
    containing custom numbers and symbols. This is not necessary, as the
    class is provided ATOMIC_NUMBER by default.
    Output is a tensor of dtype :class:`torch.long`. Usage example:

    .. code-block:: python

        # We have a species list which we want to convert to atomic numbers
        symbols_to_numbers = ChemicalSymbolsToAtomicNumbers()
        species_convert = ['C', 'S', 'O', 'F', 'H', 'H']
        atomic_numbers = symbols_to_numbers(species_convert)
        # atomic_numbers is now torch.tensor([ 6, 16,  8,  9,  1,  1])

    Arguments:
        species_convert: list of chemical symbols to convert to atomic numbers
    """

    def __init__(self, device: Device = "cpu"):
        super().__init__(ATOMIC_NUMBER, device=device)


class ChemicalSymbolsToInts(_ChemicalSymbolsConvert):
    r"""Helper that can be called to convert chemical symbol string to integers

    On initialization the class should be supplied with a :class:`list` of
    :class:`str`. The returned instance is a callable object, which can be
    called with an arbitrary list of the supported species that is converted
    into a tensor of dtype :class:`torch.long`. Usage example:

    .. code-block:: python

        from torchani.utils import ChemicalSymbolsToInts

        # We initialize ChemicalSymbolsToInts with the supported species
        elements = ['H', 'C', 'N', 'O', 'S', 'F', 'Cl']
        species_to_tensor = ChemicalSymbolsToInts(elements)

        species_convert = ['C', 'S', 'O', 'F', 'H', 'H']

        # We have a species list which we want to convert to an index tensor
        index_tensor = species_to_tensor(species_convert)

        # index_tensor is now [1, 4, 3, 5, 0, 0]

    Arguments:
        elements: list of species in your model, used for indexing
        species_convert: list of chemical symbols to convert to atomic numbers
    """

    def __init__(self, symbols: tp.Sequence[str], device: Device = "cpu"):
        if isinstance(symbols, str):
            raise ValueError("symbols must be a sequence of str, but it can't be a str")
        int_dict = {s: i for i, s in enumerate(symbols)}
        super().__init__(int_dict, device=device)


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
        device: Device = "cpu",
        dtype: torch.dtype = torch.float,
    ) -> None:
        super().__init__()
        if not masses:
            masses = MASS
        self.register_buffer(
            "atomic_masses",
            torch.tensor(masses, device=device, dtype=dtype),
            persistent=False,
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
