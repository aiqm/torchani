r"""Atomic and Density Functional constants

Values for electronegativity and hardness for elements H-Bk, all for neutral
atoms, and are taken from Table 3 of

Carlos Cardenas et. al. Benchmark Values of Chemical Potential and Chemical
Hardness for Atoms and Atomic Ions (Including Unstable Ions) from the Energies
of Isoelectronic Series.

DOI: 10.1039/C6CP04533B

Atomic masses supported are the first 119 elements, and are taken from:

Atomic weights of the elements 2013 (IUPAC Technical Report). Meija, J.,
Coplen, T., Berglund, M., et al. (2016). Pure and Applied Chemistry, 88(3), pp.
265-291. Retrieved 30 Nov. 2016, from doi:10.1515/pac-2015-0305

They are all consistent with those used in ASE

D3BJ constants for different density functionals taken from the psi4 source
code, citations:
A. Najib, L. Goerigk, J. Comput. Theory Chem., 14 5725, 2018)
N. Mardirossian, M. Head-Gordon, Phys. Chem. Chem. Phys, 16, 9904, 2014

Except for B97-3c taken from
https://aip.scitation.org/doi/pdf/10.1063/1.5012601

And for wB97X taken from
TODO: where?
"""
import json
from pathlib import Path
import typing as tp
import math

__all__ = [
    "ATOMIC_CONSTANTS",
    "ATOMIC_NUMBER",
    "ATOMIC_MASS",
    "ATOMIC_HARDNESS",
    "ATOMIC_ELECTRONEGATIVITY",
    "MASS",
    "HARDNESS",
    "ELECTRONEGATIVITY",
    "PERIODIC_TABLE",
    "FUNCTIONAL_D3BJ_CONSTANTS",
]

with open(Path(__file__).parent / "atomic_constants.json", mode="rt") as f:
    ATOMIC_CONSTANTS = json.load(f)

with open(Path(__file__).parent / "functional_d3bj_constants.json", mode="rt") as f:
    FUNCTIONAL_D3BJ_CONSTANTS = json.load(f)


# Populate convenience variables here
ATOMIC_NUMBER: tp.Dict[str, int] = {}
ATOMIC_HARDNESS: tp.Dict[str, float] = {}
ATOMIC_ELECTRONEGATIVITY: tp.Dict[str, float] = {}
ATOMIC_MASS: tp.Dict[str, float] = {}

for symbol, values in ATOMIC_CONSTANTS.items():
    if not symbol:
        continue
    znumber = values.get("znumber")
    hardness = values.get("hardness")
    electroneg = values.get("electronegativity")
    mass = values.get("mass")
    if znumber is not None:
        ATOMIC_NUMBER[symbol] = int(znumber)
    if hardness is not None:
        ATOMIC_HARDNESS[symbol] = float(hardness)
    if electroneg is not None:
        ATOMIC_ELECTRONEGATIVITY[symbol] = float(electroneg)
    if mass is not None:
        ATOMIC_MASS[symbol] = float(mass)

# When indexed with the corresponding atomic number, PERIODIC_TABLE gives the
# element associated with it. Note that there is no element with atomic number
# 0, so an empty string is returned in this case.
PERIODIC_TABLE = ("",) + tuple(
    kv[0] for kv in sorted(ATOMIC_NUMBER.items(), key=lambda x: x[1])
)


def mapping_to_znumber_indexed_seq(
    symbols_map: tp.Mapping[str, float]
) -> tp.Tuple[float, ...]:
    r"""
    Sort the values of {symbol: value} mapping by atomic number and output a
    tuple with the sorted values.

    All elements up to the highest present atomic number element must in the mapping.

    The first element (index 0) of the output will be NaN. Example:

    .. code-block:: python
        mapping = {"H": 3.0, "Li": 1.0, "He": 0.5 }
        znumber_indexed_seq = mapping_to_znumber_indexed_seq(mapping)
        # znumber_indexed_seq will be (NaN, 3.0, 0.5, 1.0)
    """
    _symbols_map = dict(symbols_map)
    seq = [math.nan] * (len(symbols_map) + 1)
    try:
        for k, v in _symbols_map.items():
            seq[ATOMIC_NUMBER[k]] = v
    except IndexError:
        raise ValueError(f"There are missing elements in {symbols_map}") from None
    return tuple(seq)


def znumber_indexed_seq_to_mapping(
    seq: tp.Sequence[float],
) -> tp.Dict[str, float]:
    r"""
    Inverse of mapping_to_znumber_indexed_list. The first element of the input
    must be NaN. Example:

    .. code-block:: python
        znumber_indexed_seq = (math.nan, 3.0, 0.5, 1.0)
        mapping = znumber_indexed_seq_to_mapping(znumber_indexed_seq)
        # mapping will be {"H": 3.0, "Li": 1.0, "He": 0.5 }
    """
    if not math.isnan(seq[0]):
        raise ValueError("The first element of the input iterable must be NaN")
    return {PERIODIC_TABLE[j]: v for j, v in enumerate(seq) if j != 0}


# Create convenience tuples
MASS = mapping_to_znumber_indexed_seq(ATOMIC_MASS)
ELECTRONEGATIVITY = mapping_to_znumber_indexed_seq(ATOMIC_ELECTRONEGATIVITY)
HARDNESS = mapping_to_znumber_indexed_seq(ATOMIC_HARDNESS)
