r"""Atomic constants, and Density Functional constants

If you use these constants in your work please cite the corresponding article(s).

Values for electronegativity and hardness for elements H-Bk, all for neutral atoms, and
are taken from Table 3 of

Carlos Cardenas et. al. Benchmark Values of Chemical Potential and Chemical Hardness for
Atoms and Atomic Ions (Including Unstable Ions) from the Energies of Isoelectronic
Series.

DOI: 10.1039/C6CP04533B

Atomic masses supported are the first 119 elements (consistent with ASE) are taken from:

Atomic weights of the elements 2013 (IUPAC Technical Report). Meija, J., Coplen, T.,
Berglund, M., et al. (2016). Pure and Applied Chemistry, 88(3), pp. 265-291. Retrieved
30 Nov. 2016, from doi:10.1515/pac-2015-0305

DFT-D3(BJ) constants for different density functionals taken from the psi4 source code,
citations: A. Najib, L. Goerigk, J. Comput. Theory Chem., 14 5725, 2018) N.
Mardirossian, M. Head-Gordon, Phys. Chem. Chem. Phys, 16, 9904, 2014

Except for B97-3c taken from https://aip.scitation.org/doi/pdf/10.1063/1.5012601

And for wB97X taken from TODO: where?

Covalent radii are in angstroms, and are are used for the calculation of coordination
numbers in DR. Taken directly from Grimme et. al. dftd3 source code, in turn taken from
Pyykko and Atsumi, Chem. Eur. J. 15, 2009, 188-197 Values for metals decreased by 10 %
with respect to Pyykko et. al.

Empirical Q in atomic units, correspond to sqrt(0.5 * sqrt(Z) * <r**2>/<r**4>). In
Grimme's code these are "r2r4", and are used to calculate the C8 values. These are
precalculated values. For details on their calculation consult the DFT-D3 papers

XTB repulsion data extracted from Grimme et. al. paper
https://pubs.acs.org/doi/10.1021/acs.jctc.8b01176
"""

import json
import typing as tp
import math

from torchani.paths import _resources_dir

__all__ = [
    "ATOMIC_CONSTANTS",
    "ATOMIC_NUMBER",
    "ATOMIC_MASS",
    "ATOMIC_HARDNESS",
    "ATOMIC_COVALENT_RADIUS",
    "ATOMIC_SQRT_EMPIRICAL_CHARGE",
    "ATOMIC_ELECTRONEGATIVITY",
    "ATOMIC_XTB_REPULSION_ALPHA",
    "ATOMIC_XTB_REPULSION_YEFF",
    "MASS",
    "XTB_REPULSION_ALPHA",
    "XTB_REPULSION_YEFF",
    "COVALENT_RADIUS",
    "SQRT_EMPIRICAL_CHARGE",
    "HARDNESS",
    "ELECTRONEGATIVITY",
    "PERIODIC_TABLE",
    "FUNCTIONAL_D3BJ_CONSTANTS",
    "GSAES",
]

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

with open(_resources_dir() / "atomic_constants.json", mode="rt") as f:
    ATOMIC_CONSTANTS = json.load(f)

with open(_resources_dir() / "functional_d3bj_constants.json", mode="rt") as f:
    FUNCTIONAL_D3BJ_CONSTANTS = json.load(f)


# Populate convenience variables here
ATOMIC_NUMBER: tp.Dict[str, int] = {}
ATOMIC_HARDNESS: tp.Dict[str, float] = {}
ATOMIC_ELECTRONEGATIVITY: tp.Dict[str, float] = {}
ATOMIC_MASS: tp.Dict[str, float] = {}
ATOMIC_SQRT_EMPIRICAL_CHARGE: tp.Dict[str, float] = {}
ATOMIC_COVALENT_RADIUS: tp.Dict[str, float] = {}
ATOMIC_XTB_REPULSION_ALPHA: tp.Dict[str, float] = {}
ATOMIC_XTB_REPULSION_YEFF: tp.Dict[str, float] = {}


for symbol, values in ATOMIC_CONSTANTS.items():
    if not symbol:
        continue
    znumber = values.get("znumber")
    hardness = values.get("hardness")
    electroneg = values.get("electronegativity")
    mass = values.get("mass")
    sqrt_empirical_charge = values.get("sqrt_empirical_charge")
    covalent_radius = values.get("covalent_radius")
    alpha = values.get("xtb_repulsion_alpha")
    yeff = values.get("xtb_repulsion_yeff")
    if znumber is not None:
        ATOMIC_NUMBER[symbol] = int(znumber)
    if hardness is not None:
        ATOMIC_HARDNESS[symbol] = float(hardness)
    if electroneg is not None:
        ATOMIC_ELECTRONEGATIVITY[symbol] = float(electroneg)
    if mass is not None:
        ATOMIC_MASS[symbol] = float(mass)
    if covalent_radius is not None:
        ATOMIC_COVALENT_RADIUS[symbol] = float(covalent_radius)
    if sqrt_empirical_charge is not None:
        ATOMIC_SQRT_EMPIRICAL_CHARGE[symbol] = float(sqrt_empirical_charge)
    if alpha is not None:
        ATOMIC_XTB_REPULSION_ALPHA[symbol] = float(alpha)
    if yeff is not None:
        ATOMIC_XTB_REPULSION_YEFF[symbol] = float(yeff)

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
COVALENT_RADIUS = mapping_to_znumber_indexed_seq(ATOMIC_COVALENT_RADIUS)
SQRT_EMPIRICAL_CHARGE = mapping_to_znumber_indexed_seq(ATOMIC_SQRT_EMPIRICAL_CHARGE)
XTB_REPULSION_ALPHA = mapping_to_znumber_indexed_seq(ATOMIC_XTB_REPULSION_ALPHA)
XTB_REPULSION_YEFF = mapping_to_znumber_indexed_seq(ATOMIC_XTB_REPULSION_YEFF)
