r"""Fetch the dispersion constants from .pkl and .csv files, and provide them as Tensor

There are 4 different kinds of constants needed for D3 dispersion:

- Precalculated C6 coefficients
    shape (Elements, Elements, Ref, Ref), where "Ref" is the number of references
    (Grimme et. al. provides 5)
    This means for each pair of elements and reference indices there is an
    associated precalc C6 coeff
- Precalculated coordination numbers
    shape (Elements, Elements, Ref, Ref, 2)
    Where the final axis indexes the coordination number of the first or second
    atom respectively.
    This means for each pair of elements and reference indices there is an
    associated coordination number for the first and second items.
"""
import typing as tp
import math
import pickle
from pathlib import Path
from collections import defaultdict

import torch
from torch import Tensor

SUPPORTED_D3_ELEMENTS = 94


def _make_symmetric(x: Tensor) -> Tensor:
    assert x.ndim == 1
    size = (math.sqrt(1 + 8 * len(x)) - 1) / 2
    if not size.is_integer():
        raise ValueError(
            "Input tensor must be of size x * (x + 1) / 2 where x is an integer"
        )
    size = int(size)
    x_symmetric = torch.zeros((size, size))
    _lower_diagonal_mask = torch.tril(torch.ones((size, size), dtype=torch.bool))
    x_symmetric.masked_scatter_(_lower_diagonal_mask, x)
    for j in range(size):
        for i in range(size):
            x_symmetric[j, i] = x_symmetric[i, j]
    return x_symmetric


def _decode_atomic_numbers(a: int, b: int) -> tp.Tuple[int, int, int, int]:
    # Translated from Grimme et. al. Fortran code this is "limit" in Fortran
    # a_ref and b_ref give the conformation's ref (?) if a or b are greater
    # than 100 this means the conformation ref has to be moved by +1 an easier
    # way to do this is with divmod
    a_ref, a = divmod(a, 100)
    b_ref, b = divmod(b, 100)
    return a, b, a_ref, b_ref


def get_c6_constants() -> tp.Tuple[Tensor, Tensor, Tensor]:
    # Hardcoded in Grimme's et. al. D3 Fortran code
    total_records = 161925
    num_lines = 32385
    records_per_line = 5
    max_refs = 5
    path = Path(__file__).parent.joinpath('c6_unraveled.pkl').resolve()
    with open(path, 'rb') as f:
        c6_unraveled = pickle.load(f)
        c6_unraveled = torch.tensor(c6_unraveled).reshape(-1, records_per_line)
    assert c6_unraveled.numel() == total_records
    assert c6_unraveled.shape[0] == num_lines

    # Element 0 is actually a dummy element
    el = SUPPORTED_D3_ELEMENTS
    # nonexistent values are filled with -1, in order to mask them,
    # same as in Grimme et. al. code
    c6_constants = torch.full((el + 1, el + 1, max_refs, max_refs), -1.0)
    c6_coordination_a = torch.full((el + 1, el + 1, max_refs, max_refs), -1.0)
    c6_coordination_b = torch.full((el + 1, el + 1, max_refs, max_refs), -1.0)
    if not ((c6_constants == -1.0) == (c6_coordination_a == -1.0)).all():
        raise RuntimeError("All missing parameters are not equal")
    if not ((c6_coordination_a == -1.0) == (c6_coordination_b == -1.0)).all():
        raise RuntimeError("All missing parameters are not equal")

    # every "line" in the unraveled c6 list has:
    # 0 1 2 3 4
    # C6, a, b, CNa, CNb
    # in that order
    # translated from Grimme et. al. Fortran code
    for line in c6_unraveled:
        constant, a, b, cn_a, cn_b = line.cpu().numpy().tolist()
        # a and b are the atomic numbers
        a, b, a_ref, b_ref = _decode_atomic_numbers(int(a), int(b))
        # get values for C6 and CNa, CNb
        c6_constants[a, b, a_ref, b_ref] = constant
        c6_coordination_a[a, b, a_ref, b_ref] = cn_a
        c6_coordination_b[a, b, a_ref, b_ref] = cn_b
        # symmetrize values
        c6_constants[b, a, b_ref, a_ref] = constant
        # these have to be inverted (cn_a given to b and cn_b given to a)
        c6_coordination_a[b, a, b_ref, a_ref] = cn_b
        c6_coordination_b[b, a, b_ref, a_ref] = cn_a
    return c6_constants, c6_coordination_a, c6_coordination_b


def get_cutoff_radii() -> Tensor:
    # cutoff radii are in angstroms
    num_cutoff_radii = SUPPORTED_D3_ELEMENTS * (SUPPORTED_D3_ELEMENTS + 1) / 2
    path = Path(__file__).parent.joinpath('cutoff_radii.pkl').resolve()
    with open(path, 'rb') as f:
        cutoff_radii = torch.tensor(pickle.load(f))
    assert len(cutoff_radii) == num_cutoff_radii
    cutoff_radii = _make_symmetric(cutoff_radii)
    cutoff_radii = torch.cat(
        (
            torch.zeros(len(cutoff_radii), dtype=cutoff_radii.dtype).unsqueeze(0),
            cutoff_radii,
        ),
        dim=0,
    )
    cutoff_radii = torch.cat(
        (
            torch.zeros(cutoff_radii.shape[0], dtype=cutoff_radii.dtype).unsqueeze(1),
            cutoff_radii,
        ),
        dim=1,
    )
    return cutoff_radii


def get_covalent_radii() -> Tensor:
    # covalent radii are in angstroms covalent radii are used for the
    # calculation of coordination numbers covalent radii in angstrom taken
    # directly from Grimme et. al. dftd3 source code, in turn taken from Pyykko
    # and Atsumi, Chem. Eur. J. 15, 2009, 188-197 values for metals decreased
    # by 10 %
    path = Path(__file__).parent.joinpath('covalent_radii.pkl').resolve()
    with open(path, 'rb') as f:
        covalent_radii = torch.tensor(pickle.load(f))
    assert len(covalent_radii) == SUPPORTED_D3_ELEMENTS
    # element 0 is a dummy element
    covalent_radii = torch.cat((torch.tensor([0.0]), covalent_radii))
    return covalent_radii


def get_sqrt_empirical_charge() -> Tensor:
    # empirical Q is in atomic units, these correspond to sqrt(0.5 * sqrt(Z) *
    # <r**2>/<r**4>) in Grimme's code these are "r2r4", and are used to
    # calculate the C8 values
    path = Path(__file__).parent.joinpath('sqrt_empirical_charge.pkl').resolve()
    with open(path, 'rb') as f:
        sqrt_empirical_charge = torch.tensor(pickle.load(f))
    assert len(sqrt_empirical_charge) == SUPPORTED_D3_ELEMENTS
    # element 0 is a dummy element
    sqrt_empirical_charge = torch.cat((torch.tensor([0.0]), sqrt_empirical_charge))
    return sqrt_empirical_charge


def get_functional_constants() -> tp.Dict[str, tp.Dict[str, float]]:
    # constants for the density functional from psi4 source code, citations:
    #    A. Najib, L. Goerigk, J. Comput. Theory Chem., 14 5725, 2018)
    #    N. Mardirossian, M. Head-Gordon, Phys. Chem. Chem. Phys, 16, 9904, 2014
    df_constants: tp.Dict[str, tp.Dict[str, float]] = defaultdict(dict)
    # TODO: check where wB97X actually comes from
    df_constants['wb97x'] = {
        's6_bj': 1.000,
        'a1': 0.0000,
        's8_bj': 0.2641,
        'a2': 5.4959,
    }
    # from Grimme's et al website directly:
    # first D3Zero constants
    # functional, s6_zero, sr6, s8_zero,
    # B97D == B97-D and B973c == B97-3c
    _zero_constants_str = """B1B95      1.0     1.613   1.868
                         B2GPPLYP   0.56    1.586   0.760
                         B3LYP      1.0     1.261   1.703
                         B97-D      1.0     0.892   0.909
                         B97D       1.0     0.892   0.909
                         BHLYP      1.0     1.370   1.442
                         BLYP       1.0     1.094   1.682
                         BP86       1.0     1.139   1.683
                         BPBE       1.0     1.087   2.033
                         mPWLYP     1.0     1.239   1.098
                         PBE        1.0     1.217   0.722
                         PBE0       1.0     1.287   0.928
                         PW6B95     1.0     1.532   0.862
                         PWB6K      1.0     1.660   0.550
                         revPBE     1.0     0.923   1.010
                         TPSS       1.0     1.166   1.105
                         TPSS0      1.0     1.252   1.242
                         TPSSh      1.0     1.223   1.219
                         BOP        1.0     0.929   1.975
                         MPW1B95    1.0     1.605   1.118
                         MPWB1K     1.0     1.671   1.061
                         OLYP       1.0     0.806   1.764
                         OPBE       1.0     0.837   2.055
                         oTPSS      1.0     1.128   1.494
                         PBE38      1.0     1.333   0.998
                         PBEsol     1.0     1.345   0.612
                         REVSSB     1.0     1.221   0.560
                         SSB        1.0     1.215   0.663
                         B3PW91     1.0     1.176   1.775
                         BMK        1.0     1.931   2.168
                         CAMB3LYP   1.0     1.378   1.217
                         LCwPBE     1.0     1.355   1.279
                         M052X      1.0     1.417   0.00
                         M05        1.0     1.373   0.595
                         M062X      1.0     1.619   0.00
                         M06HF      1.0     1.446   0.00
                         M06L       1.0     1.581   0.00
                         M06        1.0     1.325   0.00
                         HCTH120    1.0     1.221   1.206
                         B2PLYP     0.64    1.427   1.022
                         DSDBLYP    0.50    1.569   0.705
                         PTPSS      0.75    1.541   0.879
                         PWPB95     0.82    1.557   0.705
                         revPBE0    1.0     0.949   0.792
                         revPBE38   1.0     1.021   0.862
                         rPW86PBE   1.0     1.224   0.901
                         B97-3c     1.0     1.060   1.500
                         B973c      1.0     1.060   1.500"""
    # Parameters for B97-3c taken from
    # https://aip.scitation.org/doi/pdf/10.1063/1.5012601
    _zero_constants = _zero_constants_str.split('\n')
    for line in _zero_constants:
        df, s6_zero, sr6, s8_zero = line.split()
        df_constants[df.lower()] = {
            's6_zero': float(s6_zero),
            'sr6': float(sr6),
            's8_zero': float(s8_zero)
        }
    # now D3BJ constants
    # functional, s6_bj, a1, s8_bj, a2
    _bj_constants_str = """B1B95         1.000   0.2092    1.4507    5.5545
                       B2GPPLYP      0.560   0.0000    0.2597    6.3332
                       B3PW91        1.000   0.4312    2.8524    4.4693
                       BHLYP         1.000   0.2793    1.0354    4.9615
                       BMK           1.000   0.1940    2.0860    5.9197
                       BOP           1.000   0.4870    3.295     3.5043
                       BPBE          1.000   0.4567    4.0728    4.3908
                       CAMB3LYP      1.000   0.3708    2.0674    5.4743
                       LCwPBE        1.000   0.3919    1.8541    5.0897
                       MPW1B95       1.000   0.1955    1.0508    6.4177
                       MPWB1K        1.000   0.1474    0.9499    6.6223
                       mPWLYP        1.000   0.4831    2.0077    4.5323
                       OLYP          1.000   0.5299    2.6205    2.8065
                       OPBE          1.000   0.5512    3.3816    2.9444
                       oTPSS         1.000   0.4634    2.7495    4.3153
                       PBE38         1.000   0.3995    1.4623    5.1405
                       PBEsol        1.000   0.4466    2.9491    6.1742
                       PTPSS         0.750   0.000     0.2804    6.5745
                       PWB6K         1.000   0.1805    0.9383    7.7627
                       revSSB        1.000   0.4720    0.4389    4.0986
                       SSB           1.000   -0.0952   -0.1744   5.2170
                       TPSSh         1.000   0.4529    2.2382    4.6550
                       HCTH120       1.000   0.3563    1.0821    4.3359
                       B2PLYP        0.640   0.3065    0.9147    5.0570
                       B3LYP         1.000   0.3981    1.9889    4.4211
                       B97D          1.000   0.5545    2.2609    3.2297
                       BLYP          1.000   0.4298    2.6996    4.2359
                       BP86          1.000   0.3946    3.2822    4.8516
                       DSDBLYP       0.500   0.000     0.2130    6.0519
                       PBE0          1.000   0.4145    1.2177    4.8593
                       PBE           1.000   0.4289    0.7875    4.4407
                       PW6B95        1.000   0.2076    0.7257    6.3750
                       PWPB95        0.820   0.0000    0.2904    7.3141
                       revPBE0       1.000   0.4679    1.7588    3.7619
                       revPBE38      1.000   0.4309    1.4760    3.9446
                       revPBE        1.000   0.5238    2.3550    3.5016
                       rPW86PBE      1.000   0.4613    1.3845    4.5062
                       TPSS0         1.000   0.3768    1.2576    4.5865
                       TPSS          1.000   0.4535    1.9435    4.4752
                       B97-3c        1.000   0.3700    1.5000    4.1000
                       B973c         1.000   0.3700    1.5000    4.1000"""
    # Parameters for B97-3c taken from
    # https://aip.scitation.org/doi/pdf/10.1063/1.5012601
    # Other parameters taken directly from the Psi4 source code
    _bj_constants = _bj_constants_str.split('\n')
    for line in _bj_constants:
        df, s6_bj, a1, s8_bj, a2 = line.split()
        df_constants[df.lower()].update(
            {
                's6_bj': float(s6_bj),
                'a1': float(a1),
                's8_bj': float(s8_bj),
                'a2': float(a2)
            }
        )
    return df_constants
