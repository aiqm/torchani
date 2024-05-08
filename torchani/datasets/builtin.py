r"""Torchani Builtin Datasets

This module provides access to the following datasets, calculated with specific
levels of theory (LoT) which are combinations functional/basis_set or
wavefunction_method/basis_set when appropriate.

- ANI-1x, with LoT:
    - wB97X/6-31G(d)
    - wB97X/def2-TZVPP
    - B97-3c/def2-mTZVP
    - wB97M-D3BJ/def2-TZVPP
    - wB97MV/def2-TZVPP

- ANI-2x, with LoT:
    - wB97X/6-31G(d)
    - wB97X/def2-TZVPP
    - B97-3c/def2-mTZVP
    - wB97M-D3BJ/def2-TZVPP
    - wB97MV/def2-TZVPP

- ANI-1ccx, with LoT:
    - CCSD(T)star/CBS
  Note that this dataset also has Hartree Fock (HF) energies, RI-MP2 energies
  and forces and DPLNO-CCSD(T) energies for different basis sets and PNO
  settings.
  This dataset was originally used for transfer learning, not direct training.

- ANI-1e, with LoT:
    - wB97X/6-31G(d)
    This dataset consists of structures corresponding to all smiles extracted
    from the ANI-1x dataset, embedded in 3D space and optimized with PM7.
    The ANI-1e dataset is presented in
    `ANI-1E: An equilibrium database from the ANI-1 database`
    by Vazquez-Salazar, Luis Itza and Meuwly, Markus.
    This dataset does not have forces, it has instead a variety of properties that
    the QM9-style datasets have:
        - Rotational constants A, B, C (GHz)
        - Dipole and Polarizability magnitudes (Debye and a_0^3 respectively)
        - Energy of HOMO and LUMO, and HOMO-LUMO gap (Ha)
        - average <r^2> (spatial extent, a_0^2)
        - Zero point vibrational energies (ZPVE, Ha)
        - Zero Kelvin internal energy (Ha)
        - Thermal quantities, U, H, G, C_v, at 298.15 K (C_v in cal/K/mol, rest in Ha)

- COMP6-v1, with LoT:
    - wB97X/6-31G(d)
    - wB97X/def2-TZVPP
    - B97-3c/def2-mTZVP
    - wB97M-D3BJ/def2-TZVPP
    - wB97MV/def2-TZVPP
  This dataset is not meant to be trained to, it is a test set.

- COMP6-v2, with LoT:
    - wB97X/6-31G(d)
    - B97-3c/def2-mTZVP
    - wB97M-D3BJ/def2-TZVPP
    - wB97MV/def2-TZVPP
  This dataset is not meant to be trained to, it is a test set.

- AminoacidDimers, with LoT:
    - B97-3c/def2-mTZVP
  This dataset is not meant to be trained to on its own.

- ANI1q, with LoT:
    - wB97X/631G(d)
  Very limited subset of ANI-1x
  for which 'atomic CM5 charges' are available.
  This dataset is not meant to be trained to on its own.

- ANI2qHeavy, with LoT:
    - wB97X/631Gd
  Subset of ANI-2x "heavy"
  for which 'atomic CM5 charges' are available.
  This dataset is not meant to be trained to on its own.

- IonsLight, with LoT:
    - B973c/def2-mTZVP
  Dataset that includes ions, with H,C,N,O elements only
  This dataset is not meant to be trained to on its own.

- IonsHeavy, with LoT:
    - B973c/def2mTZVP
  Dataset that includes ions, with H,C,N,O elements and at least one of F,S,Cl
  (disjoint from IonsLight)
  This dataset is not meant to be trained to on its own.

- IonsVeryHeavy, with LoT:
    - B973c/def2-mTZVP
  Dataset that includes ions, with H,C,N,O,F,S,Cl elements and at least one of
  Si,As,Br,Se,P,B,I
  (disjoint from LightIons and IonsHeavy)
  This dataset is not meant to be trained to on its own.

- ANI-CCScan, with LoT:
    - B973c/def2-mTZVP
    - UwB97x/6-31Gd
  Dataset includes relaxed scans of stretching carbon-carbon single bonds
  for a number of compounds. UwB97x means "Unrestricted" Kohn-Sham was used.

- DielsAlder, with LoT:
    - UwB97x/6-31Gd
  Dataset includes structures sampled in different ways from a diels alder
  reaction. Among them, from the intrinsic reaction coordinate (IRC), and from
  specific points of the potential energy surface (PES), in some cases using
  active learning, with the QBC criteria, and with normal mode sampling.
  UwB97x means "Unrestricted" Kohn-Sham was used.

- SolvatedProteinFragments, with LoT:
    - revPBE-D3(BJ)/def2-TZVP
    The solvated protein fragments dataset probes many-body intermolecular
    interactions between "protein fragments" and water molecules, which are
    important for the description of many biologically relevant condensed phase
    systems. It contains structures for all possible "amons"
    (hydrogen-saturated covalently bonded fragments) of up to eight heavy atoms
    (C, N, O, S) that can be derived from chemical graphs of proteins
    containing the 20 natural amino acids connected via peptide bonds or
    disulfide bridges.
    Note that molecules in this dataset may have charges.
    dataset presented in the physnet paper:
    https://arxiv.org/abs/1902.08408 (2019)

- TestData, with LoT:
    - wB97X/631Gd
  GDB subset, only for debugging and code testing purposes.

- TestDataIons, with LoT:
    - B973c/def2mTZVP
  Only for debugging and code testing purposes, includes forces, dipoles and charges.

- TestDataForcesDipoles, with LoT:
    - B973c/def2mTZVP
  Only for debugging and code testing purposes, includes forces and dipoles.

- QM9Dirty, with LoT:
    - B3LYP/631G_2df_p
  QM9 dataset, as published in  https://doi.org/10.1038/sdata.2014.22,
  The dirty version of the dataset contains the full dataset including the structures
  that "failed geometry consistensy checks". This dataset does not contain forces,
  but contains various physical quantities.
  Vibrational frequencies are included, and are always 3 A - 5
  Molecules have a "is_linear" boolean flag, for non linear molecules the last
  vibrational frequency is 0.0 and should be discarded.
  Some molecules have a "has_alternative_frequencies" flag. These molecules have
  two sets of vibrational frequencies. Molecules that don't have this second set have
  zeros for this field.

- QM9Clean, with LoT:
    - B973c/def2mTZVP
  Clean version of the QM9 dataset, which does not include the structures that
  failed geometry consistency checks.

- QM9C7O2H10, with LoT:
    - G4MP2/631G_2df_p
  Subset of QM9 with thermal properties recalculated at the higher G4MP2 level
  of theory. *Note that the rest of the properties in this dataset are
  calculated at the B3LYP/631G_2df_p level of theory*. Properties correspond
  only to conformations of the stoichiometry C7O2H10.

SPICE dataset:
    All sets with LoT:
    - wB97M-D3(BJ)/def2-TZVPPD

    From paper:

    SPICE, A Dataset of Drug-like Molecules and Peptides for Training Machine
    Learning Potentials." https://doi.org/10.48550/arXiv.2209.10702 (2022).

    For more information consult the relevant paper. Note that some molecules
    in these datasets contain ions or counterions, but there are no structures
    with net charge.

    NOTE: Some of the fields in these sets are missing for some conformations,
    namely, the MBIS fields and bond-indices are missing for IonPairs and Des370K,
    and the bond-indices fields only are missing for the PubChem data (except the
    subset "6")

    - SPICEPubChem:
        With elements:
            H, C, N, O, S, F, Cl, Br, I, P

    - SPICEPubChem2xCompatible:
        Subset of SPICEPubChem that contains only H, C, N, O, S, F, Cl elements.

    - SPICEIonPairs:
        With ions:
            Li+, Na+, K+, Ca2+, Mg2+, F-, Cl-, Br-, I-

    - SPICEDipeptides:
        With elements:
            H, C, N, O, S

    - SPICESolvatedAminoacids:
        With elements:
            H, C, N, O, S

    - SPICEDesMonomers:
        With elements:
            H, C, N, O, S, F, Cl, Br, I, P

    - SPICEDes370K:
        With elements:
            H, C, N, O, S, F, Cl, Br, I, P

        And ions:
            Li+, Na+, K+, Ca2+, Mg2+

Iso17 dataset:
    All sets with LoT:
    - PBE-TS/Numerical-FHI-aims

    MD trajectories using FHI-aims, with a resolution of 1 frame / fs. . Dataset
    has total energies and atomic forces.

    From paper:

    K.T. Schütt, P.-J. Kindermans, H.E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R.
    Müller. SchNet: A continuous-filter convolutional neural network for modeling
    quantum interactions. Advances in Neural Information Processing System. 2017.

    - Iso17TrainSet1:
        Training set of trajectories of molecules with C7O2H10 stoichiometry

    - Iso17TestSet1:
        Testing set of trajectories of molecules with C7O2H10 stoichiometry,
        same molecules as TrainSet1

    - Iso17EquilibriumSet1:
        Equilibrium geometries of molecules with C7O2H10 stoichiometry,
        same molecules as TrainSet1

    - Iso17TestSet2:
        Testing set of trajectories of molecules with C7O2H10 stoichiometry,
        different molecules from TrainSet1

    - Iso17EquilibriumSet2:
        Equilibrium geometries of molecules with C7O2H10 stoichiometry,
        same molecules as TestSet2

- SN2, with LoT:
    - DSD-BLYP-D3(BJ)/def2mTZVP
    Dataset from the PhysNet paper
    Unke, O. T. and Meuwly, M. "PhysNet: A Neural Network for Predicting
    Energies, Forces, Dipole Moments and Partial Charges" arxiv:1902.08408
    (2019).
    Used to test accuracy in SN2 type reactions with charges
    CH3X + X'-- -> CH3X' + X-
    Note that this dataset has charged species, and contains all possible structures
    that can be formed in fragmentation reactions:
    H3X, HX, CHX or CH2X- as well as geometries for H2, CH2, CH3+ and XY
    All possible combinations are included.
    It is unclear what the reference point for the dipoles in the dataset are,
    the units are eA, the reference is most likely the center-of-charge.

- 3BPA dataset, with LoT:
    - wB97X/6-31G(d)
    Dataset from the ACE paper (Peter Kovacs et. al.
    https://doi.org/10.1021/acs.jctc.1c00647)

    This dataset consists on 8 different parts, two meant for training and 6 for
    testing / validation

    - Train3BPA300K
    - Train3BPAMixedT
    - Test3BPA300K
    - Test3BPA600K
    - Test3BPA1200K
    - Test3BPADihedral120
    - Test3BPADihedral150
    - Test3BPADihedral180

    For more information consult the corresponding paper

- ANI-ExCorr, with LoT:
    - PBE/DZVP

    This dataset has "coefficients" which correspond to atomic
    coefficients for the fitting density, and "energies-xc", the
    exchange-correlation energies.


Note that the conformations present in datasets with different LoT may be
different.

In all cases the "v2" and "2x" datasets are supersets of the "v1" and "1x"
datasets, so everything that is in the v1/1x datasets is also in the v2/2x
datasets, which contain extra structures.

There is some extra wb97X/def2-TZVPP data for which there are "v1" values but not
"v2" values.

Note that the ANI-BenchMD, S66x8 and the "13" molecules (with 13 heavy atoms)
of GDB-10to13 were recalculated using ORCA 5.0 instead of 4.2 so the
integration grids may be slightly different, but the difference should not be
significant.
"""
import typing as tp
import sys
from pathlib import Path
from collections import OrderedDict

import yaml
from tqdm import tqdm

from torchani.datasets.download import _download_and_extract_archive, _check_integrity
from torchani.datasets.datasets import ANIDataset
from torchani.datasets._annotations import StrPath
from torchani.storage import DATASETS_DIR

_BASE_URL = 'http://moria.chem.ufl.edu/animodel/ground_truth_data/'
_DATASETS_YAML_PATH = Path(__file__).parent / "builtin-datasets.yaml"

with open(_DATASETS_YAML_PATH, mode="rt", encoding="utf-8") as f:
    _BUILTIN_DATASETS_SPEC = yaml.safe_load(f)

# Convert csv file with format "file_name, MD5-hash" into a dictionary
_MD5S: tp.Dict[str, str] = dict()
with open(Path(__file__).resolve().parent / "md5s.csv") as f:
    lines = f.readlines()
    for line in lines[1:]:
        file_, md5 = line.split(',')
        _MD5S[file_.strip()] = md5.strip()

_BUILTIN_DATASETS: tp.List[str] = list(_BUILTIN_DATASETS_SPEC.keys())
_BUILTIN_DATASETS_LOT: tp.List[str] = list(
    {
        lot for name in _BUILTIN_DATASETS_SPEC.keys()
        for lot in _BUILTIN_DATASETS_SPEC[name]["lot"]
    }
)


def download_builtin_dataset(dataset: str, lot: str, root=None):
    """
    Download dataset at specified root folder, or at the default folder:
    ./datasets/{dataset}-{lot}/
    """
    assert dataset in _BUILTIN_DATASETS, f"{dataset} is not avaiable"
    assert lot in _BUILTIN_DATASETS_LOT, f"{lot} is not avaiable"

    parts = lot.split('-')
    assert len(parts) == 2, f"bad LoT format: {lot}"

    functional = parts[0]
    basis_set = parts[1]
    location = f'./datasets/{dataset}-{lot}/' if root is None else root
    if Path(location).exists():
        print(
            f"Found existing dataset at {Path(location).absolute().as_posix()},"
            f" will check files integrality."
        )
    else:
        print(f"Will download dataset at {Path(location).absolute().as_posix()}")
    getattr(sys.modules[__name__], dataset)(
        location,
        download=True,
        functional=functional,
        basis_set=basis_set
    )


def _check_files_integrity(
    files_and_md5s: tp.OrderedDict[str, str],
    root: Path,
    suffix: str = ".h5",
    name: str = "Dataset",
    skip_hash_check: bool = False,
) -> None:
    # Checks that:
    # (1) There are files in the dataset
    # (2) All file names in the provided path are equal to the expected ones
    # (3) They have the correct checksum
    # If any of these conditions fails the function exits with a RuntimeError
    # other files such as tar.gz archives are neglected
    present_files = sorted(root.glob(f"*{suffix}"))
    expected_file_names = set(files_and_md5s.keys())
    present_file_names = set([f.name for f in present_files])
    if not present_files:
        raise RuntimeError(f'Dataset not found in path {str(root)}')
    if expected_file_names != present_file_names:
        raise RuntimeError(
            f"Wrong files found for dataset {name} in provided path,"
            f" expected {expected_file_names} but found {present_file_names}"
        )
    if skip_hash_check:
        return
    for f in tqdm(present_files, desc=f'Checking integrity of dataset {name}'):
        if not _check_integrity(f, files_and_md5s[f.name]):
            raise RuntimeError(
                f"All expected files for dataset {name}"
                f" were found but file {f.name} failed integrity check,"
                " your dataset is corrupted or has been modified"
            )


# This Function is a Builder Factory that creates builder functions
# that instantiate builtin ani datasets.
# Functions are created using a yaml file as a template, their names are:
#   - COMP6V1
#   - ANI1x
#   - ANI2x
#   ...
# Options for the builder functions are:
#   - root
#   - functional
#   - basis_set
#   - verbose
#   - download
#   - dummy_properties
def _register_dataset_builder(name: str) -> None:
    data = _BUILTIN_DATASETS_SPEC[name]["lot"]
    default_fn, default_basis = _BUILTIN_DATASETS_SPEC[name]["default-lot"].split("-")

    def builder(
        root: tp.Optional[StrPath] = None,
        functional: str = default_fn,
        basis_set: str = default_basis,
        verbose: bool = True,
        download: bool = True,
        dummy_properties: tp.Optional[tp.Dict[str, tp.Any]] = None,
        skip_check: bool = False,
    ) -> ANIDataset:
        lot = f"{functional}-{basis_set}".lower()
        try:
            archive = data[lot]["archive"]
        except KeyError:
            raise ValueError(
                f"Unsupported functional-basis set combination"
                f" try one of {set(data.keys()) - {'default-lot'}}"
            ) from None
        suffix = ".h5"

        _root = root or DATASETS_DIR / archive.replace(".tar.gz", "")
        _root = Path(_root).resolve()

        _files_and_md5s = OrderedDict([(k, _MD5S[k]) for k in data[lot]["files"]])

        # If the dataset is not found we download it
        if download and ((not _root.is_dir()) or (not any(_root.glob(f"*{suffix}")))):
            _download_and_extract_archive(
                base_url=_BASE_URL,
                file_name=archive,
                dest_dir=_root
            )

        # Check for corruption
        _check_files_integrity(_files_and_md5s, _root, suffix, name, skip_hash_check=skip_check)

        # Order dataset paths using the order given in "files and md5s"
        filenames_order = {
            Path(k).stem: j for j, k in enumerate(_files_and_md5s.keys())
        }
        _filenames_and_paths = sorted(
            [(p.stem, p) for p in sorted(_root.glob(f"*{suffix}"))],
            key=lambda tup: filenames_order[tup[0]]
        )
        filenames_and_paths = OrderedDict(_filenames_and_paths)
        ds = ANIDataset(
            locations=filenames_and_paths.values(),
            names=filenames_and_paths.keys(),
            verbose=verbose,
            dummy_properties=dummy_properties,
        )
        if verbose:
            print(ds)
        return ds

    builder.__name__ = name
    setattr(sys.modules[__name__], name, builder)


for name in _BUILTIN_DATASETS:
    if name not in sys.modules[__name__].__dict__:
        _register_dataset_builder(name)
