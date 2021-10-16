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
    - wB97X/631Gd
  Very limited subset of ANI-1x
  for which 'atomic CM5 charges' are available.
  This dataset is not meant to be trained to on its own.

- ANI2qHeavy, with LoT:
    - wB97X/631Gd
  Subset of ANI-2x "heavy"
  for which 'atomic CM5 charges' are available.
  This dataset is not meant to be trained to on its own.

- IonsLight, with LoT:
    - B973c/def2mTZVP
  Dataset that includes ions, with H,C,N,O elements only
  This dataset is not meant to be trained to on its own.

- IonsHeavy, with LoT:
    - B973c/def2mTZVP
  Dataset that includes ions, with H,C,N,O elements and at least one of F,S,Cl
  (disjoint from IonsLight)
  This dataset is not meant to be trained to on its own.

- IonsVeryHeavy, with LoT:
    - B973c/def2mTZVP
  Dataset that includes ions, with H,C,N,O,F,S,Cl elements and at least one of
  Si,As,Br,Se,P,B,I
  (disjoint from LightIons and IonsHeavy)
  This dataset is not meant to be trained to on its own.

- TestData, with LoT:
    - wB97X/631Gd
  GDB subset, only for debugging and code testing purposes.

- TestDataIons, with LoT:
    - B973c/def2mTZVP
  Only for debugging and code testing purposes, includes forces, dipoles and charges.

- TestDataForcesDipoles, with LoT:
    - B973c/def2mTZVP
  Only for debugging and code testing purposes, includes forces and dipoles.


(note that the conformations present in datasets with different LoT may be
different).

In all cases the "v2" and "2x" datasets are supersets of the "v1" and "1x"
datasets, so everything that is in the v1/1x datasets is also in the v2/2x
datasets, which contain extra structures.
"""
import sys
from pathlib import Path
from typing import Optional, Any
from collections import OrderedDict
from copy import deepcopy

from torchvision.datasets.utils import download_and_extract_archive, list_files, check_integrity
from .datasets import ANIDataset
from ._annotations import StrPath
from ..utils import tqdm

_BASE_URL = 'http://moria.chem.ufl.edu/animodel/ground_truth_data/'
_DEFAULT_DATA_PATH = Path.home().joinpath('.local/torchani/Datasets')

_BUILTIN_DATASETS = ['ANI1x', 'ANI2x', 'COMP6v1', 'COMP6v2', 'ANI1ccx', 'AminoacidDimers',
                     'ANI1q', 'ANI2qHeavy', 'IonsLight', 'IonsHeavy',
                     'IonsVeryHeavy', 'TestData', 'TestDataIons', 'TestDataForcesDipoles']
_BUILTIN_DATASETS_LOT = ['wb97x-631gd', 'b973c-def2mtzvp', 'wb97md3bj-def2tzvpp', 'wb97mv-def2tzvpp', 'wb97x-def2tzvpp', 'ccsd(t)star-cbs']


def _read_md5_hashes():
    # This function reads a csv file with format "file_name MD5-hash"
    # and outputs a dictionary with all supported file names
    with open(Path(__file__).resolve().parent / "md5s.csv") as f:
        lines = f.readlines()
        _md5s = dict()
        for line in lines[1:]:
            file_, md5 = line.split(',')
            _md5s[file_.strip()] = md5.strip()
    return _md5s


_MD5S = _read_md5_hashes()


def download_builtin_dataset(dataset, lot, root=None):
    """
    Download dataset at specified root folder, or at the default folder: ./datasets/{dataset}-{lot}/
    """
    assert dataset in _BUILTIN_DATASETS, f"{dataset} is not avaiable"
    assert lot in _BUILTIN_DATASETS_LOT, f"{lot} is not avaiable"

    parts = lot.split('-')
    assert len(parts) == 2, f"bad LoT format: {lot}"

    functional = parts[0]
    basis_set = parts[1]
    location = f'./datasets/{dataset}-{lot}/' if root is None else root
    if Path(location).exists():
        print(f"Found existing dataset at {Path(location).absolute().as_posix()}, will check files integrality.")
    else:
        print(f"Will download dataset at {Path(location).absolute().as_posix()}")
    getattr(sys.modules[__name__], dataset)(location, download=True, functional=functional, basis_set=basis_set)


class _BaseBuiltinDataset(ANIDataset):
    # NOTE: Code heavily borrows from celeb dataset of torchvision

    def __init__(self, root: StrPath,
                       download: bool = False,
                       archive: Optional[str] = None,
                       files_and_md5s: Optional['OrderedDict[str, str]'] = None,
                       **h5_dataset_kwargs: Any):
        assert isinstance(files_and_md5s, OrderedDict)

        self._archive: str = '' if archive is None else archive
        self._files_and_md5s = OrderedDict([('', '')]) if files_and_md5s is None else files_and_md5s

        root = Path(root).resolve()
        if download:
            # If the dataset is not found we download it, if the dataset is corrupted we
            # exit with error
            self._maybe_download_hdf5_archive_and_check_integrity(root)
        else:
            self._check_hdf5_files_integrity(root)
        dataset_paths = [Path(p).resolve() for p in list_files(root, suffix='.h5', prefix=True)]

        # Order dataset paths using the order given in "files and md5s"
        filenames_order = {Path(k).stem: j for j, k in enumerate(self._files_and_md5s.keys())}
        _filenames_and_paths = sorted([(p.stem, p) for p in dataset_paths],
                                     key=lambda tup: filenames_order[tup[0]])
        filenames_and_paths = OrderedDict(_filenames_and_paths)
        super().__init__(locations=filenames_and_paths.values(), names=filenames_and_paths.keys(), **h5_dataset_kwargs)
        if h5_dataset_kwargs.get('verbose', True):
            print(self)

    def _check_hdf5_files_integrity(self, root: Path) -> None:
        # Checks that:
        # (1) There are HDF5 files in the dataset
        # (2) All HDF5 files names in the provided path are equal to the expected ones
        # (3) They have the correct checksum
        # If any of these conditions fails the function exits with a RuntimeError
        # other files such as tar.gz archives are neglected
        present_files = [Path(f).resolve() for f in list_files(root, suffix='.h5', prefix=True)]
        expected_file_names = set(self._files_and_md5s.keys())
        present_file_names = set([f.name for f in present_files])
        if not present_files:
            raise RuntimeError(f'Dataset not found in path {root.as_posix()}')
        if expected_file_names != present_file_names:
            raise RuntimeError(f"Wrong files found for dataset {self.__class__.__name__} in provided path, "
                               f"expected {expected_file_names} but found {present_file_names}")
        for f in tqdm(present_files, desc=f'Checking integrity of files for dataset {self.__class__.__name__}'):
            if not check_integrity(f, self._files_and_md5s[f.name]):
                raise RuntimeError(f"All expected files for dataset {self.__class__.__name__} "
                                   f"were found but file {f.name} failed integrity check, "
                                    "your dataset is corrupted or has been modified")

    def _maybe_download_hdf5_archive_and_check_integrity(self, root: Path) -> None:
        # Downloads only if the files have not been found,
        # If the files are corrupted it fails and asks you to delete them
        root = Path(root).resolve()
        if root.is_dir() and list(root.iterdir()):
            self._check_hdf5_files_integrity(root)
            return
        download_and_extract_archive(url=f'{_BASE_URL}{self._archive}', download_root=root, md5=None)
        tarfile = root / self._archive
        if tarfile.is_file():
            tarfile.unlink()
        self._check_hdf5_files_integrity(root)


class _ArchivedBuiltinDataset(_BaseBuiltinDataset):
    def __init__(self, files, basis_set, functional, root: StrPath = None, download: bool = False, verbose: bool = True, **kwargs):
        lot = f'{functional.lower()}-{basis_set.lower()}'
        if lot not in files.keys():
            raise ValueError(f"Unsupported functional-basis set combination, try one of {set(files.keys())}")
        archive = files[lot][0]
        files_and_md5s = OrderedDict([(k, _MD5S[k]) for k in files[lot][1]])
        if root is None:
            # the default name is the name of the tarred file, here we strip "tar" and "gz"
            root = _DEFAULT_DATA_PATH.joinpath(Path(archive).with_suffix('').with_suffix('').as_posix())
        super().__init__(root, download, archive=archive, files_and_md5s=files_and_md5s, verbose=verbose, **kwargs)


# NOTE: The order of the files is important since it deterimenes the order of iteration over the files
class TestData(_ArchivedBuiltinDataset):
    _FILES = {'wb97x-631gd': ('TestData-sample-wb97x-631gd.tar.gz',
                              ['test_data1.h5',
                               'test_data2.h5'])}

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X', **kwargs):
        super().__init__(self._FILES, basis_set, functional, root, download, verbose, **kwargs)


class TestDataIons(_ArchivedBuiltinDataset):
    _FILES = {'b973c-def2mtzvp': ('TestData-ions-b973c-def2mtzvp.tar.gz',
                                  ['ANI-1x_sample-B973c-def2mTZVP.h5',
                                   'Ions-sample-B973c-def2mTZVP.h5'])}

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='def2mTZVP', functional='B973c', **kwargs):
        super().__init__(self._FILES, basis_set, functional, root, download, verbose, **kwargs)


class TestDataForcesDipoles(_ArchivedBuiltinDataset):
    _FILES = {'b973c-def2mtzvp': ('TestData-forces_dipoles-b973c-def2mtzvp.tar.gz', ['ANI-1x_sample-B973c-def2mTZVP.h5'])}

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='def2mTZVP', functional='B973c', **kwargs):
        super().__init__(self._FILES, basis_set, functional, root, download, verbose, **kwargs)


class IonsVeryHeavy(_ArchivedBuiltinDataset):
    _FILES = {'b973c-def2mtzvp': ('Ions-very_heavy-b973c-def2mtzvp.tar.gz', ['Ions-very_heavy-B973c-def2mTZVP.h5'])}

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='def2mTZVP', functional='B973c', **kwargs):
        super().__init__(self._FILES, basis_set, functional, root, download, verbose, **kwargs)


class IonsHeavy(_ArchivedBuiltinDataset):
    _FILES = {'b973c-def2mtzvp': ('Ions-heavy-b973c-def2mtzvp.tar.gz', ['Ions-heavy-B973c-def2mTZVP.h5'])}

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='def2mTZVP', functional='B973c', **kwargs):
        super().__init__(self._FILES, basis_set, functional, root, download, verbose, **kwargs)


class IonsLight(_ArchivedBuiltinDataset):
    _FILES = {'b973c-def2mtzvp': ('Ions-light-b973c-def2mtzvp.tar.gz', ['Ions-light-B973c-def2mTZVP.h5'])}

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='def2mTZVP', functional='B973c', **kwargs):
        super().__init__(self._FILES, basis_set, functional, root, download, verbose, **kwargs)


class ANI1q(_ArchivedBuiltinDataset):
    _FILES = {'wb97x-631gd': ('ANI-1q-wb97x-631gd.tar.gz', ['ANI-1q-wB97X-631Gd.h5'])}

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X', **kwargs):
        super().__init__(self._FILES, basis_set, functional, root, download, verbose, **kwargs)


class ANI2qHeavy(_ArchivedBuiltinDataset):
    _FILES = {'wb97x-631gd': ('ANI-2q_heavy-wb97x-631gd.tar.gz', ['ANI-2q_heavy-wB97X-631Gd.h5'])}

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X', **kwargs):
        super().__init__(self._FILES, basis_set, functional, root, download, verbose, **kwargs)


class ANI1ccx(_ArchivedBuiltinDataset):
    _FILES = {'ccsd(t)star-cbs': ('ANI-1ccx-ccsd_t_star-cbs.tar.gz', ['ANI-1ccx-CCSD_parentheses_T_star-CBS.h5'])}

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='CBS', functional='CCSD(T)star', **kwargs):
        super().__init__(self._FILES, basis_set, functional, root, download, verbose, **kwargs)


class AminoacidDimers(_ArchivedBuiltinDataset):
    _FILES = {'b973c-def2mtzvp': ('Aminoacid-dimers-b973c-def2mtzvp.tar.gz', ['Aminoacid-dimers-B973c-def2mTZVP.h5'])}

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='def2mTZVP', functional='B973c', **kwargs):
        super().__init__(self._FILES, basis_set, functional, root, download, verbose, **kwargs)


_ANI2x_FILES = {'wb97x-631gd': ('ANI-2x-wb97x-631gd.tar.gz',
                                ['ANI-1x-wB97X-631Gd.h5',
                                 'ANI-2x_heavy-wB97X-631Gd.h5',
                                 'ANI-2x_dimers-wB97X-631Gd.h5']),
                'b973c-def2mtzvp': ('ANI-2x-b973c-def2mtzvp.tar.gz',
                                    ['ANI-1x-B973c-def2mTZVP.h5',
                                     'ANI-2x_heavy_and_dimers-B973c-def2mTZVP.h5']),
                'wb97md3bj-def2tzvpp': ('ANI-2x-wb97md3bj-def2tzvpp.tar.gz',
                                        ['ANI-1x-wB97MD3BJ-def2TZVPP.h5',
                                         'ANI-2x_heavy_and_dimers-wB97MD3BJ-def2TZVPP.h5']),
                'wb97mv-def2tzvpp': ('ANI-2x-wb97mv-def2tzvpp.tar.gz',
                                     ['ANI-1x-wB97MV-def2TZVPP.h5',
                                      'ANI-2x_heavy_and_dimers-wB97MV-def2TZVPP.h5']),
                'wb97x-def2tzvpp': ('ANI-2x-wb97x-def2tzvpp.tar.gz',
                                    ['ANI-1x-wB97X-def2TZVPP.h5',
                                     'ANI-2x_subset-wB97X-def2TZVPP.h5'])}


# ANI1x is the same as 2x, but all files that have heavy atoms or dimers are omitted
_ANI1x_FILES = {k: (v[0].replace('-2x-', '-1x-'), deepcopy(v[1])) for k, v in _ANI2x_FILES.items()}
for lot in _ANI2x_FILES.keys():
    for k in _ANI2x_FILES[lot][1]:
        if '-2x-' in k or '-2x_' in k:
            _ANI1x_FILES[lot][1].remove(k)


class ANI2x(_ArchivedBuiltinDataset):
    _FILES = _ANI2x_FILES

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X', **kwargs):
        super().__init__(self._FILES, basis_set, functional, root, download, verbose, **kwargs)


class ANI1x(_ArchivedBuiltinDataset):
    _FILES = _ANI1x_FILES

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X', **kwargs):
        super().__init__(self._FILES, basis_set, functional, root, download, verbose, **kwargs)


_COMP6v2_FILES = {'wb97x-631gd': ('COMP6-v2-wb97x-631gd.tar.gz',
                                  ['ANI-BenchMD-wB97X-631Gd.h5',
                                   'S66x8-v1-wB97X-631Gd.h5',
                                   'DrugBank-testset-wB97X-631Gd.h5',
                                   'Tripeptides-v1-wB97X-631Gd.h5',
                                   'GDB11-07-wB97X-631Gd.h5',
                                   'GDB11-08-wB97X-631Gd.h5',
                                   'GDB11-09-wB97X-631Gd.h5',
                                   'GDB11-10-wB97X-631Gd.h5',
                                   'GDB11-11-wB97X-631Gd.h5',
                                   'GDB13-12-wB97X-631Gd.h5',
                                   'GDB13-13-wB97X-631Gd.h5',
                                   'GDB-heavy07-wB97X-631Gd.h5',
                                   'GDB-heavy08-wB97X-631Gd.h5',
                                   'GDB-heavy09-wB97X-631Gd.h5',
                                   'GDB-heavy10-wB97X-631Gd.h5',
                                   'GDB-heavy11-wB97X-631Gd.h5',
                                   'Tripeptides-sulphur-wB97X-631Gd.h5',
                                   'DrugBank-SFCl-wB97X-631Gd.h5']),
                  'b973c-def2mtzvp': ('COMP6-v2-b973c-def2mtzvp.tar.gz',
                                      ['COMP6-v1_full-B973c-def2mTZVP.h5',
                                       'COMP6-heavy-B973c-def2mTZVP.h5']),
                  'wb97md3bj-def2tzvpp': ('COMP6-v2-wb97md3bj-def2tzvpp.tar.gz',
                                          ['COMP6-heavy-wB97MD3BJ-def2TZVPP.h5',
                                           'COMP6-v1_full-wB97MD3BJ-def2TZVPP.h5']),
                  'wb97mv-def2tzvpp': ('COMP6-v2-wb97mv-def2tzvpp.tar.gz',
                                       ['COMP6-heavy-wB97MV-def2TZVPP.h5',
                                        'COMP6-v1_full-wB97MV-def2TZVPP.h5'])}


# COMP6v1 is the same as v2, but all files that have heavy atoms are omitted
_COMP6v1_FILES = {k: (v[0].replace('-v2-', '-v1-'), deepcopy(v[1])) for k, v in _COMP6v2_FILES.items()}
for lot in _COMP6v2_FILES.keys():
    for k in _COMP6v2_FILES[lot][1]:
        if '-heavy' in k or '-sulphur-' in k or '-SFCl-' in k:
            _COMP6v1_FILES[lot][1].remove(k)

# There is some extra TZ data for which we have v1 values but not v2 values
# Note that the ANI-BenchMD, S66x8 and the "13" molecules (with 13 heavy atoms)
# of GDB-10to13 were recalculated using ORCA 5.0 instead of 4.2 so the integration
# grids may be slightly different, but the difference should not be significant
_COMP6v1_FILES.update({'wb97x-def2tzvpp': ('COMP6-v1-wb97x-def2tzvpp.tar.gz',
                                           ['ANI-BenchMD-wB97X-def2TZVPP.h5',
                                            'S66x8-v1-wB97X-def2TZVPP.h5',
                                            'DrugBank-testset-wB97X-def2TZVPP.h5',
                                            'Tripeptides-v1-wB97X-def2TZVPP.h5',
                                            'GDB-7to9-wB97X-def2TZVPP.h5',
                                            'GDB-10to13-wB97X-def2TZVPP.h5'])})


class COMP6v1(_ArchivedBuiltinDataset):
    _FILES = _COMP6v1_FILES

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X', **kwargs):
        super().__init__(self._FILES, basis_set, functional, root, download, verbose, **kwargs)


class COMP6v2(_ArchivedBuiltinDataset):
    _FILES = _COMP6v2_FILES

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X', **kwargs):
        super().__init__(self._FILES, basis_set, functional, root, download, verbose, **kwargs)
