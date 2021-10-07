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


(note that the conformations present in datasets with different LoT may be
different).

In all cases the "v2" and "2x" datasets are supersets of the "v1" and "1x"
datasets, so everything that is in the v1/1x datasets is also in the v2/2x
datasets, which contain extra structures.

Known issues:
- There are small inconsistencies with the names of some files:
    * COMP6 files are v1_full instead of full_v1 for wB97MV
    * for wB97M-D3BJ some files are labeled wB97D3BJ instead of wB97MD3BJ
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

_BASE_URL = 'http://moria.chem.ufl.edu/animodel/datasets/'
_DEFAULT_DATA_PATH = Path.home().joinpath('.local/torchani/Datasets')

_BUILTIN_DATASETS = ['ANI1x', 'ANI2x', 'COMP6v1', 'COMP6v2', 'ANI1ccx', 'AminoacidDimers', 'ANI1q', 'HeavyANI2q', 'LightIons', 'HeavyIons', 'VeryHeavyIons', 'TestData']
_BUILTIN_DATASETS_LOT = ['wb97x-631gd', 'b973c-def2mtzvp', 'wb97md3bj-def2tzvpp', 'wb97mv-def2tzvpp', 'wb97x-def2tzvpp', 'ccsd(t)star-cbs']


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


# NOTE: The order of the _FILES_AND_MD5S is important since it deterimenes the order of iteration over the files
class TestData(_BaseBuiltinDataset):
    _ARCHIVE = 'test-data.tar.gz'
    _FILES_AND_MD5S = OrderedDict([('test_data1.h5', '05c8eb5f92cc2e1623355229b53b7f30'),
                                   ('test_data2.h5', 'a496d2792c5fb7a9f6d9ce2116819626')])

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X'):
        assert basis_set.lower() == '631gd', "Only wB97X/631Gd data is available for this dataset"
        assert functional.lower() == 'wb97x'
        lot = f'{functional.lower()}-{basis_set.lower()}'
        if root is None:
            root = _DEFAULT_DATA_PATH.joinpath(f'Test-Data-{lot}')
        super().__init__(root, download, archive=self._ARCHIVE, files_and_md5s=self._FILES_AND_MD5S, verbose=verbose)


class IonsVeryHeavy(_BaseBuiltinDataset):
    _ARCHIVE = 'Ions-very_heavy-B973c-def2mTZVP-data.tar.gz'
    _FILES_AND_MD5S = OrderedDict([('Ions-very_heavy-B973c-def2mTZVP.h5', '64d872442fb6226ce2010a50565fe7bb')])

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='def2mTZVP', functional='B973c'):
        assert basis_set.lower() == 'def2mtzvp', "Only B973c/def2mTZVP data is available for this dataset"
        assert functional.lower() == 'b973c'
        lot = f'{functional.lower()}-{basis_set.lower()}'
        if root is None:
            root = _DEFAULT_DATA_PATH.joinpath(f'Ions-very_heavy-{lot}')
        super().__init__(root, download, archive=self._ARCHIVE, files_and_md5s=self._FILES_AND_MD5S, verbose=verbose)


class IonsHeavy(_BaseBuiltinDataset):
    _ARCHIVE = 'Ions-heavy-B973c-def2mTZVP-data.tar.gz'
    _FILES_AND_MD5S = OrderedDict([('Ions-heavy-B973c-def2mTZVP.h5', 'ef1b406d453b71488683c1cf9f1aa316')])

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='def2mTZVP', functional='B973c'):
        assert basis_set.lower() == 'def2mtzvp', "Only B973c/def2mTZVP data is available for this dataset"
        assert functional.lower() == 'b973c'
        lot = f'{functional.lower()}-{basis_set.lower()}'
        if root is None:
            root = _DEFAULT_DATA_PATH.joinpath(f'Ions-heavy-{lot}')
        super().__init__(root, download, archive=self._ARCHIVE, files_and_md5s=self._FILES_AND_MD5S, verbose=verbose)


class IonsLight(_BaseBuiltinDataset):
    _ARCHIVE = 'Ions-light-B973c-def2mTZVP-data.tar.gz'
    _FILES_AND_MD5S = OrderedDict([('Ions-light-B973c-def2mTZVP.h5', 'af2e520d3eace248a1ad7a8bdb8ec7c7')])

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='def2mTZVP', functional='B973c'):
        assert basis_set.lower() == 'def2mtzvp', "Only B973c/def2mTZVP data is available for this dataset"
        assert functional.lower() == 'b973c'
        lot = f'{functional.lower()}-{basis_set.lower()}'
        if root is None:
            root = _DEFAULT_DATA_PATH.joinpath(f'Ions-light-{lot}')
        super().__init__(root, download, archive=self._ARCHIVE, files_and_md5s=self._FILES_AND_MD5S, verbose=verbose)


class ANI1q(_BaseBuiltinDataset):
    _ARCHIVE = 'ANI-1q-wB97X-631Gd-data.tar.gz'
    _FILES_AND_MD5S = OrderedDict([('ANI-1q-wB97X-631Gd.h5', 'a66e3d50e44ed0c863abc52e943ca1c2')])

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X'):
        assert basis_set.lower() == '631gd', "Only wB97X/631Gd data is available for this dataset"
        assert functional.lower() == 'wb97x'
        r"""ANI-1x subset with CM5 atomic charges and multipoles

         Very limited subset of the wB97X/631G(d) ANI-1x dataset for which 'atomic CM5 charges',
        'QM dipoles' and 'QM quadrupoles' are available"""
        lot = f'{functional.lower()}-{basis_set.lower()}'
        if root is None:
            root = _DEFAULT_DATA_PATH.joinpath(f'ANI-1q-{lot}')
        super().__init__(root, download, archive=self._ARCHIVE, files_and_md5s=self._FILES_AND_MD5S, verbose=verbose)


class ANI2qHeavy(_BaseBuiltinDataset):
    _ARCHIVE = 'ANI-2q_heavy-wB97X-631Gd-data.tar.gz'
    _FILES_AND_MD5S = OrderedDict([('ANI-2q_heavy-wB97X-631Gd.h5', '3d5a1ab8f6065f130b89e633f453abaf')])

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X'):
        assert basis_set.lower() == '631gd', "Only wB97X/631Gd data is available for this dataset"
        assert functional.lower() == 'wb97x'
        r"""ANI-2x "heavy" subset with CM5 atomic charges

        Subset of the wB97X/631G(d) ANI-2x dataset ("heavy" part only) for
        which 'atomic CM5 charges' and 'atomic hirshfeld dipole magnitudes' are
        available"""
        lot = f'{functional.lower()}-{basis_set.lower()}'
        if root is None:
            root = _DEFAULT_DATA_PATH.joinpath(f'ANI-2q_heavy-{lot}')
        super().__init__(root, download, archive=self._ARCHIVE, files_and_md5s=self._FILES_AND_MD5S, verbose=verbose)


class ANI1ccx(_BaseBuiltinDataset):
    _ARCHIVE = {'ccsd(t)star-cbs': 'ANI-1ccx-CCSD_parentheses_T_star-CBS-data.tar.gz'}
    _FILES_AND_MD5S = {'ccsd(t)star-cbs': OrderedDict([('ANI-1ccx-CCSD_parentheses_T_star-CBS.h5', 'a7218b99f843bc56a1ec195271082c40')])}

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='CBS', functional='CCSD(T)star'):
        lot = f'{functional.lower()}-{basis_set.lower()}'
        if lot not in self._ARCHIVE.keys():
            raise ValueError(f"Unsupported functional-basis set combination, try one of {set(self._ARCHIVE.keys())}")
        if root is None:
            root = _DEFAULT_DATA_PATH.joinpath(f'ANI-1ccx-{lot}')
        super().__init__(root, download, archive=self._ARCHIVE[lot], files_and_md5s=self._FILES_AND_MD5S[lot], verbose=verbose)


class AminoacidDimers(_BaseBuiltinDataset):
    _ARCHIVE = {'b973c-def2mtzvp': 'Aminoacid-dimers-B973c-def2mTZVP-data.tar.gz'}
    _FILES_AND_MD5S = {'b973c-def2mtzvp': OrderedDict([('Aminoacid-dimers-B973c-def2mTZVP.h5', '7db327a3cf191c19a06f5495453cfe56')])}

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='def2mTZVP', functional='B973c'):
        lot = f'{functional.lower()}-{basis_set.lower()}'
        if lot not in self._ARCHIVE.keys():
            raise ValueError(f"Unsupported functional-basis set combination, try one of {set(self._ARCHIVE.keys())}")
        if root is None:
            root = _DEFAULT_DATA_PATH.joinpath(f'Aminoacid-Dimers-{lot}')
        super().__init__(root, download, archive=self._ARCHIVE[lot], files_and_md5s=self._FILES_AND_MD5S[lot], verbose=verbose)


_ANI_LOT = {'wb97x-631gd', 'b973c-def2mtzvp', 'wb97md3bj-def2tzvpp', 'wb97mv-def2tzvpp', 'wb97x-def2tzvpp'}

_ANI2x_ARCHIVE = {'wb97x-631gd': 'ANI-2x-wB97X-631Gd-data.tar.gz',
                  'b973c-def2mtzvp': 'ANI-2x-B973c-def2mTZVP-data.tar.gz',
                  'wb97md3bj-def2tzvpp': 'ANI-2x-wB97MD3BJ-def2TZVPP-data.tar.gz',
                  'wb97mv-def2tzvpp': 'ANI-2x-wB97MV-def2TZVPP-data.tar.gz',
                  'wb97x-def2tzvpp': 'ANI-2x-wB97X-def2TZVPP-data.tar.gz'}

_ANI2x_FILES_AND_MD5S = {'wb97x-631gd': OrderedDict([('ANI-1x-wB97X-631Gd.h5', '2cd8cbc7a5106f88d8b21cde58074aef'),
                                                     ('ANI-2x_heavy-wB97X-631Gd.h5', '0bf1f7fb8c97768116deea672cae8d8e'),
                                                     ('ANI-2x_dimers-wB97X-631Gd.h5', '0043cc1f908851601d9cfbbec2d957e8')]),
                         'b973c-def2mtzvp': OrderedDict([('ANI-1x-B973c-def2mTZVP.h5', '4351aedf74683d858b834ccef3a727b8'),
                                                         ('ANI-2x_heavy_and_dimers-B973c-def2mTZVP.h5', '09f7c991c78327d92c6de479aff69aca')]),
                         'wb97md3bj-def2tzvpp': OrderedDict([('ANI-1x-wB97D3BJ-def2TZVPP.h5', '6d7d3ba93d4c57e4ac6a6d5dc9598596'),
                                                             ('ANI-2x_heavy_and_dimers-wB97D3BJ-def2TZVPP.h5', '827a3eb6124ef2c0c3ab4487b63ff329')]),
                         'wb97mv-def2tzvpp': OrderedDict([('ANI-1x-wB97MV-def2TZVPP.h5', '7f3107e3474f3f673922a0155e11d3aa'),
                                                          ('ANI-2x_heavy_and_dimers-wB97MV-def2TZVPP.h5', 'b60d7938e16b776eb72209972c54721c')]),
                         'wb97x-def2tzvpp': OrderedDict([('ANI-1x-wB97X-def2TZVPP.h5', '8bd2f258c8b4588d2d64499af199dc74'),
                                                          ('ANI-2x_subset-wB97X-def2TZVPP.h5', '6db204f90128a9ad470575cb17373586')])}


# ANI1x is the same as 2x, but all files that have heavy atoms or dimers are omitted
_ANI1x_ARCHIVE = {k: v.replace('-2x-', '-1x-') for k, v in _ANI2x_ARCHIVE.items()}
_ANI1x_FILES_AND_MD5S = deepcopy(_ANI2x_FILES_AND_MD5S)
for lot in _ANI_LOT:
    for k in _ANI2x_FILES_AND_MD5S[lot]:
        if '-2x-' in k or '-2x_' in k:
            del _ANI1x_FILES_AND_MD5S[lot][k]


class ANI2x(_BaseBuiltinDataset):
    _ARCHIVE = _ANI2x_ARCHIVE
    _FILES_AND_MD5S = _ANI2x_FILES_AND_MD5S

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X'):
        lot = f'{functional.lower()}-{basis_set.lower()}'
        if lot not in self._ARCHIVE.keys():
            raise ValueError(f"Unsupported functional-basis set combination, try one of {set(self._ARCHIVE.keys())}")
        if root is None:
            root = _DEFAULT_DATA_PATH.joinpath(f'ANI-2x-{lot}')
        super().__init__(root, download, archive=self._ARCHIVE[lot], files_and_md5s=self._FILES_AND_MD5S[lot], verbose=verbose)


class ANI1x(_BaseBuiltinDataset):
    _ARCHIVE = _ANI1x_ARCHIVE
    _FILES_AND_MD5S = _ANI1x_FILES_AND_MD5S

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X'):
        lot = f'{functional.lower()}-{basis_set.lower()}'
        if lot not in self._ARCHIVE.keys():
            raise ValueError(f"Unsupported functional-basis set combination, try one of {set(self._ARCHIVE.keys())}")
        if root is None:
            root = _DEFAULT_DATA_PATH.joinpath(f'ANI-1x-{lot}')
        super().__init__(root, download, archive=self._ARCHIVE[lot], files_and_md5s=self._FILES_AND_MD5S[lot], verbose=verbose)


_COMP6_LOT = {'wb97x-631gd', 'b973c-def2mtzvp', 'wb97md3bj-def2tzvpp', 'wb97mv-def2tzvpp'}

_COMP6v2_FILES_AND_MD5S = {'wb97x-631gd': OrderedDict([('ANI-BenchMD-wB97X-631Gd.h5', '04c03ec8796359a0e3eb301346efbb03'),
                                                       ('S66x8-v1-wB97X-631Gd.h5', '2b932f920397ae92bf55cfbc26de9a33'),
                                                       ('DrugBank-testset-wB97X-631Gd.h5', 'ed92ec0b47061f8a1ae370390c8eff6e'),
                                                       ('Tripeptides-v1-wB97X-631Gd.h5', '7fd7ddf224b2c329135b16f80d5cad75'),
                                                       ('GDB11-07-wB97X-631Gd.h5', '719d5442ddf1cd2f02b94eb048ce0c56'),
                                                       ('GDB11-08-wB97X-631Gd.h5', 'abf76ddcfed962ba8b91d7a99fb86a1b'),
                                                       ('GDB11-09-wB97X-631Gd.h5', '70841880e1bbdf063ed943af94367b70'),
                                                       ('GDB11-10-wB97X-631Gd.h5', 'cb86b0ee9de2d719b7e7bca789f297d9'),
                                                       ('GDB11-11-wB97X-631Gd.h5', '367c0fa78b8eac584009fbe81f7198ba'),
                                                       ('GDB13-12-wB97X-631Gd.h5', '9757ac7e7c937074894b314aa82de41a'),
                                                       ('GDB13-13-wB97X-631Gd.h5', '86fb89bb64066a60e6013e33c704565b'),
                                                       ('GDB-heavy07-wB97X-631Gd.h5', '3b2b6fc298d06acb8380de70e4dca5dc'),
                                                       ('GDB-heavy08-wB97X-631Gd.h5', '8877bdfc95419c6b818b6f07bf147fa0'),
                                                       ('GDB-heavy09-wB97X-631Gd.h5', '416aa86b57ca9915135a6d99d4a2d23a'),
                                                       ('GDB-heavy10-wB97X-631Gd.h5', 'e45852f95ec876dfd41984d1c159130b'),
                                                       ('GDB-heavy11-wB97X-631Gd.h5', 'e4716b57d02e64e3b2bb7096d0ab70ab'),
                                                       ('Tripeptides-sulphur-wB97X-631Gd.h5', '3309d50ede42ceaa96e5d3f897e9bac0'),
                                                       ('DrugBank-SFCl-wB97X-631Gd.h5', '76db51f3750d9322656682b104299442')]),
                           'b973c-def2mtzvp': OrderedDict([('COMP6-full_v1-B97c3-def2mTZVP.h5', '9cd1e1403c0ca91e07c480eda86332f8'),
                                                           ('COMP6-heavy-B97c3-def2mTZVP.h5', '1cf3ecd2a2ec7257909c693350d66d18')]),
                           'wb97md3bj-def2tzvpp': OrderedDict([('COMP6-heavy-wB97D3BJ-def2TZVPP.h5', '88aac626d4963aacf9e856ca1408f47b'),
                                                               ('COMP6-full_v1-wB97D3BJ-def2TZVPP.h5', '057d89c8d046ccd9155ee24f3f47faa6')]),
                           'wb97mv-def2tzvpp': OrderedDict([('COMP6-heavy-wB97MV-def2TZVPP.h5', '804e7a7655903c8a4599f2c48bd584aa'),
                                                            ('COMP6-v1_full-wB97MV-def2TZVPP.h5', 'bccdf302f361c0213450381b493e17d8')])}

_COMP6v2_ARCHIVE = {'wb97x-631gd': 'COMP6-v2-wB97X-631Gd-data.tar.gz',
                    'b973c-def2mtzvp': 'COMP6-v2-B973c-def2mTZVP-data.tar.gz',
                    'wb97md3bj-def2tzvpp': 'COMP6-v2-wB97MD3BJ-def2TZVPP-data.tar.gz',
                    'wb97mv-def2tzvpp': 'COMP6-v2-wB97MV-def2TZVPP-data.tar.gz'}

# COMP6v1 is the same as v2, but all files that have heavy atoms are omitted
_COMP6v1_ARCHIVE = {k: v.replace('-v2-', '-v1-') for k, v in _COMP6v2_ARCHIVE.items()}
_COMP6v1_FILES_AND_MD5S = deepcopy(_COMP6v2_FILES_AND_MD5S)
for lot in _COMP6_LOT:
    for k in _COMP6v2_FILES_AND_MD5S[lot]:
        if '-heavy' in k or '-sulphur-' in k or '-SFCl-' in k:
            del _COMP6v1_FILES_AND_MD5S[lot][k]

# There is some extra TZ data for which we have v1 values but not v2 values
# Note that the ANI-BenchMD, S66x8 and the "13" molecules (with 13 heavy atoms)
# of GDB-10to13 were recalculated using ORCA 5.0 instead of 4.2 so the integration
# grids may be slightly different, but the difference should not be significant
_COMP6v1_FILES_AND_MD5S.update({'wb97x-def2tzvpp': OrderedDict([('ANI-BenchMD-wB97X-def2TZVPP.h5', '9cd6d267b2d3d651cba566650642ed62'),
                                                            ('S66x8-v1-wB97X-def2TZVPP.h5', 'a7aa6ce11497d182c1265219e5e2925f'),
                                                            ('DrugBank-testset-wB97X-def2TZVPP.h5', '977e1d6863fccdbbc6340acb15b1eec2'),
                                                            ('Tripeptides-v1-wB97X-def2TZVPP.h5', '6b838fee970244ad85419165bb71c557'),
                                                            ('GDB-7to9-wB97X-def2TZVPP.h5', '23b80666f75cb71030534efdc7df7c97'),
                                                            ('GDB-10to13-wB97X-def2TZVPP.h5', 'bd9730961eaf15a3d823b97f39c41908')])})
_COMP6v1_ARCHIVE.update({'wb97x-def2tzvpp': 'COMP6-v1-wB97X-def2TZVPP-data.tar.gz'})


class COMP6v1(_BaseBuiltinDataset):
    _ARCHIVE = _COMP6v1_ARCHIVE
    _FILES_AND_MD5S = _COMP6v1_FILES_AND_MD5S

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X'):
        lot = f'{functional.lower()}-{basis_set.lower()}'
        if lot not in self._ARCHIVE.keys():
            raise ValueError(f"Unsupported functional-basis set combination, try one of {set(self._ARCHIVE.keys())}")
        if root is None:
            root = _DEFAULT_DATA_PATH.joinpath(f'COMP6-v1-{lot}')
        super().__init__(root, download, archive=self._ARCHIVE[lot], files_and_md5s=self._FILES_AND_MD5S[lot], verbose=verbose)


class COMP6v2(_BaseBuiltinDataset):
    _ARCHIVE = _COMP6v2_ARCHIVE
    _FILES_AND_MD5S = _COMP6v2_FILES_AND_MD5S

    def __init__(self, root: StrPath = None, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X'):
        lot = f'{functional.lower()}-{basis_set.lower()}'
        if lot not in self._ARCHIVE.keys():
            raise ValueError(f"Unsupported functional-basis set combination, try one of {set(self._ARCHIVE.keys())}")
        if root is None:
            root = _DEFAULT_DATA_PATH.joinpath(f'COMP6-v2-{lot}')
        super().__init__(root, download, archive=self._ARCHIVE[lot], files_and_md5s=self._FILES_AND_MD5S[lot], verbose=verbose)
