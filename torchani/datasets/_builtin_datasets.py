r"""Torchani Builtin Datasets

This module provides access to the following datasets, calculated with specific
levels of theory (LoT) which are combinations functional/basis_set or
wavefunction_method/basis_set when appropriate.

- ANI-1x, with LoT:
    - wB97X/6-31G(d)
    - B97-3c/def2-mTZVP
    - wB97M-D3BJ/def2-TZVPP
    - wB97MV/def2-TZVPP

- ANI-2x, with LoT:
    - wB97X/6-31G(d)
    - B97-3c/def2-mTZVP
    - wB97M-D3BJ/def2-TZVPP
    - wB97MV/def2-TZVPP

- ANI-1ccx, with LoT:
    - CCSD(T)star/CBS
  Note that this dataset also has Hartree Fock (HF) energies, RI-MP2 energies
  and forces and DPLNO-CCSD(T) energies for different basis sets and PNO
  settings.

- COMP6-v1, with LoT:
    - wB97X/6-31G(d)
    - B97-3c/def2-mTZVP
    - wB97M-D3BJ/def2-TZVPP
    - wB97MV/def2-TZVPP

- COMP6-v2, with LoT:
    - wB97X/6-31G(d)
    - B97-3c/def2-mTZVP
    - wB97M-D3BJ/def2-TZVPP
    - wB97MV/def2-TZVPP

- AminoacidDimers, with LoT:
    - B97-3c/def2-mTZVP

(note that the conformations present in datasets with different LoT may be
different).

In all cases the "v2" and "2x" datasets are supersets of the "v1" and "1x"
datasets, so everything that is in the v1/1x datasets is also in the v2/2x
datasets, which contain extra structures.

Known issues:
- The wB97X/def2-TZVPP datasets are still not available through this module
- There are small inconsistencies with the names of some files:
    * COMP6 files are v1_full instead of full_v1 for wB97MV
    * for wB97M-D3BJ some files are labeled wB97D3BJ instead of wB97MD3BJ
"""
from pathlib import Path
from typing import Optional, Any
from collections import OrderedDict
from copy import deepcopy

from torchvision.datasets.utils import download_and_extract_archive, list_files, check_integrity
from .datasets import ANIDataset
from ._annotations import StrPath
from ..utils import tqdm

_BASE_URL = 'http://moria.chem.ufl.edu/animodel/datasets/'


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
            if not self._maybe_download_hdf5_archive_and_check_integrity(root):
                raise RuntimeError('Dataset could not be download or is corrupted, '
                                   'please try downloading again')
        else:
            if not self._check_hdf5_files_integrity(root):
                raise RuntimeError('Dataset not found or is corrupted, '
                                   'you can use "download = True" to download it')
        dataset_paths = [Path(p).resolve() for p in list_files(root, suffix='.h5', prefix=True)]

        # Order dataset paths using the order given in "files and md5s"
        filenames_order = {Path(k).stem: j for j, k in enumerate(self._files_and_md5s.keys())}
        _filenames_and_paths = sorted([(p.stem, p) for p in dataset_paths],
                                     key=lambda tup: filenames_order[tup[0]])
        filenames_and_paths = OrderedDict(_filenames_and_paths)
        super().__init__(locations=filenames_and_paths.values(), names=filenames_and_paths.keys(), **h5_dataset_kwargs)
        if h5_dataset_kwargs.get('verbose', True):
            print(self)

    def _check_hdf5_files_integrity(self, root: Path) -> bool:
        # Checks that all HDF5 files in the provided path are equal to the
        # expected ones and have the correct checksum, other files such as
        # tar.gz archives are neglected
        present_files = [Path(f).resolve() for f in list_files(root, suffix='.h5', prefix=True)]
        expected_file_names = set(self._files_and_md5s.keys())
        present_file_names = set([f.name for f in present_files])
        if expected_file_names != present_file_names:
            print(f"Wrong files found for dataset {self.__class__.__name__}, "
                  f"expected {expected_file_names} but found {present_file_names}")
            return False
        for f in tqdm(present_files, desc=f'Checking integrity of files for dataset {self.__class__.__name__}'):
            if not check_integrity(f, self._files_and_md5s[f.name]):
                print(f"All expected files for dataset {self.__class__.__name__} "
                      f"were found but file {f.name} failed integrity check")
                return False
        return True

    def _maybe_download_hdf5_archive_and_check_integrity(self, root: Path) -> bool:
        # Downloads only if the files have not been found or are corrupted
        root = Path(root).resolve()
        if root.is_dir() and self._check_hdf5_files_integrity(root):
            return True
        download_and_extract_archive(url=f'{_BASE_URL}{self._archive}', download_root=root, md5=None)
        return self._check_hdf5_files_integrity(root)


# NOTE: The order of the _FILES_AND_MD5S is important since it deterimenes the order of iteration over the files
class TestData(_BaseBuiltinDataset):
    _ARCHIVE = 'test-data.tar.gz'
    _FILES_AND_MD5S = OrderedDict([('test_data1.h5', '05c8eb5f92cc2e1623355229b53b7f30'),
                                   ('test_data2.h5', 'a496d2792c5fb7a9f6d9ce2116819626')])

    def __init__(self, root: StrPath, download: bool = False, verbose: bool = True):
        super().__init__(root, download, archive=self._ARCHIVE, files_and_md5s=self._FILES_AND_MD5S, verbose=verbose)


class ANI1ccx(_BaseBuiltinDataset):
    _ARCHIVE = {'CCSD(T)star-CBS': 'ANI-1ccx-CCSD_parentheses_T_star-CBS-data.tar.gz'}
    _FILES_AND_MD5S = {'CCSD(T)star-CBS': OrderedDict([('ANI-1ccx-CCSD_parentheses_T_star-CBS.h5', 'a7218b99f843bc56a1ec195271082c40')])}

    def __init__(self, root: StrPath, download: bool = False, verbose: bool = True, basis_set='CBS', functional='CCSD(T)star'):
        lot = f'{functional}-{basis_set}'
        if lot not in self._ARCHIVE.keys():
            raise ValueError(f"Unsupported functional-basis set combination, try one of {set(self._ARCHIVE.keys())}")
        super().__init__(root, download, archive=self._ARCHIVE[lot], files_and_md5s=self._FILES_AND_MD5S[lot], verbose=verbose)


class AminoacidDimers(_BaseBuiltinDataset):
    _ARCHIVE = {'B973c-def2mTZVP': 'Aminoacid-dimers-B973c-def2mTZVP-data.tar.gz'}
    _FILES_AND_MD5S = {'B973c-def2mTZVP': OrderedDict([('Aminoacid-dimers-B973c-def2mTZVP.h5', '7db327a3cf191c19a06f5495453cfe56')])}

    def __init__(self, root: StrPath, download: bool = False, verbose: bool = True, basis_set='def2mTZVP', functional='B973c'):
        lot = f'{functional}-{basis_set}'
        if lot not in self._ARCHIVE.keys():
            raise ValueError(f"Unsupported functional-basis set combination, try one of {set(self._ARCHIVE.keys())}")
        super().__init__(root, download, archive=self._ARCHIVE[lot], files_and_md5s=self._FILES_AND_MD5S[lot], verbose=verbose)


_ANI_LOT = {'wB97X-631Gd', 'B973c-def2mTZVP', 'wB97MD3BJ-def2TZVPP', 'wB97MV-def2TZVPP'}

_ANI2x_ARCHIVE = {'wB97X-631Gd': 'ANI-2x-wB97X-631Gd-data.tar.gz',
                  'B973c-def2mTZVP': 'ANI-2x-B973c-def2mTZVP-data.tar.gz',
                  'wB97MD3BJ-def2TZVPP': 'ANI-2x-wB97MD3BJ-def2TZVPP-data.tar.gz',
                  'wB97MV-def2TZVPP': 'ANI-2x-wB97MV-def2TZVPP-data.tar.gz'}

_ANI2x_FILES_AND_MD5S = {'wB97X-631Gd': OrderedDict([('ANI-1x-wB97X-631Gd.h5', '2cd8cbc7a5106f88d8b21cde58074aef'),
                                                     ('ANI-2x_heavy-wB97X-631Gd.h5', '0bf1f7fb8c97768116deea672cae8d8e'),
                                                     ('ANI-2x_dimers-wB97X-631Gd.h5', '0043cc1f908851601d9cfbbec2d957e8')]),
                         'B973c-def2mTZVP': OrderedDict([('ANI-1x-B973c-def2mTZVP.h5', '4351aedf74683d858b834ccef3a727b8'),
                                                         ('ANI-2x_heavy_and_dimers-B973c-def2mTZVP.h5', '09f7c991c78327d92c6de479aff69aca')]),
                         'wB97MD3BJ-def2TZVPP': OrderedDict([('ANI-1x-wB97D3BJ-def2TZVPP.h5', '6d7d3ba93d4c57e4ac6a6d5dc9598596'),
                                                             ('ANI-2x_heavy_and_dimers-wB97D3BJ-def2TZVPP.h5', '827a3eb6124ef2c0c3ab4487b63ff329')]),
                         'wB97MV-def2TZVPP': OrderedDict([('ANI-1x-wB97MV-def2TZVPP.h5', '7f3107e3474f3f673922a0155e11d3aa'),
                                                          ('ANI-2x_heavy_and_dimers-wB97MV-def2TZVPP.h5', 'b60d7938e16b776eb72209972c54721c')])}


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

    def __init__(self, root: StrPath, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X'):
        lot = f'{functional}-{basis_set}'
        if lot not in self._ARCHIVE.keys():
            raise ValueError(f"Unsupported functional-basis set combination, try one of {set(self._ARCHIVE.keys())}")
        super().__init__(root, download, archive=self._ARCHIVE[lot], files_and_md5s=self._FILES_AND_MD5S[lot], verbose=verbose)


class ANI1x(_BaseBuiltinDataset):
    _ARCHIVE = _ANI1x_ARCHIVE
    _FILES_AND_MD5S = _ANI1x_FILES_AND_MD5S

    def __init__(self, root: StrPath, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X'):
        lot = f'{functional}-{basis_set}'
        if lot not in self._ARCHIVE.keys():
            raise ValueError(f"Unsupported functional-basis set combination, try one of {set(self._ARCHIVE.keys())}")
        super().__init__(root, download, archive=self._ARCHIVE[lot], files_and_md5s=self._FILES_AND_MD5S[lot], verbose=verbose)


_COMP6_LOT = {'wB97X-631Gd', 'B973c-def2mTZVP', 'wB97MD3BJ-def2TZVPP', 'wB97MV-def2TZVPP'}

_COMP6v2_FILES_AND_MD5S = {'wB97X-631Gd': OrderedDict([('ANI-BenchMD-wB97X-631Gd.h5', '04c03ec8796359a0e3eb301346efbb03'),
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
                           'B973c-def2mTZVP': OrderedDict([('COMP6-full_v1-B97c3-def2mTZVP.h5', '9cd1e1403c0ca91e07c480eda86332f8'),
                                                           ('COMP6-heavy-B97c3-def2mTZVP.h5', '1cf3ecd2a2ec7257909c693350d66d18')]),
                           'wB97MD3BJ-def2TZVPP': OrderedDict([('COMP6-heavy-wB97D3BJ-def2TZVPP.h5', '88aac626d4963aacf9e856ca1408f47b'),
                                                               ('COMP6-full_v1-wB97D3BJ-def2TZVPP.h5', '057d89c8d046ccd9155ee24f3f47faa6')]),
                           'wB97MV-def2TZVPP': OrderedDict([('COMP6-heavy-wB97MV-def2TZVPP.h5', '804e7a7655903c8a4599f2c48bd584aa'),
                                                            ('COMP6-v1_full-wB97MV-def2TZVPP.h5', 'bccdf302f361c0213450381b493e17d8')])}

_COMP6v2_ARCHIVE = {'wB97X-631Gd': 'COMP6-v2-wB97X-631Gd-data.tar.gz',
                    'B973c-def2mTZVP': 'COMP6-v2-B973c-def2mTZVP-data.tar.gz',
                    'wB97MD3BJ-def2TZVPP': 'COMP6-v2-wB97MD3BJ-def2TZVPP-data.tar.gz',
                    'wB97MV-def2TZVPP': 'COMP6-v2-wB97MV-def2TZVPP-data.tar.gz'}

# COMP6v1 is the same as v2, but all files that have heavy atoms are omitted
_COMP6v1_ARCHIVE = {k: v.replace('-v2-', '-v1-') for k, v in _COMP6v2_ARCHIVE.items()}
_COMP6v1_FILES_AND_MD5S = deepcopy(_COMP6v2_FILES_AND_MD5S)
for lot in _COMP6_LOT:
    for k in _COMP6v2_FILES_AND_MD5S[lot]:
        if '-heavy' in k or '-sulphur-' in k or '-SFCl-' in k:
            del _COMP6v1_FILES_AND_MD5S[lot][k]


class COMP6v1(_BaseBuiltinDataset):
    _ARCHIVE = _COMP6v1_ARCHIVE
    _FILES_AND_MD5S = _COMP6v1_FILES_AND_MD5S

    def __init__(self, root: StrPath, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X'):
        lot = f'{functional}-{basis_set}'
        if lot not in self._ARCHIVE.keys():
            raise ValueError(f"Unsupported functional-basis set combination, try one of {set(self._ARCHIVE.keys())}")
        super().__init__(root, download, archive=self._ARCHIVE[lot], files_and_md5s=self._FILES_AND_MD5S[lot], verbose=verbose)


class COMP6v2(_BaseBuiltinDataset):
    _ARCHIVE = _COMP6v2_ARCHIVE
    _FILES_AND_MD5S = _COMP6v2_FILES_AND_MD5S

    def __init__(self, root: StrPath, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X'):
        lot = f'{functional}-{basis_set}'
        if lot not in self._ARCHIVE.keys():
            raise ValueError(f"Unsupported functional-basis set combination, try one of {set(self._ARCHIVE.keys())}")
        super().__init__(root, download, archive=self._ARCHIVE[lot], files_and_md5s=self._FILES_AND_MD5S[lot], verbose=verbose)
