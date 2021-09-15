r"""Torchani Builtin Datasets

This module provides access to the following datasets, calculated with specific
levels of theory (LoT) which are combinations functional/basis_set or
wavefunction_method/basis_set when appropriate.

- ANI-1x, with LoT:
    - wB97X/6-31G(d)
    - B97-3c/def2-mTZVP

- ANI-2x, with LoT:
    - wB97X/6-31G(d)
    - B97-3c/def2-mTZVP

- COMP6-v1, with LoT:
    - wB97X/6-31G(d)
    - B97-3c/def2-mTZVP

- COMP6-v2, with LoT:
    - wB97X/6-31G(d)
    - B97-3c/def2-mTZVP

- AminoacidDimers, with LoT:
    - B97-3c/def2-mTZVP

(note that the conformations present in datasets with different LoT may be
different).
"""
from pathlib import Path
from typing import Optional, Any
from collections import OrderedDict

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


class AminoacidDimers(_BaseBuiltinDataset):
    _ARCHIVE = {'B973c-def2mTZVP': 'Aminoacid-dimers-B973c-def2mTZVP-data.tar.gz'}
    _FILES_AND_MD5S = {'B973c-def2mTZVP': OrderedDict([('Aminoacid-dimers-B973c-def2mTZVP.h5', '7db327a3cf191c19a06f5495453cfe56')])}

    def __init__(self, root: StrPath, download: bool = False, verbose: bool = True, basis_set='def2mTZVP', functional='B973c'):
        lot = f'{functional}-{basis_set}'
        if lot not in self._ARCHIVE.keys():
            raise ValueError(f"Unsupported functional-basis set combination, try one of {set(self._ARCHIVE.keys())}")
        super().__init__(root, download, archive=self._ARCHIVE[lot], files_and_md5s=self._FILES_AND_MD5S[lot], verbose=verbose)


class ANI1x(_BaseBuiltinDataset):
    _ARCHIVE = {'wB97X-631Gd': 'ANI-1x-wB97X-6-31Gd-data.tar.gz',
                'B973c-def2mTZVP': 'ANI-1x-B973c-def2mTZVP-data.tar.gz'}
    _FILES_AND_MD5S = {'wB97X-631Gd': OrderedDict([('ANI-1x-wB97X-6-31Gd.h5', 'c9d63bdbf90d093db9741c94d9b20972')]),
                       'B973c-def2mTZVP': OrderedDict([('ANI-1x-B973c-def2mTZVP.h5', '2f50da8c73236a41f33a8e561a80c77e')]),
                       }

    def __init__(self, root: StrPath, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X'):
        lot = f'{functional}-{basis_set}'
        if lot not in self._ARCHIVE.keys():
            raise ValueError(f"Unsupported functional-basis set combination, try one of {set(self._ARCHIVE.keys())}")
        super().__init__(root, download, archive=self._ARCHIVE[lot], files_and_md5s=self._FILES_AND_MD5S[lot], verbose=verbose)


class ANI2x(_BaseBuiltinDataset):
    _ARCHIVE = {'wB97X-631Gd': 'ANI-2x-wB97X-6-31Gd-data.tar.gz',
                'B973c-def2mTZVP': 'ANI-2x-B973c-def2mTZVP-data.tar.gz'}
    _FILES_AND_MD5S = {'wB97X-631Gd': OrderedDict([('ANI-1x-wB97X-6-31Gd.h5', 'c9d63bdbf90d093db9741c94d9b20972'),
                                                   ('ANI-2x-heavy-wB97X-6-31Gd.h5', '49ec3dc5d046f5718802f5d1f102391c'),
                                                   ('ANI-2x-dimers-wB97X-6-31Gd.h5', '3455d82a50c63c389126b68607fb9ca8')]),
                       'B973c-def2mTZVP': OrderedDict([('ANI-1x-B973c-def2mTZVP.h5', '2f50da8c73236a41f33a8e561a80c77e'),
                                                       ('ANI-2x-heavy_and_dimers-B973c-def2mTZVP.h5', 'cffbe6e0e076d2fa7de7c3d15d4dd1f2')])}

    def __init__(self, root: StrPath, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X'):
        lot = f'{functional}-{basis_set}'
        if lot not in self._ARCHIVE.keys():
            raise ValueError(f"Unsupported functional-basis set combination, try one of {set(self._ARCHIVE.keys())}")
        super().__init__(root, download, archive=self._ARCHIVE[lot], files_and_md5s=self._FILES_AND_MD5S[lot], verbose=verbose)


class COMP6v1(_BaseBuiltinDataset):
    _ARCHIVE = {'wB97X-631Gd': 'COMP6-v1-wB97X-631Gd-data.tar.gz',
                'B973c-def2mTZVP': 'COMP6-v1-B973c-def2mTZVP-data.tar.gz'}
    _FILES_AND_MD5S = {'wB97X-631Gd': OrderedDict([('ANI-BenchMD-wB97X-631Gd.h5', '04c03ec8796359a0e3eb301346efbb03'),
                                                   ('S66x8-v1-wB97X-631Gd.h5', '2b932f920397ae92bf55cfbc26de9a33'),
                                                   ('DrugBank-testset-wB97X-631Gd.h5', 'ed92ec0b47061f8a1ae370390c8eff6e'),
                                                   ('Tripeptides-v1-wB97X-631Gd.h5', '7fd7ddf224b2c329135b16f80d5cad75'),
                                                   ('GDB11-07-wB97X-631Gd.h5', '719d5442ddf1cd2f02b94eb048ce0c56'),
                                                   ('GDB11-08-wB97X-631Gd.h5', 'abf76ddcfed962ba8b91d7a99fb86a1b'),
                                                   ('GDB11-09-wB97X-631Gd.h5', '70841880e1bbdf063ed943af94367b70'),
                                                   ('GDB11-10-wB97X-631Gd.h5', 'cb86b0ee9de2d719b7e7bca789f297d9'),
                                                   ('GDB11-11-wB97X-631Gd.h5', '367c0fa78b8eac584009fbe81f7198ba'),
                                                   ('GDB13-12-wB97X-631Gd.h5', '9757ac7e7c937074894b314aa82de41a'),
                                                   ('GDB13-13-wB97X-631Gd.h5', '86fb89bb64066a60e6013e33c704565b')]),
                       'B973c-def2mTZVP': OrderedDict([('COMP6-full_v1-B97c3-def2mTZVP.h5', '044556f8490cc9e92975b949c0da5099')])}

    def __init__(self, root: StrPath, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X'):
        lot = f'{functional}-{basis_set}'
        if lot not in self._ARCHIVE.keys():
            raise ValueError(f"Unsupported functional-basis set combination, try one of {set(self._ARCHIVE.keys())}")
        super().__init__(root, download, archive=self._ARCHIVE[lot], files_and_md5s=self._FILES_AND_MD5S[lot], verbose=verbose)


class COMP6v2(_BaseBuiltinDataset):
    _ARCHIVE = {'wB97X-631Gd': 'COMP6-v2-wB97X-631Gd-data.tar.gz',
                'B973c-def2mTZVP': 'COMP6-v2-B973c-def2mTZVP-data.tar.gz'}
    _FILES_AND_MD5S = {'wB97X-631Gd': OrderedDict([('ANI-BenchMD-wB97X-631Gd.h5', '04c03ec8796359a0e3eb301346efbb03'),
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
                       'B973c-def2mTZVP': OrderedDict([('COMP6-full_v1-B97c3-def2mTZVP.h5', '044556f8490cc9e92975b949c0da5099'),
                                                       ('COMP6-heavy-B97c3-def2mTZVP.h5', '425d73d6a1c14c5897907b415e6f7f92')])}

    def __init__(self, root: StrPath, download: bool = False, verbose: bool = True, basis_set='631Gd', functional='wB97X'):
        lot = f'{functional}-{basis_set}'
        if lot not in self._ARCHIVE.keys():
            raise ValueError(f"Unsupported functional-basis set combination, try one of {set(self._ARCHIVE.keys())}")
        super().__init__(root, download, archive=self._ARCHIVE[lot], files_and_md5s=self._FILES_AND_MD5S[lot], verbose=verbose)
