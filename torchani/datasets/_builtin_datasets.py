r"""Torchani Builtin Datasets"""
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


class ANI1x(_BaseBuiltinDataset):
    _ARCHIVE = 'ANI-1x-wB97X-6-31Gd-data.tar.gz'
    _FILES_AND_MD5S = OrderedDict([('ANI-1x-wB97X-6-31Gd.h5', 'c9d63bdbf90d093db9741c94d9b20972')])

    def __init__(self, root: StrPath, download: bool = False, verbose: bool = True):
        super().__init__(root, download, archive=self._ARCHIVE, files_and_md5s=self._FILES_AND_MD5S, verbose=verbose)


class ANI2x(_BaseBuiltinDataset):
    _ARCHIVE = 'ANI-2x-wB97X-6-31Gd-data.tar.gz'
    _FILES_AND_MD5S = OrderedDict([('ANI-1x-wB97X-6-31Gd.h5', 'c9d63bdbf90d093db9741c94d9b20972'),
                                   ('ANI-2x-heavy-wB97X-6-31Gd.h5', '49ec3dc5d046f5718802f5d1f102391c'),
                                   ('ANI-2x-dimers-wB97X-6-31Gd.h5', '3455d82a50c63c389126b68607fb9ca8')])

    def __init__(self, root: StrPath, download: bool = False, verbose: bool = True):
        super().__init__(root, download, archive=self._ARCHIVE, files_and_md5s=self._FILES_AND_MD5S, verbose=verbose)


class COMP6v1(_BaseBuiltinDataset):
    _ARCHIVE = 'COMP6-v1-data.tar.gz'
    _FILES_AND_MD5S = OrderedDict([('GDB11-07-test-500.h5', '9200755bfc755405e64100a53a9f7468'),
                                   ('GDB11-08-test-500.h5', '202b078f98a911a7a9bdc21ee0ae1af7'),
                                   ('GDB11-09-test-500.h5', '5d2f6573c07e01493e4c7f72edabe483'),
                                   ('GDB11-10-test-500.h5', '96acd0003f6faeacb51b4db483c1d6f8'),
                                   ('GDB11-11-test-500.h5', 'b7bf4fa7d2f78b8168f243b1a6aa6071'),
                                   ('GDB13-12-test-1000.h5', '4317beed9425ee63659e41144475115c'),
                                   ('GDB13-13-test-1000.h5', '4095ae8981a5e4b10fbc1f29669b0af5'),
                                   ('DrugBank-Testset.h5', 'fae59730172c7849478271dbf585c8ce'),
                                   ('DrugBank-Testset-SFCl.h5', 'dca0987a6030feca5b8e9a1e24102b44'),
                                   ('Tripeptides-Full.h5', 'bb7238f3634217e834b7eee94febc816'),
                                   ('ANI-MD-Bench.h5', '9e3a1327d01730033edeeebd6fac4d6c'),
                                   ('S66-x8-wB97X-6-31Gd.h5', 'df1a5f3b9b6599d56f1a78631a83b720')])

    def __init__(self, root: StrPath, download: bool = False, verbose: bool = True):
        super().__init__(root, download, archive=self._ARCHIVE, files_and_md5s=self._FILES_AND_MD5S, verbose=verbose)
