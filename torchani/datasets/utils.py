r"""
Internal module with utilities, not meant for users
"""
from pathlib import Path
import typing as tp
import hashlib

from tqdm import tqdm

from torchani.datasets.anidataset import ANIDataset
from torchani.utils import download_and_extract
from torchani.paths import datasets_dir

_BASE_URL = "http://moria.chem.ufl.edu/animodel/ground_truth_data/"


# The functions that instantiate datasets, located in the automatically
# generated _builtin.py file, use _builder, _check_files_integrity and
# _calc_file_md5
def _builder(
    archive: str,
    files_and_md5s: tp.Dict[str, str],
    dummy_properties: tp.Optional[tp.Dict[str, tp.Any]],
    download: bool,
    verbose: bool,
    skip_check: bool,
    name: str,
    suffix: str = ".h5",
) -> ANIDataset:
    root = (datasets_dir() / archive.replace(".tar.gz", "")).resolve()
    # If the dataset is not found we download it
    if download and ((not root.is_dir()) or (not any(root.glob(f"*{suffix}")))):
        download_and_extract(
            url=f"{_BASE_URL}{archive}",
            file_name=archive,
            dest_dir=root,
            verbose=verbose,
        )

    # Check for corruption and missing files
    _check_files_integrity(
        files_and_md5s,
        root,
        suffix,
        name,
        skip_hash_check=skip_check,
        verbose=verbose,
    )

    # Order dataset paths using the order given in "files and md5s"
    filenames_order = {
        Path(k).stem: j for j, k in enumerate(files_and_md5s.keys())
    }
    filenames_and_paths = sorted(
        [(p.stem, p) for p in sorted(root.glob(f"*{suffix}"))],
        key=lambda tup: filenames_order[tup[0]],
    )
    ds = ANIDataset(
        locations=(tup[1] for tup in filenames_and_paths),
        names=(tup[0] for tup in filenames_and_paths),
        verbose=verbose,
        dummy_properties=dummy_properties,
    )
    if verbose:
        print(ds)
    return ds


def _check_files_integrity(
    files_and_md5s: tp.Dict[str, str],
    root: Path,
    suffix: str = ".h5",
    name: str = "Dataset",
    skip_hash_check: bool = False,
    verbose: bool = True,
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
        raise RuntimeError(f"Dataset not found in path {str(root)}")
    if expected_file_names != present_file_names:
        raise RuntimeError(
            f"Wrong files found for dataset {name} in provided path,"
            f" expected {expected_file_names} but found {present_file_names}"
        )
    if skip_hash_check:
        return
    for f in tqdm(
        present_files,
        desc=f"Checking integrity of dataset {name}",
        disable=not verbose,
        leave=False,
    ):
        if _calc_file_md5(f) != files_and_md5s[f.name]:
            raise RuntimeError(
                f"All expected files for dataset {name}"
                f" were found but file {f.name} failed integrity check,"
                " your dataset is corrupted or has been modified"
            )


def _calc_file_md5(file_path: Path) -> str:
    _CHUNK_SIZE = 1024 * 32
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
