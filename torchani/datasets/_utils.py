from pathlib import Path
import json
import typing as tp
import hashlib

from tqdm import tqdm

from torchani.datasets.anidataset import ANIDataset
from torchani.utils import download_and_extract

_BASE_URL = "http://moria.chem.ufl.edu/animodel/ground_truth_data/"
_DATASETS_JSON_PATH = Path(__file__).parent / "builtin_datasets.json"


class DatasetIntegrityError(RuntimeError):
    pass


with open(_DATASETS_JSON_PATH, mode="rt", encoding="utf-8") as f:
    _DATASETS_SPEC = json.load(f)

# Convert csv file with format "file_name, MD5-hash" into a dictionary
_MD5S: tp.Dict[str, str] = dict()
with open(Path(__file__).resolve().parent / "md5s.csv") as f:
    lines = f.readlines()
    for line in lines[1:]:
        file_, md5 = line.split(",")
        _MD5S[file_.strip()] = md5.strip()


# TODO clean this up
def _available_archives() -> tp.Dict[str, tp.Tuple[str, str]]:
    ars: tp.Dict[str, tp.Tuple[str, str]] = {}
    for k, v in _DATASETS_SPEC.items():
        if k.endswith("-meta"):
            continue
        for lot in _available_dataset_lots(k):
            ars[v["lot"][lot]["archive"].split(".")[0]] = (k, lot)
    return ars


def _available_dataset_lots(ds_name: str) -> tp.List[str]:
    return list(_DATASETS_SPEC[ds_name]["lot"].keys())


def _default_dataset_lot(ds_name: str) -> str:
    return _DATASETS_SPEC[ds_name]["default-lot"]


# The functions that download and instantiate datasets, located in the
# automatically generated _builtin.py file, use
# _fetch_and_create_builtin_dataset, _check_files_integrity and _calc_file_md5
def _fetch_and_create_builtin_dataset(
    root: Path,
    ds_name: str,
    lot: str = "",
    verbose: bool = True,
    download: bool = True,
    dummy_properties: tp.Optional[tp.Dict[str, tp.Any]] = None,
    skip_check: bool = False,
    suffix: str = ".h5",
) -> ANIDataset:
    lot = lot.lower()
    if not lot:
        lot = _DATASETS_SPEC[ds_name]["default-lot"]
    lots = _DATASETS_SPEC[ds_name]["lot"]
    if lot not in lots:
        raise ValueError(f"Wrong LoT, supported are: {set(lots) - {'default-lot'}}")

    archive = lots[lot]["archive"]
    files_and_md5s = {k: _MD5S[k] for k in lots[lot]["files"]}
    dest_dir = (root / archive.replace(".tar.gz", "")).resolve()
    # If the dataset is not found we download it
    if download and ((not dest_dir.is_dir()) or (not any(dest_dir.glob(f"*{suffix}")))):
        download_and_extract(
            url=f"{_BASE_URL}{archive}",
            file_name=archive,
            dest_dir=dest_dir,
            verbose=verbose,
        )

    # Check for corruption and missing files
    _check_files_integrity(
        files_and_md5s,
        dest_dir,
        suffix,
        ds_name,
        skip_hash_check=skip_check,
        verbose=verbose,
    )

    # Order dataset paths using the order given in "files and md5s"
    filenames_order = {
        Path(k).stem: j for j, k in enumerate(files_and_md5s.keys())
    }
    filenames_and_paths = sorted(
        [(p.stem, p) for p in sorted(dest_dir.glob(f"*{suffix}"))],
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
        raise DatasetIntegrityError(f"Dataset not found in path {str(root)}")
    if expected_file_names != present_file_names:
        raise DatasetIntegrityError(
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
            raise DatasetIntegrityError(
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
