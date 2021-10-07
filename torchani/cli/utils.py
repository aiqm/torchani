from pathlib import Path
from ..datasets import ANIDataset


def h5info(path):
    path_ = Path(path)
    assert path_.exists(), f"{path} does not exist"
    if path_.is_file():
        files = [path_]
    else:
        files = list(Path(path_).glob("*.h5"))
        assert len(files) > 0, f"no h5 files found at {path_}"
    names = [f.stem for f in files]
    ds = ANIDataset(locations=files, names=names)
    print(ds)
