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
    groups = list(ds.keys())
    conformer = ds.get_numpy_conformers(groups[0], 0)
    key_max_len = max([len(k) for k in conformer.keys()]) + 3
    shapes = [str(list(conformer[k].shape)) for k in conformer.keys()]
    shape_max_len = max([len(s) for s in shapes]) + 3
    print('\nFirst Conformer Properties (Non-batched): ')
    for i, k in enumerate(conformer.keys()):
        print(f'  {k.ljust(key_max_len)} shape: {shapes[i].ljust(shape_max_len)} dtype: {conformer[k].dtype}')
