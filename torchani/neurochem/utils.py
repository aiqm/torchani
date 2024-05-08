from pathlib import Path


def model_dir_from_prefix(prefix: Path, idx: int) -> Path:
    network_path = (prefix.parent / f"{prefix.name}{idx}") / "networks"
    return network_path
