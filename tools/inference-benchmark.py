import typing as tp
import sys
from pathlib import Path
import argparse
import time

import torch
from tqdm import tqdm

from torchani.models import ANI1x
from torchani.grad import energies_and_forces
from torchani.io import read_xyz


def main(filename: str, use_tqdm: bool, device: tp.Literal["cpu", "cuda"]) -> int:
    if not filename:
        xyz_file_path = Path(
            Path(__file__).parent.parent, "dataset", "xyz_files", "CH4-5.xyz"
        )
    elif filename.startswith("/"):
        xyz_file_path = Path(filename)
    else:
        xyz_file_path = Path.cwd() / filename

    model = ANI1x()[0].to(device)
    species, coordinates, _ = read_xyz(xyz_file_path, device=device)
    num_conformations = species.shape[0]
    print(f"Num conformations: {num_conformations}")
    start = time.perf_counter()
    energies_and_forces(model, species, coordinates)
    print(f"Time for energies and forces [batch]: {time.perf_counter() - start}")
    start = time.perf_counter()
    for species, coordinates in tqdm(
        zip(species, coordinates),
        disable=not use_tqdm,
        total=num_conformations,
        leave=False,
    ):
        _, _ = energies_and_forces(
            model,
            species.unsqueeze(0),
            coordinates.unsqueeze(0).detach(),
        )
    print(f"Time for energies and forces [batch]: {time.perf_counter() - start}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Path to xyz file", default="", nargs="?")
    parser.add_argument(
        "-d",
        "--device",
        help="Device for modules and tensors",
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument(
        "--tqdm",
        dest="use_tqdm",
        action="store_true",
        help="Whether to use tqdm to display progress",
    )
    args = parser.parse_args()
    sys.exit(main(args.filename, args.use_tqdm, args.device))
