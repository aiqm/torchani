import typing as tp
import sys
from pathlib import Path
import argparse

import torch
from rich.console import Console
from tqdm import tqdm

from torchani.models import ANI1x
from torchani.grad import energies_and_forces
from torchani.io import read_xyz

from tool_utils import Timer

console = Console()
ROOT = Path(__file__).resolve().parent.parent


def main(
    file: str,
    nvtx: bool,
    sync: bool,
    no_tqdm: bool,
    device: tp.Literal["cpu", "cuda"],
) -> int:
    if not file:
        xyz_file_path = Path(ROOT, "tests", "test_data", "CH4-5.xyz")
    elif file.startswith("/"):
        xyz_file_path = Path(file)
    else:
        xyz_file_path = Path.cwd() / file

    model = ANI1x()[0].to(device)
    species, coordinates, _ = read_xyz(xyz_file_path, device=device)
    num_conformations = species.shape[0]
    timer = Timer(
        modules=[
            model,
            model.aev_computer,
            model.neural_networks,
            model.energy_shifter,
            model.aev_computer.neighborlist,
            model.aev_computer.angular_terms,
            model.aev_computer.radial_terms,
        ],
        device=device,
        nvtx=nvtx,
        sync=sync,
        extra_title="Batch",
    )
    console.print(f"Profiling on {num_conformations} conformations")
    timer.start_profiling()
    energies_and_forces(model, species, coordinates)
    timer.stop_profiling()
    timer.display()

    model = ANI1x()[0].to(device)
    timer = Timer(
        modules=[
            model,
            model.aev_computer,
            model.neural_networks,
            model.energy_shifter,
            model.aev_computer.neighborlist,
            model.aev_computer.angular_terms,
            model.aev_computer.radial_terms,
        ],
        device=device,
        nvtx=nvtx,
        sync=sync,
        extra_title="Single Molecule",
        reduction="sum",
    )
    timer.start_profiling()
    for species, coordinates in tqdm(
        zip(species, coordinates),
        desc="Inference on single molecules",
        disable=no_tqdm,
        total=num_conformations,
        leave=False,
    ):
        _, _ = energies_and_forces(
            model,
            species.unsqueeze(0),
            coordinates.unsqueeze(0).detach(),
        )
    timer.stop_profiling()
    timer.display()
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
        "--no-tqdm",
        dest="no_tqdm",
        action="store_true",
        help="Whether to disable tqdm to display progress",
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Whether to emit nvtx for NVIDIA Nsight systems",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Whether to disable sync between CUDA calls",
    )
    args = parser.parse_args()
    if args.nvtx and not torch.cuda.is_available():
        raise ValueError("CUDA is needed to profile with NVTX")
    sync = False
    if args.device == "cuda" and not args.no_sync:
        sync = True
    console.print(
        f"NVTX {'[green]ENABLED[/green]' if args.nvtx else '[red]DISABLED[/red]'}"
    )
    console.print(
        f"CUDA sync {'[green]ENABLED[/green]' if sync else '[red]DISABLED[/red]'}"
    )
    sys.exit(
        main(
            args.filename,
            nvtx=args.nvtx,
            sync=sync,
            no_tqdm=args.no_tqdm,
            device=args.device,
        )
    )
