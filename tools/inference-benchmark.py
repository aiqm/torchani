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
    optimize: str,
    file: str,
    nvtx: bool,
    sync: bool,
    no_tqdm: bool,
    device: tp.Literal["cpu", "cuda"],
    detail: bool = False,
) -> int:
    console.print(
        f"Profiling with optimization={optimize}, on device: {device.upper()}"
    )
    if not file:
        xyz_file_path = Path(ROOT, "tests", "test_data", "CH4-5.xyz")
    elif file.startswith("/"):
        xyz_file_path = Path(file)
    else:
        xyz_file_path = Path.cwd() / file
    model = ANI1x()[0].to(device)
    if optimize == "jit":
        model = torch.jit.script(model)
    elif optimize == "compile":
        # Compile transforms the model into a Callable
        model = torch.compile(model)  # type: ignore
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
        ]
        if (optimize == "none" and detail)
        else [],
        device=device,
        nvtx=nvtx,
        sync=sync,
    )
    console.print(f"Batch of {num_conformations} conformations")
    for _ in tqdm(range(30), desc="Warm up", total=30, leave=False):
        energies_and_forces(model, species, coordinates)
    timer.start_profiling()
    for _ in tqdm(range(10), desc="Profiling", total=10, leave=False):
        timer.start_batch()
        energies_and_forces(model, species, coordinates)
        timer.end_batch()
    timer.stop_profiling()
    timer.display()

    model = ANI1x()[0].to(device)
    if optimize == "jit":
        model = torch.jit.script(model)
    elif optimize == "compile":
        # Compile transforms the model into a Callable
        model = torch.compile(model)  # type: ignore
    timer = Timer(
        modules=[
            model,
            model.aev_computer,
            model.neural_networks,
            model.energy_shifter,
            model.aev_computer.neighborlist,
            model.aev_computer.angular_terms,
            model.aev_computer.radial_terms,
        ]
        if (optimize == "none" and detail)
        else [],
        device=device,
        nvtx=nvtx,
        sync=sync,
    )
    console.print("Batch of 1 conformation")
    for j, (_species, _coordinates) in tqdm(
        enumerate(zip(species, coordinates)),
        desc="Warm Up",
        disable=no_tqdm,
        total=200,
        leave=False,
    ):
        _, _ = energies_and_forces(
            model,
            _species.unsqueeze(0),
            _coordinates.unsqueeze(0).detach(),
        )
        if j == 199:
            break
    timer.start_profiling()
    for j, (_species, _coordinates) in tqdm(
        enumerate(zip(species, coordinates)),
        desc="Profiling",
        disable=no_tqdm,
        total=100,
        leave=False,
    ):
        timer.start_batch()
        _, _ = energies_and_forces(
            model,
            _species.unsqueeze(0),
            _coordinates.unsqueeze(0).detach(),
        )
        timer.end_batch()
        if j == 100:
            break
    timer.stop_profiling()
    timer.display()
    console.print()
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
    parser.add_argument(
        "-c",
        "--compile",
        action="store_true",
        help="Whether to use torch.compile for optimization",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Detailed breakdown of benchmark",
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
    main(
        optimize="none",
        file=args.filename,
        nvtx=args.nvtx,
        sync=sync,
        no_tqdm=args.no_tqdm,
        device=args.device,
        detail=args.detail,
    )
    main(
        optimize="jit",
        file=args.filename,
        nvtx=args.nvtx,
        sync=sync,
        no_tqdm=args.no_tqdm,
        device=args.device,
    )
    if args.compile:
        if not tuple(map(int, torch.__version__.split("."))) >= (2, 0):
            raise RuntimeError("PyTorch 2.0 or later needed for torch.compile")
        main(
            optimize="compile",
            file=args.filename,
            nvtx=args.nvtx,
            sync=sync,
            no_tqdm=args.no_tqdm,
            device=args.device,
        )
    sys.exit(0)
