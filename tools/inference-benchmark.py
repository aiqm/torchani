import sys
from pathlib import Path
import argparse

import torch
from rich.console import Console
from tqdm import tqdm

from torchani.annotations import Device
from torchani.models import ANI1x
from torchani.grad import energies_and_forces
from torchani.io import read_xyz

from tool_utils import Timer, Opt

console = Console()
ROOT = Path(__file__).resolve().parent.parent


def main(
    opt: Opt,
    file: str,
    nvtx: bool,
    sync: bool,
    no_tqdm: bool,
    device: Device,
    detail: bool,
    num_warm_up: int,
    num_profile: int,
) -> int:
    device = torch.device(device)
    detail = (opt is Opt.NONE) and detail
    console.print(
        f"Profiling with optimization={opt.value}, on device: {device.type.upper()}"
    )
    if not file:
        xyz_file_path = Path(ROOT, "tests", "test_data", "CH4-5.xyz")
    elif file.startswith("/"):
        xyz_file_path = Path(file)
    else:
        xyz_file_path = Path.cwd() / file
    model = ANI1x()[0].to(device)
    if opt is Opt.JIT:
        model = torch.jit.script(model)
    elif opt is Opt.COMPILE:
        # Compile transforms the model into a Callable
        model = torch.compile(model)  # type: ignore
    species, coordinates, _ = read_xyz(xyz_file_path, device=device)
    num_conformations = species.shape[0]
    timer = Timer(
        modules_and_fns=[
            (model, "forward"),
            (model.aev_computer, "forward"),
            (model.neural_networks, "forward"),
            (model.energy_shifter, "forward"),
            (model.aev_computer.neighborlist, "forward"),
            (model.aev_computer.angular_terms, "forward"),
            (model.aev_computer.radial_terms, "forward"),
        ]
        if detail
        else [],
        nvtx=nvtx,
        sync=sync,
    )

    for _ in tqdm(
        range(num_warm_up),
        desc="Warm up",
        total=num_warm_up,
        leave=False,
        disable=no_tqdm,
    ):
        energies_and_forces(model, species, coordinates)
    timer.start_profiling()
    for _ in tqdm(
        range(num_profile),
        desc="Profiling",
        total=num_profile,
        leave=False,
        disable=no_tqdm,
    ):
        timer.start_range(f"batch-size-{num_conformations}")
        energies_and_forces(model, species, coordinates)
        timer.end_range(f"batch-size-{num_conformations}")
    timer.stop_profiling()

    for j, (_species, _coordinates) in tqdm(
        enumerate(zip(species[:num_warm_up], coordinates[:num_warm_up])),
        desc="Warm Up",
        disable=no_tqdm,
        total=num_warm_up,
        leave=False,
    ):
        _, _ = energies_and_forces(
            model,
            _species.unsqueeze(0),
            _coordinates.unsqueeze(0).detach(),
        )
    timer.start_profiling()
    for j, (_species, _coordinates) in tqdm(
        enumerate(zip(species[:num_profile], coordinates[:num_profile])),
        desc="Profiling",
        disable=no_tqdm,
        total=num_profile,
        leave=False,
    ):
        timer.start_range("batch-size-1")
        _, _ = energies_and_forces(
            model,
            _species.unsqueeze(0),
            _coordinates.unsqueeze(0).detach(),
        )
        timer.end_range("batch-size-1")
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
    parser.add_argument(
        "-w",
        "--num-warm-up",
        help="Number of warm up steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-e",
        "--num-profile",
        help="Number of profiling steps",
        type=int,
        default=10,
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
    console.print()
    main(
        opt=Opt.NONE,
        file=args.filename,
        nvtx=args.nvtx,
        sync=sync,
        no_tqdm=args.no_tqdm,
        device=args.device,
        detail=args.detail,
        num_warm_up=args.num_warm_up,
        num_profile=args.num_profile,
    )
    main(
        opt=Opt.JIT,
        file=args.filename,
        nvtx=args.nvtx,
        sync=sync,
        no_tqdm=args.no_tqdm,
        device=args.device,
        detail=args.detail,
        num_warm_up=args.num_warm_up,
        num_profile=args.num_profile,
    )
    if args.compile:
        if not tuple(map(int, torch.__version__.split("."))) >= (2, 0):
            raise RuntimeError("PyTorch 2.0 or later needed for torch.compile")
        main(
            opt=Opt.COMPILE,
            file=args.filename,
            nvtx=args.nvtx,
            sync=sync,
            no_tqdm=args.no_tqdm,
            device=args.device,
            detail=args.detail,
            num_warm_up=args.num_warm_up,
            num_profile=args.num_profile,
        )
    sys.exit(0)
