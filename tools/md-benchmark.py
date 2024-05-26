import argparse
import sys
from pathlib import Path

import torch
import ase
import ase.io
import ase.md
from rich.console import Console

from torchani.models import ANI1x
from tool_utils import Timer

console = Console()
ROOT = Path(__file__).resolve().parent.parent


def main(
    sync: bool,
    nvtx: bool,
    device: str,
    file: str,
    num_warm_up: int,
    num_profile: int,
) -> int:
    if not file:
        xyz_file_path = Path(ROOT, "tests", "test_data", "small.xyz")
    elif file.startswith("/"):
        xyz_file_path = Path(file)
    else:
        xyz_file_path = Path.cwd() / file

    molecule = ase.io.read(str(xyz_file_path))
    model = ANI1x(model_index=0).to(torch.device(device))
    molecule.calc = model.ase()
    dyn = ase.md.verlet.VelocityVerlet(molecule, timestep=1 * ase.units.fs)

    timer = Timer(
        modules=[
            model,
            model.aev_computer,
            model.aev_computer.neighborlist,
            model.aev_computer.angular_terms,
            model.aev_computer.radial_terms,
            model.neural_networks,
            model.energy_shifter,
        ],
        device=device,
        nvtx=nvtx,
        sync=sync,
    )
    console.print(f"Warm up for {num_warm_up} steps")
    dyn.run(num_warm_up)
    console.print(f"Profiling for {num_profile} steps")
    timer.start_profiling()
    with torch.autograd.profiler.emit_nvtx(enabled=nvtx, record_shapes=True):
        dyn.run(num_profile)
    timer.stop_profiling()
    timer.display()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file",
        help="Path to xyz file to use as input for MD",
        nargs="?",
        default="",
    )
    parser.add_argument(
        "-d",
        "--device",
        help="Device of modules and tensors",
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument(
        "-w",
        "--num-warm-up",
        help="Number of warm up steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-e",
        "--num-profile",
        help="Number of profiling steps",
        type=int,
        default=100,
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
    if args.device == "cuda":
        console.print(
            f"NVTX {'[green]ENABLED[/green]' if args.nvtx else '[red]DISABLED[/red]'}"
        )
        console.print(
            f"CUDA sync {'[green]ENABLED[/green]' if sync else '[red]DISABLED[/red]'}"
        )
    sys.exit(
        main(
            sync=sync,
            nvtx=args.nvtx,
            device=args.device,
            file=args.file,
            num_warm_up=args.num_warm_up,
            num_profile=args.num_profile,
        )
    )
