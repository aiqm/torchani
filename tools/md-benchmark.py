import argparse
import sys
from pathlib import Path

import torch
import ase
import ase.io
import ase.md
from rich.console import Console
from tqdm import tqdm

from torchani.models import ANI1x
from tool_utils import Timer, Opt

console = Console()
ROOT = Path(__file__).resolve().parent.parent


def main(
    detail: bool,
    opt: Opt,
    sync: bool,
    nvtx: bool,
    device: str,
    file: str,
    no_tqdm: bool,
    num_warm_up: int,
    num_profile: int,
) -> int:
    detail = (opt is Opt.NONE) and detail
    console.print(
        f"Profiling with optimization={opt.value}, on device: {device.upper()}"
    )
    if not file:
        xyz_file_path = Path(ROOT, "tests", "test_data", "small.xyz")
    elif file.startswith("/"):
        xyz_file_path = Path(file)
    else:
        xyz_file_path = Path.cwd() / file

    molecule = ase.io.read(str(xyz_file_path))
    model = ANI1x(model_index=0).to(torch.device(device))
    molecule.calc = model.ase(jit=opt is Opt.JIT)
    dyn = ase.md.verlet.VelocityVerlet(molecule, timestep=1 * ase.units.fs)

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
        dyn.run(1)
    timer.start_profiling()
    with torch.autograd.profiler.emit_nvtx(enabled=nvtx, record_shapes=True):
        for _ in tqdm(
            range(num_profile),
            desc="Profiling",
            total=num_profile,
            leave=False,
            disable=no_tqdm,
        ):
            timer.start_range("md-step")
            dyn.run(1)
            timer.end_range("md-step")
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
        default=10,
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
        "--detail",
        action="store_true",
        help="Whether to enable detailed profiling",
    )
    parser.add_argument(
        "--no-tqdm",
        dest="no_tqdm",
        action="store_true",
        help="Whether to disable tqdm to display progress",
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
        console.print()
    main(
        detail=args.detail,
        opt=Opt.NONE,
        sync=sync,
        nvtx=args.nvtx,
        device=args.device,
        file=args.file,
        num_warm_up=args.num_warm_up,
        num_profile=args.num_profile,
        no_tqdm=args.no_tqdm,
    )
    main(
        detail=args.detail,
        opt=Opt.JIT,
        sync=sync,
        nvtx=args.nvtx,
        device=args.device,
        file=args.file,
        num_warm_up=args.num_warm_up,
        num_profile=args.num_profile,
        no_tqdm=args.no_tqdm,
    )
    sys.exit(0)
