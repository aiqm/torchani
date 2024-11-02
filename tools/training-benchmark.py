import typing as tp
import sys
import argparse

import torch
from tqdm import tqdm
from rich.console import Console

from torchani import datasets
from torchani.datasets import batch_all_in_ram
from torchani.models import ANI1x, ANI
from tool_utils import Timer, Opt

console = Console()


def main(
    opt: Opt,
    sync: bool,
    nvtx: bool,
    device: str,
    no_tqdm: bool,
    batch_size: int,
    num_profile: int,
    num_warm_up: int,
    dataset: str,
    detail: bool,
) -> int:
    console.print(
        f"Profiling with optimization={opt.value}, on device: {device.upper()}"
    )
    detail = (opt is Opt.NONE) and detail
    model = ANI1x(model_index=0).to(device)
    model.requires_grad_(True)
    if opt is Opt.JIT:
        model = tp.cast(ANI, torch.jit.script(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    mse = torch.nn.MSELoss(reduction="none")

    try:
        ds = getattr(datasets, args.dataset)(verbose=False)
    except AttributeError:
        raise RuntimeError(f"Dataset {args.dataset} could not be found")
    batched_ds = batch_all_in_ram(ds, args.batch_size, verbose=False)
    train = torch.utils.data.DataLoader(batched_ds, batch_size=None, shuffle=True)

    counter = 0
    timer = Timer(
        modules_and_fns=(
            [
                (model, "forward"),
                (model.aev_computer, "forward"),
                (model.neural_networks, "forward"),
                (model.energy_shifter, "forward"),
                (model.aev_computer.neighborlist, "forward"),
                (model.aev_computer.angular, "forward"),
                (model.aev_computer.radial, "forward"),
            ]
            if detail
            else []
        ),
        nvtx=nvtx,
        sync=sync,
    )
    total_batches = num_warm_up + num_profile
    pbar = tqdm(desc="Warm up", total=total_batches, leave=False, disable=no_tqdm)
    while True:
        for properties in train:
            if not timer.is_profiling and (counter == num_warm_up):
                pbar.set_description("Profiling")
                timer.start_profiling()
            pbar.update(1)

            timer.start_range("prepare-batch")
            properties = {k: v.to(device) for k, v in properties.items()}
            species = properties["species"]
            coordinates = properties["coordinates"].to(dtype=torch.float)
            targ_energies = properties["energies"].to(dtype=torch.float)
            num_atoms = (species >= 0).sum(dim=1, dtype=torch.float)
            timer.end_range("prepare-batch")

            timer.start_range("loss-fw")
            with torch.autograd.profiler.emit_nvtx(
                enabled=(timer.is_profiling and nvtx), record_shapes=True
            ):
                pred_energies = model((species, coordinates)).energies
            loss = (mse(pred_energies, targ_energies) / num_atoms.sqrt()).mean()
            timer.end_range("loss-fw")

            timer.start_range("loss-bw")
            loss.backward()
            timer.end_range("loss-bw")

            timer.start_range("optimizer")
            optimizer.step()
            timer.end_range("optimizer")
            counter += 1
            if counter == total_batches:
                break
        if counter == total_batches:
            break
    pbar.close()
    timer.stop_profiling()
    timer.display()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        help="Name of builtin dataset to train on",
        nargs="?",
        default="TestData",
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
        help="Number of warm up batches",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-e",
        "--num-profile",
        help="Number of profiling batches",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        help="Number of conformations of each batch",
        default=2560,
        type=int,
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
        help="Whether to include benchmark detail",
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
        opt=Opt.NONE,
        sync=sync,
        nvtx=args.nvtx,
        device=args.device,
        batch_size=args.batch_size,
        dataset=args.dataset,
        num_warm_up=args.num_warm_up,
        num_profile=args.num_profile,
        detail=args.detail,
        no_tqdm=args.no_tqdm,
    )
    main(
        opt=Opt.JIT,
        sync=sync,
        nvtx=args.nvtx,
        device=args.device,
        batch_size=args.batch_size,
        dataset=args.dataset,
        num_warm_up=args.num_warm_up,
        num_profile=args.num_profile,
        detail=args.detail,
        no_tqdm=args.no_tqdm,
    )
    sys.exit(0)
