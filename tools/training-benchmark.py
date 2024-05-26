import sys
import argparse

import torch
from tqdm import tqdm
from rich.console import Console

from torchani import datasets
from torchani.datasets import create_batched_dataset
from torchani.models import ANI1x
from tool_utils import Timer
console = Console()


def main(
    jit: bool,
    sync: bool,
    nvtx: bool,
    device: str,
    batch_size: int,
    num_profile: int,
    num_warm_up: int,
    dataset: str,
    detail: bool = False,
) -> int:
    model = ANI1x(model_index=0).to(device)
    if jit:
        model = torch.jit.script(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    mse = torch.nn.MSELoss(reduction="none")

    try:
        ds = getattr(datasets, args.dataset)(verbose=False)
    except AttributeError:
        raise RuntimeError(f"Dataset {args.dataset} could not be found")
    splits = create_batched_dataset(
        ds,
        splits={"training": 1.0},
        direct_cache=True,
        batch_size=args.batch_size,
        verbose=False,
    )
    train = torch.utils.data.DataLoader(
        splits["training"], batch_size=None, shuffle=True
    )

    counter = 0
    timer = Timer(
        modules=[
            model,
            model.aev_computer,
            model.neural_networks,
            model.energy_shifter,
            model.aev_computer.neighborlist,
            model.aev_computer.angular_terms,
            model.aev_computer.radial_terms,
        ] if detail else [],
        device=device,
        nvtx=nvtx,
        sync=sync,
    )
    total_batches = num_warm_up + num_profile
    pbar = tqdm(desc="Warm up", total=total_batches, leave=False)
    while True:
        for properties in train:
            properties = {k: v.to(device) for k, v in properties.items()}
            if not timer.is_profiling and (counter == num_warm_up):
                pbar.set_description("Profiling")
                timer.start_profiling()
            pbar.update(1)

            timer.start_batch()
            species = properties["species"]
            coordinates = properties["coordinates"].to(dtype=torch.float)
            targ_energies = properties["energies"].to(dtype=torch.float)
            num_atoms = (species >= 0).sum(dim=1, dtype=torch.float)
            with torch.autograd.profiler.emit_nvtx(
                enabled=(timer.is_profiling and nvtx), record_shapes=True
            ):
                pred_energies = model((species, coordinates)).energies
            loss = (mse(pred_energies, targ_energies) / num_atoms.sqrt()).mean()
            timer.start_loss()
            loss.backward()
            timer.end_loss()
            timer.end_batch()
            timer.start_opt()
            optimizer.step()
            timer.end_opt()
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
        default=100,
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
    main(
        jit=False,
        sync=sync,
        nvtx=args.nvtx,
        device=args.device,
        batch_size=args.batch_size,
        dataset=args.dataset,
        num_warm_up=args.num_warm_up,
        num_profile=args.num_profile,
        detail=args.detail,
    )
    main(
        jit=True,
        sync=sync,
        nvtx=args.nvtx,
        device=args.device,
        batch_size=args.batch_size,
        dataset=args.dataset,
        num_warm_up=args.num_warm_up,
        num_profile=args.num_profile,
        detail=False,
    )
    sys.exit(0)
