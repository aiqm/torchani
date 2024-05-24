import typing as tp
import time
import argparse

import torch
from tqdm import tqdm

from torchani import datasets
from torchani.datasets import create_batched_dataset
from torchani.models import ANI1x
from torchani.units import hartree2kcalpermol
from tool_utils import time_functions


def start_profiling(
    model: torch.nn.Module,
    timers: tp.Dict[str, float],
    sync: bool,
    nvtx: bool,
) -> None:
    time_functions(
        [
            ("forward", model.aev_computer.neighborlist),
            ("forward", model.aev_computer.angular_terms),
            ("forward", model.aev_computer.radial_terms),
            (
                (
                    "_compute_radial_aev",
                    "_compute_angular_aev",
                    "_compute_aev",
                    "_triple_by_molecule",
                    "forward",
                ),
                model.aev_computer,
            ),
            ("forward", model.neural_networks),
            ("forward", model.energy_shifter),
        ],
        timers,
        sync,
        nvtx=nvtx,
    )
    if nvtx:
        torch.cuda.cudart().cudaProfilerStart()


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
        default=50,
    )
    parser.add_argument(
        "-e",
        "--num-profile",
        help="Number of profiling batches",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Number of conformations of each batch",
        default=2560,
        type=int,
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Whether to use NVIDIA Nsight systems",
    )
    args = parser.parse_args()
    if args.nvtx and not torch.cuda.is_available():
        raise ValueError("Nvtx needs CUDA to be available")

    if args.device == "cuda":
        sync = True
        print("CUDA sync enabled between function calls")
    else:
        sync = False

    model = ANI1x(model_index=0).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    mse = torch.nn.MSELoss(reduction="none")

    # time these functions
    timers: tp.Dict[str, float] = {}

    print("=> loading dataset")
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

    print("=> starting training")
    total_batches = args.num_warm_up + args.num_profile
    pbar = tqdm(desc=f"Batch 0/{total_batches}", total=total_batches)
    counter = 0
    profiling = False
    while True:
        for properties in train:
            counter += 1
            pbar.set_description(f"Batch {counter}/{total_batches}")
            pbar.update()
            if not profiling and (counter > args.num_warm_up):
                start_profiling(model, timers, sync, args.nvtx)
                profiling = True
                start = time.perf_counter()

            if (profiling and args.nvtx):
                torch.cuda.nvtx.range_push(f"batch-{counter}")
            species = properties["species"].to(device=args.device)
            coordinates = properties["coordinates"].to(
                device=args.device, dtype=torch.float
            )
            true_energies = properties["energies"].to(
                device=args.device, dtype=torch.float
            )
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            with torch.autograd.profiler.emit_nvtx(
                enabled=(profiling and args.nvtx), record_shapes=True
            ):
                predicted_energies = model((species, coordinates)).energies
            loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
            rmse = (
                hartree2kcalpermol((mse(predicted_energies, true_energies)).mean())
                .detach()
                .cpu()
                .numpy()
            )
            if (profiling and args.nvtx):
                torch.cuda.nvtx.range_push("backward")
            loss.backward()
            if (profiling and args.nvtx):
                torch.cuda.nvtx.range_pop()
            if (profiling and args.nvtx):
                torch.cuda.nvtx.range_push("optimizer-step")
            optimizer.step()
            if (profiling and args.nvtx):
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_pop()
            if counter == total_batches:
                break
        if counter == total_batches:
            break
    stop = time.perf_counter()

    for k in timers.keys():
        timers[k] = timers[k] / args.num_profile
    total = (stop - start) / args.num_profile

    for k in timers:
        print(f"{k} - {timers[k]:.3e}s")
    print(f"Total epoch time - {total:.3e}s")
