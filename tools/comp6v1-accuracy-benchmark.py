import sys
import argparse
from pathlib import Path
import math
import torch
from tqdm import tqdm
from rich.console import Console

from torchani.annotations import Device
from torchani.models import ANI1x
from torchani import datasets
from torchani.datasets import ANIDataset
from torchani.units import hartree2kcalpermol
from torchani.grad import energies_and_forces

console = Console()


def main(
    subset: str,
    max_size: int,
    device: Device,
    no_tqdm: bool,
) -> int:
    device = torch.empty(0, device=device).device
    ds = getattr(datasets, "COMP6v1")(verbose=False)
    locations = [
        Path(p)
        for p in ds.store_locations
        if Path(p).name.lower().startswith(subset.lower())
    ]
    if not len(locations) == 1:
        raise ValueError(f"Subset {subset} could not be found")
    model = ANI1x(device=device)
    console.print(f"Benchmarking on subset {subset} on device {device.type.upper()}")
    ds = ANIDataset(locations=locations)
    count = 0
    energies_rmse = 0.0
    forces_rmse = 0.0
    energies_mae = 0.0
    forces_mae = 0.0
    pbar = tqdm(
        total=ds.num_conformers, desc="COMP6v1 benchmark", leave=False, disable=no_tqdm
    )
    for k, j, d in ds.chunked_items(max_size=max_size):
        size = d["species"].shape[0]
        count += size
        d = {k: v.to(device) for k, v in d.items()}
        pred_energies, pred_forces = energies_and_forces(
            model, d["species"], d["coordinates"]
        )
        energies_diff = (d["energies"] - pred_energies).abs()
        forces_diff = (d["energies"] - pred_energies).abs()
        energies_rmse += energies_diff.pow(2).sum()
        forces_rmse += forces_diff.pow(2).sum()
        energies_mae += energies_diff.sum()
        forces_mae += forces_diff.sum()
        pbar.update(size)
    pbar.close()

    energies_mae = energies_mae / count
    energies_rmse = math.sqrt(energies_rmse / count)

    forces_mae = forces_mae / 3 / count
    forces_rmse = math.sqrt(forces_rmse / 3 / count)
    console.print(f"Energy RMSE (kcal/mol): {hartree2kcalpermol(energies_rmse)}")
    console.print(f"Energy MAE (kcal/mol): {hartree2kcalpermol(energies_mae)}")
    console.print(f"Force RMSE (kcal/mol/ang): {hartree2kcalpermol(forces_rmse)}")
    console.print(f"Force MAE (kcal/mol/ang): {hartree2kcalpermol(forces_mae)}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "subset",
        help="COMP6v1 subset",
        nargs="?",
        type=str,
        default="S66x8",
    )
    parser.add_argument(
        "-s",
        "--max-size",
        type=int,
        default=100,
        help="Maximum number of conformations in each batch",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        help="Device of modules and tensors",
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument(
        "--no-tqdm",
        dest="no_tqdm",
        action="store_true",
        help="Whether to disable tqdm to display progress",
    )
    args = parser.parse_args()
    sys.exit(
        main(
            device=args.device,
            subset=args.subset,
            max_size=args.max_size,
            no_tqdm=args.no_tqdm,
        )
    )
