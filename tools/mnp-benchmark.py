import sys
import json
import time
import argparse
import typing as tp
from pathlib import Path

import torch
import numpy as np
from rich.table import Table
from rich.console import Console

import torchani
from torchani.io import read_xyz
from torchani.csrc import CUAEV_IS_INSTALLED, MNP_IS_INSTALLED
from torchani.arch import ANI
from torchani.models import ANI2x
from torchani.grad import energies_and_forces

ROOT = Path(__file__).resolve().parent.parent

if not MNP_IS_INSTALLED:
    raise RuntimeError("MNP is not installed, can't run benchmark")


def _build_ani2x(
    device: torch.device,
    idx: int | None,
    jit: bool,
    infer: bool,
    mnp: bool,
) -> ANI:
    if device.type == "cuda" and CUAEV_IS_INSTALLED:
        strat = "cuaev-fused"
    else:
        strat = "pyaev"
    model = ANI2x(model_index=idx, strategy=strat)
    if infer:
        model = model.to_infer_model(mnp)
    model.to(device=device)
    if jit:
        model = tp.cast(ANI, torch.jit.script(model))
    return model


def _run_steps(model: ANI, species: torch.Tensor, coords: torch.Tensor, steps: int):
    for _ in range(steps):
        energies_and_forces(model, species, coords)


def _time_sample(
    model: ANI, species: torch.Tensor, coords: torch.Tensor, steps: int
) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    _run_steps(model, species, coords, steps)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / steps


def _variants(idx: int | None, jit: bool) -> list[tuple[str, bool, bool]]:
    # (label, infer, mnp)
    variants = [("no-MNP", False, False)]
    # pyMNP can't be scripted
    if not (jit and idx == 0):
        variants.append(("pyMNP" if idx == 0 else "BMM", True, False))
    variants.append(("cppMNP" if idx == 0 else "cppMNP+BMM", True, True))
    return variants


def benchmark(
    device: torch.device,
    idx: int | None,
    jit: bool,
    samples: int,
    steps: int,
    warmup: int,
) -> list[dict[str, tp.Any]]:
    species, coords, _, _ = read_xyz(
        ROOT / "tests" / "resources" / "small.xyz", device=device, dtype=torch.float
    )
    variants = _variants(idx, jit)
    models = [_build_ani2x(device, idx, jit, infer, mnp) for _, infer, mnp in variants]

    ref_energy = energies_and_forces(models[0], species, coords).energies
    for model in models:
        energy = energies_and_forces(model, species, coords).energies
        if not torch.allclose(energy, ref_energy, atol=1e-4, rtol=0):
            raise RuntimeError("Energies differ between variants")
        _run_steps(model, species, coords, warmup)

    # Samples of the variants are interleaved, so that drift (e.g. GPU clocks or
    # thermal throttling) affects all of them equally
    times_ms: list[list[float]] = [[] for _ in variants]
    for _ in range(samples):
        for j, model in enumerate(models):
            times_ms[j].append(_time_sample(model, species, coords, steps))

    rows = []
    for (label, _, _), t in zip(variants, times_ms):
        rows.append(
            {
                "model": "ensemble" if idx is None else "single",
                "device": device.type,
                "jit": jit,
                "strategy": label,
                "median_ms": float(np.median(t)),
                "p25_ms": float(np.percentile(t, 25)),
                "p75_ms": float(np.percentile(t, 75)),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Call torch.set_num_threads with this value",
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--cuda-steps", type=int, default=20)
    parser.add_argument("--cpu-steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--json", type=Path, default=None, help="Dump results here")
    args = parser.parse_args()

    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    # Disable Tensorfloat, errors between two run of same model for large system
    # could reach 1e-3. However note that this error for large system is not that
    # big actually.
    torch.backends.cuda.matmul.allow_tf32 = False

    console = Console()
    console.print(f"torchani: {torchani.__file__}")
    console.print(f"torch.get_num_threads(): {torch.get_num_threads()}")

    rows = []
    for idx in (None, 0):
        for device in (torch.device("cpu"), torch.device("cuda")):
            for jit in (True, False):
                if device.type == "cuda":
                    steps = args.cuda_steps
                else:
                    steps = args.cpu_steps
                rows.extend(
                    benchmark(device, idx, jit, args.samples, steps, args.warmup)
                )

    table = Table(title="MNP Benchmark", box=None)
    for col in ("model", "device", "JIT", "strategy", "median (ms)", "IQR (ms)"):
        table.add_column(col)
    table.add_column("speedup vs no-MNP")
    ref_ms = 0.0
    for row in rows:
        if row["strategy"] == "no-MNP":
            ref_ms = row["median_ms"]
        table.add_row(
            row["model"],
            row["device"],
            str(row["jit"]),
            row["strategy"],
            f"{row['median_ms']:.3f}",
            f"{row['p25_ms']:.3f}-{row['p75_ms']:.3f}",
            f"{ref_ms / row['median_ms']:.2f}x",
        )
    console.print(table)
    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
