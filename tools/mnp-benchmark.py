import sys
import os
import typing as tp
from pathlib import Path
import itertools

import torch
from rich.table import Table
from rich.console import Console

from torchani.annotations import Device
from torchani.io import read_xyz
from torchani.csrc import CUAEV_IS_INSTALLED, MNP_IS_INSTALLED
from torchani.arch import ANI
from torchani.models import ANI2x
from torchani.grad import energies_and_forces

from tool_utils import timeit

ROOT = Path(__file__).resolve().parent.parent

if not MNP_IS_INSTALLED:
    raise RuntimeError("MNP is not installed, can't run benchmark")


def _build_ani2x(
    idx: tp.Optional[int] = None,
    mnp: bool = False,
    infer: bool = False,
    jit: bool = False,
    device: Device = None,
):
    device = torch.empty(0, device=device).device
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


def benchmark(
    table: Table,
    jit: bool = False,
    device: Device = None,
    idx: tp.Optional[int] = None,
) -> None:
    """
    Sample benchmark result on 2080 Ti
    cuda:
        run_ani2x                          : 21.739 ms/step
        run_ani2x_infer                    : 9.630 ms/step
    cpu:
        run_ani2x                          : 756.459 ms/step
        run_ani2x_infer                    : 32.482 ms/step
    """
    device = torch.empty(0, device=device).device

    def _run(model, file):
        species, coords, _, _ = read_xyz(
            Path(ROOT, "tests", "resources", file),
            device=device,
            dtype=torch.float,
        )
        _, _ = energies_and_forces(model, species, coords)

    steps = 10 if device.type == "cpu" else 30

    def run():
        _run(ani2x, "small.xyz")

    ani2x = _build_ani2x(idx=idx, device=device, jit=jit)
    time_ms_ref = timeit(run, steps=steps, verbose=False)
    table.add_row(
        device.type,
        "--",
        str(jit),
        str((device.type == "cuda") and CUAEV_IS_INSTALLED),
        f"{time_ms_ref:.5f}",
        "--",
    )
    if not (jit and idx == 0):

        def run_infer():
            _run(ani2x_infer, "small.xyz")

        # This combination is not supported
        ani2x_infer = _build_ani2x(
            idx=idx, mnp=False, infer=True, device=device, jit=jit
        )
        time_ms = timeit(run_infer, steps=steps, verbose=False)
        color = "green" if time_ms < time_ms_ref else "red"
        table.add_row(
            device.type,
            "pyMNP" if idx is not None else "BMM",
            str(jit),
            str((device.type == "cuda") and CUAEV_IS_INSTALLED),
            f"{time_ms:.5f}",
            f"[{color}]{((time_ms_ref / time_ms) - 1) * 100:.2f}[/{color}]",
        )
    ani2x_infer_mnp = _build_ani2x(
        idx=idx, mnp=True, infer=True, device=device, jit=jit
    )

    def run_infer_mnp():
        _run(ani2x_infer_mnp, "small.xyz")

    time_ms = timeit(run_infer_mnp, steps=steps, verbose=False)
    color = "green" if time_ms < time_ms_ref else "red"
    table.add_row(
        device.type,
        "cppMNP" if idx is not None else "cppMNP+BMM",
        str(jit),
        str((device.type == "cuda") and CUAEV_IS_INSTALLED),
        f"{time_ms:.5f}",
        f"[{color}]{((time_ms_ref / time_ms) - 1) * 100:.2f}[/{color}]",
    )


def main() -> int:
    os.environ["OMP_NUM_THREADS"] = "2"
    # Disable Tensorfloat, errors between two run of same model for large system
    # could reach 1e-3. However note that this error for large system is not that
    # big actually.
    torch.backends.cuda.matmul.allow_tf32 = False
    console = Console()
    ensemble_table = Table(
        title="MNP Ensemble Benchmark",
        box=None,
    )
    single_table = Table(
        title="MNP Single Model Benchmark",
        box=None,
    )
    for table in (single_table, ensemble_table):
        table.add_column("device", style="magenta")
        table.add_column("strategy", style="blue")
        table.add_column("JIT", style="blue")
        table.add_column("cuAEV", style="blue")
        table.add_column("time (ms)", style="cyan")
        table.add_column("speedup (%)")
    for jit, device in itertools.product(
        (True, False),
        (torch.device("cpu"), torch.device("cuda")),
    ):
        console.print(f"Profiling Ensemble in {device.type}, JIT: {jit}")
        benchmark(ensemble_table, jit, device, idx=None)
    console.print(ensemble_table)
    console.print()

    for jit, device in itertools.product(
        (True, False),
        (torch.device("cpu"), torch.device("cuda")),
    ):
        console.print(f"Profiling Single Model in {device.type}, JIT: {jit}")
        benchmark(single_table, jit, device, idx=0)
    console.print(single_table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
