import itertools
from typer import Option, Typer
import typing_extensions as tpx
import typing as tp
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from rich.console import Console
from tqdm import tqdm


console = Console()
ROOT = Path(__file__).resolve().parent.parent


app = Typer()


@app.command()
def run(
    cuda: tpx.Annotated[
        bool,
        Option("--cuda/--no-cuda", help="Use a CUDA enabled gpu for benchmark"),
    ] = True,
    sync: tpx.Annotated[
        bool,
        Option(
            "-s/-S",
            "--sync/--no-sync",
            help="Whether to sync cuda threads in between function calls",
        ),
    ] = True,
    use_tqdm: tpx.Annotated[
        bool,
        Option("-t/-T", "--tqdm/--no-tqdm", help="Whether to use a progress bar"),
    ] = True,
    num_warm_up: tpx.Annotated[
        int,
        Option("-n", "--num-warm-up", help="Num of warm up steps"),
    ] = 50,
    num_profile: tpx.Annotated[
        int,
        Option("-n", "--num-profile", help="Num of profiling steps"),
    ] = 20,
    pbc: tpx.Annotated[
        bool,
        Option("-p/-P", "--pbc/--no-pbc", help="Benchmark for the PBC case"),
    ] = True,
) -> None:
    import torch
    from torchani.neighbors import _parse_neighborlist
    from torchani._testing import make_molecs

    from tool_utils import Timer

    target_atomic_density = 0.1
    # num_atoms / cell_size has to be a constant, equal to the water atomic density
    # which is around ~0.1 Ang^-1

    no_tqdm = not use_tqdm
    device = torch.device("cuda" if cuda else "cpu")

    if cuda:
        console.print(f"CUDA sync {'[green]ON[/green]' if sync else '[red]OFF[/red]'}")

    # Set up required models for benchmark
    models = {
        "dummy": _parse_neighborlist("adaptive"),  # Only for general warm-up
        "all_pairs": _parse_neighborlist("all_pairs"),
        "cell_list": _parse_neighborlist("cell_list"),
        "adaptive": _parse_neighborlist("adaptive"),
    }

    # Loop over molecules and calculate timings
    for k, nl in models.items():
        console.print(f"Neighborlist kind={k}")
        timer = Timer(modules_and_fns=[], nvtx=False, sync=sync)
        atoms_num: tp.Iterable[int]
        if pbc:
            # Smaller than 25 atoms risks self-interaction
            atoms_num = range(25, 301)
        else:
            atoms_num = itertools.chain(
                range(30, 2500, 10),
                range(2500, 10000, 500),
            )
        cutoff = 5.2
        for n in map(int, atoms_num):
            cell_side = (n / target_atomic_density) ** (1 / 3)
            molecs = make_molecs(
                num_warm_up + num_profile,
                n,
                cell_side,
                pbc=pbc,
                device=tp.cast(tp.Literal["cpu", "cuda"], device.type),
                seed=1234,
            )
            slice_ = slice(None, num_warm_up)
            for j, (_species, _coordinates) in tqdm(
                enumerate(zip(molecs.atomic_nums[slice_], molecs.coords[slice_])),
                desc=f"Warm up {n} atoms, cell side length {cell_side}",
                disable=no_tqdm,
                total=num_warm_up,
                leave=False,
            ):
                _ = nl(
                    _species.unsqueeze(0),
                    _coordinates.unsqueeze(0),
                    cutoff=cutoff,
                    cell=molecs.cell if molecs.pbc.any() else None,
                    pbc=molecs.pbc,
                )
                if cuda:
                    torch.cuda.empty_cache()
            if k != "dummy":
                timer.start_profiling()
            slice_ = slice(num_warm_up, num_warm_up + num_profile)
            for j, (_species, _coordinates) in tqdm(
                enumerate(zip(molecs.atomic_nums[slice_], molecs.coords[slice_])),
                desc=f"Profiling {n} atoms, cell side length {cell_side}",
                disable=no_tqdm,
                total=num_profile,
                leave=False,
            ):
                timer.start_range(str(n))
                _ = nl(
                    _species.unsqueeze(0),
                    _coordinates.unsqueeze(0),
                    cutoff=cutoff,
                    cell=molecs.cell if molecs.pbc.any() else None,
                    pbc=molecs.pbc,
                )
                timer.end_range(str(n))
                if cuda:
                    torch.cuda.empty_cache()
            if k != "dummy":
                timer.stop_profiling()
                timer.dump_csv(
                    Path(__file__).parent / f"{k}{'-nopbc' if not pbc else ''}.csv"
                )
        timer.display()


@app.command()
def plot() -> None:
    fig, ax = plt.subplots()
    for k in (
        "cell_list",
        "all_pairs",
        "adaptive",
    ):
        csv = Path(__file__).parent / f"{k}.csv"
        if csv.is_file():
            df = pd.read_csv(csv, sep=",")
        ax.scatter(df["timing"], df["median"], label=k, s=3)
        ax.set_xlabel("Num. atoms")
        ax.set_ylabel("Walltime (ms)")
        ax.set_title("PBC benchmark")
        ax.legend()
    plt.show(block=False)

    fig, ax = plt.subplots()
    for k in (
        "cell_list-nopbc",
        "all_pairs-nopbc",
        "adaptive-nopbc",
    ):
        csv = Path(__file__).parent / f"{k}.csv"
        if csv.is_file():
            df = pd.read_csv(csv, sep=",")
        ax.scatter(df["timing"], df["median"], label=k.replace("-nopbc", ""), s=3)
        ax.set_xlabel("Num. atoms")
        ax.set_ylabel("Walltime (ms)")
        ax.set_title("No-PBC benchmark")
        ax.legend()
    plt.show()


if __name__ == "__main__":
    app()
