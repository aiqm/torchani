from enum import Enum
from typer import Option, Argument, Typer
import typing_extensions as tpx
import typing as tp
import sys
from pathlib import Path

from rich.console import Console
from tqdm import tqdm


console = Console()
ROOT = Path(__file__).resolve().parent.parent


app = Typer()


class Opt(Enum):
    JIT = "jit"
    NONE = "none"
    COMPILE = "compile"


@app.command()
def cmd(
    file: tpx.Annotated[
        tp.Optional[Path],
        Argument(help=".xyz file to use for the benchmark", show_default=False),
    ] = None,
    optims: tpx.Annotated[
        tp.Optional[tp.List[Opt]],
        Option(
            "-o",
            "--optim",
            help="Torch optimizations to perform. Default is all",
            show_default=False,
        ),
    ] = None,
    device_str: tpx.Annotated[
        str,
        Option("--device", help="Device to use for benchmark (cpu or cuda)"),
    ] = "cpu",
    nvtx: tpx.Annotated[
        bool,
        Option(
            "-n/-N",
            "--nvtx/--no-nvtx",
            help="Whether to emit nvtx for NVIDIA Nsight systems",
        ),
    ] = False,
    strategy: tpx.Annotated[
        str,
        Option("--strategy"),
    ] = "pyaev",
    neighborlist: tpx.Annotated[
        str,
        Option("--neighborlist"),
    ] = "all_pairs",
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
    detail: tpx.Annotated[
        bool,
        Option("-d/-D", "--detail/--no-detail", help="Detailed breakdown of benchmark"),
    ] = False,
    num_warm_up: tpx.Annotated[
        int,
        Option("-n", "--num-warm-up", help="Num of warm up steps"),
    ] = 50,
    num_profile: tpx.Annotated[
        int,
        Option("-n", "--num-profile", help="Num of profiling steps"),
    ] = 20,
    profile_batches: tpx.Annotated[
        bool,
        Option(
            "-b/-B",
            "--profile-batches/--no-profile-batches",
            help="Whether to profile batched molecules",
        ),
    ] = True,
) -> None:
    import torch
    from torchani.arch import ANI
    from torchani.models import ANI1x
    from torchani.grad import energies_and_forces
    from torchani.io import read_xyz

    from tool_utils import Timer

    assert strategy in ["pyaev", "cuaev", "cuaev-fused"]
    assert neighborlist in ["all_pairs", "cell_list", "adaptive"]
    neighborlist = tp.cast(
        tp.Literal["all_pairs", "cell_list", "adaptive"], neighborlist
    )

    if optims is None:
        optims = [Opt.NONE, Opt.JIT, Opt.COMPILE]

    no_tqdm = not use_tqdm
    device = torch.device(device_str)

    if nvtx and device.type != "cuda":
        console.print("NVTX can only be enabled on cuda device", style="red")
        sys.exit(1)
    if device.type == "cuda":
        console.print(
            f"NVTX {'[green]ENABLED[/green]' if nvtx else '[red]DISABLED[/red]'}"
        )
        console.print(
            f"CUDA sync {'[green]ENABLED[/green]' if sync else '[red]DISABLED[/red]'}"
        )
    if file is None:
        xyz_file_path = Path(ROOT, "tests", "resources", "CH4-5.xyz")
    else:
        xyz_file_path = file

    # Set up required models for benchmark
    model = ANI1x(
        model_index=0, device=device, strategy=strategy, neighborlist=neighborlist
    )
    models: tp.Dict[Opt, ANI] = {}
    for opt in optims:
        if opt is Opt.NONE:
            models[opt] = model
        elif opt is Opt.JIT:
            models[opt] = tp.cast(ANI, torch.jit.script(model))
        elif opt is Opt.COMPILE:
            models[opt] = tp.cast(ANI, torch.compile(model))

    # Loop over molecules and calculate timings
    species, coordinates, _, _ = read_xyz(xyz_file_path, device=device)
    for opt, m in models.items():
        console.print(
            "Profiling energy *and* force."
            f" optim={opt.value}, device={device.type.upper()}, strat={strategy}"
        )
        num_conformations = species.shape[0]
        if detail and isinstance(m, torch.nn.Module):
            calls = [
                (model, "forward"),
                (model.aev_computer, "forward"),
                (model.neural_networks, "forward"),
                (model.energy_shifter, "forward"),
            ]
            if strategy == "pyaev":
                calls.extend(
                    [
                        (model.aev_computer.neighborlist, "forward"),
                        (model.aev_computer.angular, "forward"),
                        (model.aev_computer.radial, "forward"),
                    ]
                )
        else:
            calls = []
        timer = Timer(modules_and_fns=calls, nvtx=nvtx, sync=sync)
        for j, (_species, _coords) in tqdm(
            enumerate(zip(species[:num_warm_up], coordinates[:num_warm_up])),
            desc="Warm Up",
            disable=no_tqdm,
            total=num_warm_up,
            leave=False,
        ):
            _, _ = energies_and_forces(
                model,
                _species.unsqueeze(0),
                _coords.unsqueeze(0).detach(),
            )
        timer.start_profiling()
        for j, (_species, _coords) in tqdm(
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
                _coords.unsqueeze(0).detach(),
            )
            timer.end_range("batch-size-1")
        timer.stop_profiling()

        #  Cell List does not support batches
        if neighborlist != "cell_list" and profile_batches:
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
            timer.display()


if __name__ == "__main__":
    app()
