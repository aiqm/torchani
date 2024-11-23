import typing as tp
from pathlib import Path
import time
import pickle
import copy

import torch
import numpy as np
from numpy.typing import NDArray
import ase
from ase import units
from ase.md.langevin import Langevin
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

from torchani.models import ANI1x, ANI2x, ANI1ccx
from torchani.io import read_xyz


def plot_file(file_path, comment, show=False):
    with open(Path(file_path).resolve(), "rb") as f:
        times_sizes = pickle.load(f)
        all_trials = np.asarray(times_sizes["times"])
        sizes = times_sizes["atoms"]

    fig, ax = plt.subplots()
    std = all_trials.std(axis=0)
    mean = all_trials.mean(axis=0)
    assert len(std) == len(mean)
    assert len(std) == len(sizes)
    for times in all_trials:
        ax.errorbar(
            x=sizes, y=mean, yerr=std * 2, ecolor="k", capsize=2, fmt="s--", ms=1
        )
    ax.set_xlabel("System size (atoms)")
    ax.set_ylabel("Total Walltime per ns (days)")
    ax.set_title(comment)
    if show:
        plt.show()


def plot_many(path_to_files, comment, show=False):
    mpl.rc("font", size=16)
    file_paths = [
        f for f in Path(path_to_files).resolve().iterdir() if f.suffix == ".pkl"
    ]

    fig, ax = plt.subplots(1, len(file_paths), sharex=True, sharey=True)
    for j, p in enumerate(file_paths):
        with open(Path(p).resolve(), "rb") as f:
            times_sizes = pickle.load(f)
            sizes = times_sizes["atoms"]
            timers = times_sizes["timers"]
            all_trials = np.asarray(times_sizes["times"])
            total_times = all_trials.mean(axis=0)

        keys = timers[0].keys()
        for k in keys:
            values = np.asarray([timers[j][k] for j in range(len(sizes))])
            if k == "forward":
                forward = values
            if k == "backward":
                backward = values
            ax[j].plot(sizes, values, label=k)

        ax[j].plot(sizes, total_times, label="Total time")
        ax[j].plot(sizes, forward + backward, label="forward + backward")
        ax[j].legend()

        ax[j].set_xlabel("System size (atoms)")
        ax[j].set_ylabel("Total Walltime per ns (h)")
        ax[j].set_title(p.stem)
    if show:
        plt.show(block=False)

    fig, ax = plt.subplots(2, 1, sharex=True)
    for p in file_paths:
        with open(Path(p).resolve(), "rb") as f:
            times_sizes = pickle.load(f)
            all_trials = np.asarray(times_sizes["times"])
            sizes = times_sizes["atoms"]
            timers = times_sizes["timers"]

        std = all_trials.std(axis=0)
        mean = all_trials.mean(axis=0)
        inverse_mean = (1 / all_trials).mean(axis=0)
        inverse_std = (1 / all_trials).std(axis=0)
        assert len(std) == len(mean)
        assert len(std) == len(sizes)
        if "_clist_update_all_steps" in p.stem:
            c = "b"
            label = "TorchANI+pmemd (improved codebase)"
        elif "clist_reuse" in p.stem:
            c = "r"
            label = "TorchANI + cell list (Not Updating every step)"
        else:
            c = "g"
            label = "Original TorchANI"
        fmt = "s-"
        ax[0].errorbar(
            x=sizes,
            y=mean,
            color=c,
            yerr=std * 2,
            ecolor="k",
            capsize=2,
            fmt=fmt,
            ms=4,
            label=label,
        )
        ax[1].errorbar(
            x=sizes,
            y=inverse_mean,
            color=c,
            yerr=inverse_std * 2,
            ecolor="k",
            capsize=2,
            fmt=fmt,
            ms=4,
            label=label,
        )
    ax[0].set_xlabel("System size (atoms)")
    ax[0].set_ylabel("Walltime for 1 ns (days)")
    ax[0].legend()

    ax[1].set_xlabel("System size (atoms)")
    ax[1].set_ylabel("Performance, (ns/day)")

    fig.suptitle("Benchmarks for ANI-1x model, bulk water")

    if show:
        plt.show()


def get_model(model_arg, cell_list, model_index, verlet_cell_list):
    args = {"cell_list": cell_list, "verlet_cell_list": verlet_cell_list}

    if model_index:
        args.update({"model_index": model_index})

    if model_arg == "ani1x":
        model = ANI1x(**args).to(device, dtype=torch.double)
    elif model_arg == "ani2x":
        model = ANI2x(**args).to(device, dtype=torch.double)
    elif model_arg == "ani1ccx":
        model = ANI1ccx(**args).to(device, dtype=torch.double)
    return model.to(torch.float)


def print_info(device, steps, sizes):
    print(f"Running on {device} {torch.cuda.get_device_name()}")
    print(f"CUDA is avaliable: {torch.cuda.is_available()}")
    print(f"Running benchmark for {steps} steps")
    print(f"Running on the following sizes: {sizes}")


if __name__ == "__main__":
    import argparse

    # parse command line arguments
    parser = argparse.ArgumentParser(description="MD scaling benchmark for torchani")
    # generally good defaults
    parser.add_argument("-j", "--jit", action="store_true", default=False)
    parser.add_argument("-m", "--model", default="ani1x")
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument(
        "-s", "--steps", type=int, default=100, help="Timesteps to run in dynamics"
    )
    parser.add_argument(
        "--no-pbc",
        action="store_true",
        default=False,
        help="Use periodic boundary conditions",
    )
    parser.add_argument("--model-index", default=None, help="Specify a model index")

    # may want to change
    parser.add_argument(
        "-x", "--xyz", default=None, help="path to directory with xyz files"
    )
    parser.add_argument(
        "-t", "--trials", default=1, help="Repetitions to calculate std dev"
    )
    parser.add_argument(
        "-b",
        "--box-repeats",
        type=int,
        default=15,
        help="Number of replications of molecule in all directions",
    )
    parser.add_argument(
        "--cell-list", action="store_true", default=False, help="Use a cell list"
    )
    parser.add_argument(
        "--verlet-cell-list", action="store_true", default=False, help="Reuse cell list"
    )

    parser.add_argument("-p", "--plot", action="store_true", default=False)
    parser.add_argument("-f", "--file-name", default=None)
    parser.add_argument("-F", "--path-to-files", default=None)
    parser.add_argument("--no-show", action="store_false", default=True)

    args = parser.parse_args()
    show = args.no_show
    assert args.box_repeats > 3
    if args.xyz is None:
        raise ValueError("xyz is a required argument")
    path_to_xyz = Path(args.xyz).resolve()

    # the output file name is the model name by default
    if args.cell_list:
        clist_str = "_clist_update_all_steps"
    elif args.verlet_cell_list:
        clist_str = "_clist_reuse"
    else:
        clist_str = ""
    model_index_str = f"_network_{args.model_index}" if args.model_index else ""
    file_name = args.file_name or f"{args.model}{model_index_str}{clist_str}"

    root = Path(__file__).parent.resolve().joinpath("plots/")
    if not root.is_dir():
        root.mkdir()

    pickle_file = root.joinpath(f"{file_name}.pkl")
    csv_file = root.joinpath(f"{file_name}.csv")
    comment = "".join(s.capitalize() for s in file_name.split("_"))

    if args.plot or args.path_to_files:
        if not args.path_to_files:
            plot_file(pickle_file, comment, show)
        else:
            plot_many(args.path_to_files, comment, show)
    else:
        device = torch.device(args.device)
        sizes_list: tp.List[int] = []
        xyz_files: tp.Union[tp.List[Path], NDArray[tp.Any]]

        xyz_files = sorted(path_to_xyz.rglob("*.xyz"))
        for f in xyz_files:
            sizes_list.append(read_xyz(f)[0].shape[1])
        sizes = np.asarray(sizes_list)
        xyz_files = np.asarray(xyz_files)
        idx = np.argsort(sizes)
        sizes = sizes[idx]
        xyz_files = xyz_files[idx].tolist()

        print_info(device, args.steps, sizes)
        model = get_model(
            args.model, args.cell_list, args.model_index, args.verlet_cell_list
        )

        timers = {
            "forward": 0.0,
            "backward": 0.0,
            "neighborlist": 0.0,
            "aev_forward": 0.0,
        }

        def time_func(key, func):
            def wrapper(*args, **kwargs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                ret = func(*args, **kwargs)
                torch.cuda.synchronize()
                end = time.perf_counter()
                timers[key] += (end - start) / (3600 * 24) * 1e6 / 100
                return ret

            return wrapper

        if not args.jit:
            model.aev_computer._compute_pyaev = time_func(  # type: ignore
                "aev_forward", model.aev_computer._compute_pyaev
            )
            model.aev_computer.neighborlist.forward = time_func(  # type: ignore
                "neighborlist", model.aev_computer.neighborlist.forward
            )
            model.forward = time_func("forward", model.forward)  # type: ignore
        all_trials = []
        timers_list = []
        raw_trials = []
        for j in range(args.trials):
            times = []
            raw_times = []
            for r in tqdm(xyz_files):
                # reset timers
                timers = {k: 0.0 for k in timers}
                try:
                    model.aev_computer.neighborlist.reset_cached_values()
                except AttributeError:
                    pass
                species, coordinates, cell, _ = read_xyz(r)

                coordinates.requires_grad_()
                calc = model.ase()
                if args.jit:
                    torch._C._jit_set_profiling_executor(False)
                    torch._C._jit_set_profiling_mode(False)  # this also has an effect
                    torch._C._jit_override_can_fuse_on_cpu(False)
                    torch._C._jit_set_texpr_fuser_enabled(False)  # this has an effect
                    torch._C._jit_set_nvfuser_enabled(False)
                    calc.model = torch.jit.script(calc.model)
                atoms_args = {
                    "symbols": species.squeeze(0).tolist(),
                    "positions": coordinates.to(torch.float).squeeze(0).tolist(),
                    "calculator": calc,
                }

                if not args.no_pbc:
                    assert cell is not None
                    atoms_args.update({"cell": cell.cpu().numpy(), "pbc": True})

                molecule = ase.Atoms(**atoms_args)

                # run and time Langevin dynamics
                dyn = Langevin(molecule, 1 * units.fs, 300 * units.kB, 0.2)
                start = time.perf_counter()
                dyn.run(args.steps)
                end = time.perf_counter()

                times.append((end - start) * 1e6 / args.steps / (3600 * 24))
                timers_list.append(copy.deepcopy(timers))
                raw_times.append(end - start)
            all_trials.append(times)
            raw_trials.append(raw_times)

        with open(pickle_file, "wb") as fb:
            pickle.dump(
                {
                    "times": all_trials,
                    "atoms": sizes,
                    "timers": timers_list,
                    "raw_times": raw_trials,
                },
                fb,
            )

        with open(csv_file, "w") as fc:
            fc.write(f"#{comment}")
            titles = "#" + " ".join(
                [f"Trial {j} walltime per ns (days)" for j in range(args.trials)]
            )
            titles += "\n"
            fc.write(titles)
            all_trials_arr = np.array(all_trials)
            for times, s in zip(all_trials_arr, sizes):
                _times = np.asarray(times)
                string = " ".join(_times.astype(str)) + f" {s}\n"
                fc.write(string)
