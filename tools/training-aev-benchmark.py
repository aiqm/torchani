import time
import argparse
import gc
import os

import torch
import pynvml
from tqdm import tqdm

from torchani.models import ANI1x
from torchani.units import hartree2kcalpermol
from torchani.datasets import batch_all_in_ram
from torchani import datasets

summary = ""
runcounter = 0


def checkgpu(device=None):
    i = device if device else torch.cuda.current_device()
    t = torch.cuda.get_device_properties(i).total_memory
    c = torch.cuda.memory_reserved(i)
    name = torch.cuda.get_device_properties(i).name
    print(
        "   GPU Memory Cached (pytorch) : {:7.1f}MB / {:.1f}MB ({})".format(
            c / 1024 / 1024, t / 1024 / 1024, name
        )
    )
    real_i = (
        int(os.environ["CUDA_VISIBLE_DEVICES"][0])
        if "CUDA_VISIBLE_DEVICES" in os.environ
        else i
    )
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(real_i)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    _name = pynvml.nvmlDeviceGetName(h)
    print(
        "   GPU Memory Used (nvidia-smi): {:7.1f}MB / {:.1f}MB ({})".format(
            info.used / 1024 / 1024, info.total / 1024 / 1024, _name
        )
    )
    return f"{(info.used / 1024 / 1024):.1f}MB"


def alert(text):
    print("\033[91m{}\33[0m".format(text))  # red


def sync_cuda(sync):
    if sync:
        torch.cuda.synchronize()


def print_timer(label, t):
    if abs(t) < 1:
        t = f"{t * 1000:.1f} ms"
    else:
        t = f"{t:.3f} sec"
    print(f"{label} - {t}")


def format_time(t):
    if t < 1:
        t = f"{t * 1000:.1f} ms"
    else:
        t = f"{t:.3f} sec"
    return t


def benchmark(args, dataset, strat: str = "pyaev", force_train=False):
    global summary
    global runcounter

    if args.nsight and runcounter >= 0:
        torch.cuda.nvtx.range_push(args.runname)
    synchronize = True
    _model = ANI1x(model_index=0, strategy=strat).to(args.device)
    aev_computer = _model.aev_computer
    nn = _model.neural_networks
    model = torch.nn.Sequential(aev_computer, nn).to(args.device)
    mse = torch.nn.MSELoss(reduction="none")

    # unfreeze the parameters
    for param in model.parameters():
        param.requires_grad_(True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # enable timers

    print("=> start training")
    torch.cuda.synchronize()
    start = time.time()
    aev_time = 0.0
    nn_time = 0.0
    loss_time = 0.0
    force_time = 0.0

    for epoch in range(0, args.num_epochs):
        print("Epoch: %d/%d" % (epoch + 1, args.num_epochs))
        pbar = tqdm(desc="rmse: ?", total=len(dataset))

        for i, properties in enumerate(dataset):
            species = properties["species"].to(args.device)
            species = _model.species_converter(species)
            coordinates = (
                properties["coordinates"]
                .to(args.device)
                .float()
                .requires_grad_(force_train)
            )
            true_energies = properties["energies"].to(args.device).float()
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            # Run AEV
            sync_cuda(synchronize)
            aev_start = time.time()
            aev = aev_computer(species, coordinates)
            sync_cuda(synchronize)
            aev_time += time.time() - aev_start
            # Run NN
            nn_start = time.time()
            _, predicted_energies = nn((species, aev))
            sync_cuda(synchronize)
            nn_time += time.time() - nn_start
            energy_loss = (
                mse(predicted_energies, true_energies) / num_atoms.sqrt()
            ).mean()
            if force_train:
                sync_cuda(synchronize)
                force_coefficient = 0.1
                true_forces = properties["forces"].to(args.device).float()
                force_start = time.time()
                try:
                    sync_cuda(synchronize)
                    forces = -torch.autograd.grad(
                        predicted_energies.sum(),
                        coordinates,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    sync_cuda(synchronize)
                except Exception as e:
                    alert("Error: {}".format(e))
                    return
                force_time += time.time() - force_start
                force_loss = (
                    mse(true_forces, forces).sum(dim=(1, 2)) / num_atoms
                ).mean()
                loss = energy_loss + force_coefficient * force_loss
            else:
                loss = energy_loss
            rmse = (
                hartree2kcalpermol((mse(predicted_energies, true_energies)).mean())
                .detach()
                .cpu()
                .numpy()
            )
            pbar.update()
            pbar.set_description(f"rmse: {rmse}")
            optimizer.zero_grad()

            # Run backward
            sync_cuda(synchronize)
            loss_start = time.time()
            loss.backward()
            sync_cuda(synchronize)
            loss_time += time.time() - loss_start

            optimizer.step()

        gpumem = checkgpu()
    sync_cuda(synchronize)
    stop = time.time()

    if args.nsight and runcounter >= 0:
        torch.cuda.nvtx.range_pop()
    print("=> More detail about benchmark PER EPOCH")
    total_time = (stop - start) / args.num_epochs
    loss_time = loss_time / args.num_epochs
    force_time = force_time / args.num_epochs
    nn_time = nn_time / args.num_epochs
    aev_time = aev_time / args.num_epochs
    print_timer("   AEV forward", aev_time)
    print_timer("   NN forward", nn_time)
    print_timer("   Backward", loss_time)
    print_timer("   Force", force_time)
    others_time = total_time - loss_time - aev_time - nn_time - force_time
    print_timer("   Others", others_time)
    print_timer("   Epoch time", total_time)

    if runcounter == 0:
        summary += (
            "\n"
            + "RUN".ljust(27)
            + "Total AEV".ljust(13)
            + "NN Forward".ljust(13)
            + "Backward".ljust(13)
            + "Force".ljust(13)
            + "Others".ljust(13)
            + "Epoch time".ljust(13)
            + "GPU".ljust(13)
            + "\n"
        )
    if runcounter >= 0:
        summary += (
            f"{runcounter} {args.runname}".ljust(27)
            + f"{format_time(aev_time)}".ljust(13)
            + f"{format_time(nn_time)}".ljust(13)
            + f"{format_time(loss_time)}".ljust(13)
            + f"{format_time(force_time)}".ljust(13)
            + f"{format_time(others_time)}".ljust(13)
            + f"{format_time(total_time)}".ljust(13)
            + f"{gpumem}".ljust(13)
            + "\n"
        )
    runcounter += 1


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        help="Name of builtin dataset to train on, TestDataForcesDipoles or ANI1x",
        nargs="?",
        default="ANI1x",
    )
    parser.add_argument(
        "-d",
        "--device",
        help="Device of modules and tensors",
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Number of conformations of each batch",
        default=2560,
        type=int,
    )
    parser.add_argument(
        "-p", "--pickle", action="store_true", help="Dataset is pickled or not"
    )
    parser.add_argument("--nsight", action="store_true", help="use nsight profile")
    parser.add_argument("-n", "--num_epochs", help="epochs", default=1, type=int)
    args = parser.parse_args()

    print("=> loading dataset...")
    ds = getattr(datasets, args.dataset)(verbose=False)
    dataset = batch_all_in_ram(
        ds,
        properties=("species", "coordinates", "energies", "forces"),
        batch_size=args.batch_size,
    )

    print("=> CUDA info:")
    devices = torch.cuda.device_count()
    print("Total devices: {}".format(devices))
    for i in range(devices):
        d = "cuda:{}".format(i)
        print("{}: {}".format(i, torch.cuda.get_device_name(d)))
        print("   {}".format(torch.cuda.get_device_properties(i)))
        checkgpu(i)

    # Always warmup
    if True:
        runcounter = -1
        args.runname = "Warning UP"
        print(f"\n\n=> Test 0: {args.runname}")
        torch.cuda.empty_cache()
        gc.collect()
        benchmark(args, dataset, strat="pyaev", force_train=False)

    if args.nsight:
        torch.cuda.profiler.start()

    args.runname = "cu Energy train"
    print(f"\n\n=> Test 1: {args.runname}")
    torch.cuda.empty_cache()
    gc.collect()
    benchmark(args, dataset, strat="cuaev-fused", force_train=False)

    args.runname = "py Energy train"
    print(f"\n\n=> Test 2: {args.runname}")
    torch.cuda.empty_cache()
    gc.collect()
    benchmark(args, dataset, strat="pyaev", force_train=False)
    try:
        args.runname = "cu Energy + Force train"
        print(f"\n\n=> Test 3: {args.runname}")
        torch.cuda.empty_cache()
        gc.collect()
        benchmark(args, dataset, strat="cuaev-fused", force_train=True)

        args.runname = "py Energy + Force train"
        print(f"\n\n=> Test 4: {args.runname}")
        torch.cuda.empty_cache()
        gc.collect()
        benchmark(args, dataset, strat="pyaev", force_train=True)
    except AttributeError:
        print(
            "Skipping force training benchmark. A dataset without forces was provided"
        )
        print("Please provide a dataset with forces for this benchmark")

    print(summary)

    if args.nsight:
        torch.cuda.profiler.stop()
