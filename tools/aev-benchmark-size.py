import time
import torch
import torchani
import pynvml
import gc
import os
import numpy as np
from ase.io import read
import argparse
import textwrap


summary = '\n'
runcounter = 0
N = 200
last_py_speed = None


def getGpuName(device=None):
    i = device if device else torch.cuda.current_device()
    real_i = int(os.environ['CUDA_VISIBLE_DEVICES'][0]) if 'CUDA_VISIBLE_DEVICES' in os.environ else i
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(real_i)
    name = pynvml.nvmlDeviceGetName(h)
    return name.decode("utf-8")


def synchronize(flag=False):
    if flag:
        torch.cuda.synchronize()


def checkgpu(device=None):
    i = device if device else torch.cuda.current_device()
    t = torch.cuda.get_device_properties(i).total_memory
    c = torch.cuda.memory_reserved(i)
    name = torch.cuda.get_device_properties(i).name
    print('   GPU Memory Cached (pytorch) : {:7.1f}MB / {:.1f}MB ({})'.format(c / 1024 / 1024, t / 1024 / 1024, name))
    real_i = int(os.environ['CUDA_VISIBLE_DEVICES'][0]) if 'CUDA_VISIBLE_DEVICES' in os.environ else i
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(real_i)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    name = pynvml.nvmlDeviceGetName(h)
    print('   GPU Memory Used (nvidia-smi): {:7.1f}MB / {:.1f}MB ({})'.format(info.used / 1024 / 1024, info.total / 1024 / 1024, name.decode()))
    return f'{(info.used / 1024 / 1024):.1f}MB'


def alert(text):
    print('\033[91m{}\33[0m'.format(text))  # red


def info(text):
    print('\033[32m{}\33[0m'.format(text))  # green


def format_time(t):
    if t < 1:
        t = f'{t * 1000:.1f} ms'
    else:
        t = f'{t:.3f} sec'
    return t


def addSummaryLine(items=None, init=False):
    if init:
        addSummaryEmptyLine()
        items = ['RUN', 'PDB', 'Size', 'forward', 'backward', 'Others', 'Total', f'Total({N})', 'Speedup', 'GPU']
    global summary
    summary += items[0].ljust(20) + items[1].ljust(13) + items[2].ljust(13) + items[3].ljust(13) + items[4].ljust(13) + items[5].ljust(13) + \
        items[6].ljust(13) + items[7].ljust(13) + items[8].ljust(13) + items[9].ljust(13) + '\n'


def addSummaryEmptyLine():
    global summary
    summary += f"{'-'*20}".ljust(20) + f"{'-'*13}".ljust(13) + f"{'-'*13}".ljust(13) + f"{'-'*13}".ljust(13) + f"{'-'*13}".ljust(13) + f"{'-'*13}".ljust(13) + \
        f"{'-'*13}".ljust(13) + f"{'-'*13}".ljust(13) + f"{'-'*13}".ljust(13) + f"{'-'*13}".ljust(13) + '\n'


def benchmark(speciesPositions, aev_comp, runbackward=False, mol_info=None, verbose=True):
    global runcounter
    global last_py_speed

    runname = f"{'cu' if aev_comp.use_cuda_extension else 'py'} aev fd{'+bd' if runbackward else''}"
    items = [f'{(runcounter+1):02} {runname}', f"{mol_info['name']}", f"{mol_info['atoms']}", '-', '-', '-', '-', '-', '-', '-']

    forward_time = 0
    force_time = 0
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    start = time.time()

    aev = None
    force = None
    gpumem = None
    for i in range(N):
        species, coordinates = speciesPositions
        coordinates = coordinates.requires_grad_(runbackward)

        synchronize(not args.nsight)
        forward_start = time.time()
        try:
            _, aev = aev_comp((species, coordinates))
            if args.run_energy:
                torch.cuda.nvtx.range_push('Network')
                if args.single_nn:
                    species_energies = single_model((species, aev))
                else:
                    species_energies = neural_networks((species, aev))
                torch.cuda.nvtx.range_pop()
                # _, energies = energy_shifter(species_energies)
                energies = species_energies[1]
        except Exception as e:
            alert(f"  AEV faild: {str(e)[:50]}...")
            addSummaryLine(items)
            runcounter += 1
            return None, None, None
        synchronize(not args.nsight)
        forward_time += time.time() - forward_start

        if runbackward:  # backward
            force_start = time.time()
            try:
                if args.run_energy:
                    force = -torch.autograd.grad(energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
                else:
                    force = -torch.autograd.grad(aev.sum(), coordinates, create_graph=True, retain_graph=True)[0]
            except Exception as e:
                alert(f" Force faild: {str(e)[:50]}...")
                addSummaryLine(items)
                runcounter += 1
                return None, None, None
            synchronize(not args.nsight)
            force_time += time.time() - force_start

        if i == 2 and verbose:
            gpumem = checkgpu()

    torch.cuda.synchronize()
    total_time = (time.time() - start) / N
    force_time = force_time / N
    forward_time = forward_time / N
    others_time = total_time - force_time - forward_time

    if verbose:
        if aev_comp.use_cuda_extension:
            if last_py_speed is not None:
                speed_up = last_py_speed / total_time
                speed_up = f'{speed_up:.2f}'
            else:
                speed_up = '-'
            last_py_speed = None
        else:
            last_py_speed = total_time
            speed_up = '-'

    if verbose:
        print(f'  Duration: {total_time * N:.2f} s')
        print(f'  Speed: {total_time*1000:.2f} ms/it')
        if runcounter == 0:
            addSummaryLine(init=True)
            addSummaryEmptyLine()
        if runcounter >= 0:
            items = [f'{(runcounter+1):02} {runname}',
                     f"{mol_info['name']}",
                     f"{mol_info['atoms']}",
                     f'{format_time(forward_time)}',
                     f'{format_time(force_time)}',
                     f'{format_time(others_time)}',
                     f'{format_time(total_time)}',
                     f'{format_time(total_time * N)}',
                     f'{speed_up}',
                     f'{gpumem}']
            addSummaryLine(items)
        runcounter += 1

    return aev, total_time, force


def check_speedup_error(aev, aev_ref, force_cuaev, force_ref, speed, speed_ref):
    if (speed_ref is not None) and (speed is not None) and (aev is not None) and (aev_ref is not None):
        # aev error
        aev_error = torch.max(torch.abs(aev - aev_ref))
        print(f'  aev_error: {aev_error:.2e}')
        assert aev_error < 1e-4, f'  Error: {aev_error:.1e}\n'
        # force error
        if (force_cuaev is not None) and (force_cuaev is not None):
            force_error = torch.max(torch.abs(force_cuaev - force_ref))
            print(f'  force_error: {force_error:.2e}')
            assert force_error < 4e-4, f'  Error: {aev_error:.1e}\n'
        # speedup
        speedUP = speed_ref / speed
        if speedUP > 1:
            info(f'  Speed up: {speedUP:.2f} X\n')
        else:
            alert(f'  Speed up (slower): {speedUP:.2f} X\n')


def run(file, nnp_ref, nnp_cuaev, runbackward, maxatoms=10000):
    filepath = os.path.join(path, f'../dataset/pdb/{file}')
    mol = read(filepath)
    species = torch.tensor([mol.get_atomic_numbers()], device=device)
    positions = torch.tensor([mol.get_positions()], dtype=torch.float32, requires_grad=False, device=device)
    spelist = list(torch.unique(species.flatten()).cpu().numpy())
    realmolsize = species.shape[-1]
    species = species[:, :maxatoms]
    positions = positions[:, :maxatoms, :]
    molsize = species.shape[-1]
    speciesPositions = nnp_ref.species_converter((species, positions))
    print(f'File: {file}, Molecule size: {molsize} / {realmolsize}, Species: {spelist}\n')

    if args.nsight:
        torch.cuda.nvtx.range_push(f'{molsize}-{file}')

    mol_info = {'name': file, 'atoms': species.shape[-1]}
    if not args.nsight:
        print('Original TorchANI:')
        aev_ref, delta_ref, force_ref = benchmark(speciesPositions, nnp_ref.aev_computer, runbackward, mol_info)
        print()
    else:
        delta_ref = None

    print('CUaev:')
    # warm up
    _, _, _ = benchmark(speciesPositions, nnp_cuaev.aev_computer, runbackward, mol_info, verbose=False)
    # run
    aev, delta, force_cuaev = benchmark(speciesPositions, nnp_cuaev.aev_computer, runbackward, mol_info)

    if args.nsight:
        torch.cuda.nvtx.range_pop()

    if not args.nsight:
        check_speedup_error(aev, aev_ref, force_cuaev, force_ref, delta, delta_ref)
    print('-' * 70 + '\n')

    delta = np.nan if delta is None else delta
    delta_ref = np.nan if delta_ref is None else delta_ref
    return delta, delta_ref


def plot(maxatoms, aev_fd, cuaev_fd, aev_fdbd, cuaev_fdbd):
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from matplotlib import rc
    import datetime
    import subprocess
    from distutils.spawn import find_executable

    rc('mathtext', fontset='cm')
    usetex = bool(find_executable('latex')) and bool(find_executable('dvipng'))
    # usetex = False
    rc('axes.spines', **{'right': False, 'top': False})
    rc('axes', grid=True)
    plt.rcParams['grid.linewidth'] = 0.15
    if usetex:
        rc('xtick', labelsize=14)
        rc('ytick', labelsize=14)
        rc('text', usetex=True)
        rc('font', **{'size': 16, 'family': 'serif', 'sans-serif': ['Lucid', 'Arial', 'Helvetica'], 'serif': ['Times', 'Times New Roman']})
    else:
        rc('font', **{'size': 14})

    aev_fd = np.array(aev_fd) * 1000
    cuaev_fd = np.array(cuaev_fd) * 1000
    aev_fdbd = np.array(aev_fdbd) * 1000
    cuaev_fdbd = np.array(cuaev_fdbd) * 1000
    plt.figure(figsize=(11, 6), dpi=200)
    # labels
    num_nn = 1 if args.single_nn else num_models
    nn_label = ""
    if args.run_energy:
        nn_label = f"+ nn ({num_nn}) "
    energy_label = ""
    if args.run_energy:
        energy_label = f"Energy: NN ({num_nn}), Infer Model ({'on' if args.infer_model else 'off'})"
        if args.infer_model:
            energy_label += f", MNP ({'on' if args.mnp else 'off'})"
    # plot
    plt.plot(maxatoms, aev_fd, '--bo', label=f'pyaev {nn_label}forward')
    plt.plot(maxatoms, cuaev_fd, '--ro', label=f'cuaev {nn_label}forward')
    plt.plot(maxatoms, aev_fdbd, '-bo', label=f'pyaev {nn_label}forward + backward')
    plt.plot(maxatoms, cuaev_fdbd, '-ro', label=f'cuaev {nn_label}forward + backward')
    for i, txt in enumerate(aev_fd):
        plt.annotate(f'{txt:.2f}', (maxatoms[i], aev_fd[i] - 2.5), ha='center', va='center', fontsize=12, color='b')
    for i, txt in enumerate(cuaev_fd):
        plt.annotate(f'{txt:.2f}', (maxatoms[i], cuaev_fd[i] - 2.5), ha='center', va='center', fontsize=12, color='r')
    for i, txt in enumerate(aev_fdbd):
        plt.annotate(f'{txt:.2f}', (maxatoms[i], aev_fdbd[i] + 2.5), ha='center', va='center', fontsize=12, color='b')
    for i, txt in enumerate(cuaev_fdbd):
        plt.annotate(f'{txt:.2f}', (maxatoms[i], cuaev_fdbd[i] + 2.5), ha='center', va='center', fontsize=12, color='r')
    # plt.legend(frameon=False, fontsize=15, loc='upper left')
    plt.legend(frameon=True, fontsize=15, loc='best')
    plt.xlim(maxatoms[0] - 500, maxatoms[-1] + 500)
    plt.ylim(-3, 85)
    plt.xlabel(r'System Size (atoms)')
    plt.ylabel(r'Time (ms)')
    plt.title(f'Benchmark of cuaev (cuda extension) and pyaev (torch operators) \non {getGpuName()}\n{energy_label}', fontsize=16)
    plt.show()

    dir = 'benchmark'
    if not os.path.exists(dir):
        os.makedirs(dir)
    timenow = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"
    imgfilename = f"{dir}/{timenow}.png"
    plt.savefig(imgfilename, dpi=200, bbox_inches='tight')
    cmd = f"if command -v imgcat > /dev/null; then imgcat {imgfilename} --height 30; fi"
    subprocess.run(cmd, shell=True, check=True, universal_newlines=True)
    info(f"\nBenchmark plot saved at {os.path.realpath(imgfilename)}")


def run_for_plot(file, maxatoms, nnp_ref, nnp_cuaev):
    aev_fd = []
    cuaev_fd = []
    aev_fdbd = []
    cuaev_fdbd = []
    for maxatom in maxatoms:
        cuaev_t, aev_t = run(file, nnp_ref, nnp_cuaev, runbackward=False, maxatoms=maxatom)
        cuaev_fd.append(cuaev_t)
        aev_fd.append(aev_t)
    addSummaryEmptyLine()
    info('Add Backward\n')
    for maxatom in maxatoms:
        cuaev_t, aev_t = run(file, nnp_ref, nnp_cuaev, runbackward=True, maxatoms=maxatom)
        cuaev_fdbd.append(cuaev_t)
        aev_fdbd.append(aev_t)
    addSummaryEmptyLine()
    plot(maxatoms, aev_fd, cuaev_fd, aev_fdbd, cuaev_fdbd)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=textwrap.dedent('''\
        Benchmark Tool
        Example usage:
        python tools/aev-benchmark-size.py -p
        python tools/aev-benchmark-size.py -p -e
        python tools/aev-benchmark-size.py -p -e --infer_model
        python tools/aev-benchmark-size.py -p -e --infer_model --mnp
        python tools/aev-benchmark-size.py -p -e --infer_model --mnp --single_nn
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-s', '--nsight',
                        action='store_true',
                        help='use nsight profile')
    parser.add_argument('-p', '--plot',
                        action='store_true',
                        help='plot benchmark result')
    parser.add_argument('-e', '--run_energy',
                        action='store_true',
                        help='Run NN to predict energy')
    parser.add_argument('--single_nn',
                        action='store_true',
                        help='Energy: Only Run single NN to predict energy, instead of an ensemble')
    parser.add_argument('--infer_model',
                        action='store_true',
                        help='Energy: Use infer model')
    parser.add_argument('--mnp',
                        action='store_true',
                        help='Energy: Use multi net parallel (mnp) extension')
    parser.add_argument('-n', '--N',
                        help='Number of Repeat',
                        default=200, type=int)
    parser.add_argument('--no-cell-list',
                        help='No use of cell list for pyaev',
                        action='store_true', default=False)
    parser.set_defaults(backward=0)
    parser.set_defaults(run_energy=0)
    parser.set_defaults(single_nn=0)
    parser.set_defaults(infer_model=0)
    parser.set_defaults(mnp=0)
    parser.set_defaults(plot=0)
    args = parser.parse_args()
    path = os.path.dirname(os.path.realpath(__file__))
    N = args.N

    device = torch.device('cuda')
    files = ['small.pdb', '1hz5.pdb', '6W8H.pdb']
    if args.no_cell_list:
        use_cell_list = False
    else:
        use_cell_list = True
    nnp_ref = torchani.models.ANI2x(periodic_table_index=True, model_index=None, cell_list=use_cell_list).to(device)
    nnp_cuaev = torchani.models.ANI2x(periodic_table_index=True, model_index=None).to(device)
    nnp_cuaev.aev_computer.use_cuda_extension = True
    maxatoms = [6000, 10000]

    if args.nsight:
        N = 10
        torch.cuda.profiler.start()
        maxatoms = [10000]

    neural_networks = nnp_ref.neural_networks
    num_models = len(neural_networks)
    single_model = neural_networks[0]
    if args.infer_model:
        neural_networks = neural_networks.to_infer_model(use_mnp=args.mnp).to(device)
        single_model = single_model.to_infer_model(use_mnp=args.mnp).to(device)
    energy_shifter = nnp_ref.energy_shifter

    # if run for plots
    if args.plot:
        maxatoms = np.concatenate([[100, 3000, 5000, 6000, 8000], np.arange(10000, 31000, 5000)])
        file = '6ZDH.pdb'
        run_for_plot(file, maxatoms, nnp_ref, nnp_cuaev)
    else:
        # if not for nsight
        if not args.nsight:
            for file in files:
                run(file, nnp_ref, nnp_cuaev, runbackward=False)
            for maxatom in maxatoms:
                file = '1C17.pdb'
                run(file, nnp_ref, nnp_cuaev, runbackward=False, maxatoms=maxatom)
            addSummaryEmptyLine()
            info('Add Backward\n')
            for file in files:
                run(file, nnp_ref, nnp_cuaev, runbackward=True)
        for maxatom in maxatoms:
            file = '1C17.pdb'
            run(file, nnp_ref, nnp_cuaev, runbackward=True, maxatoms=maxatom)
        addSummaryEmptyLine()

    print(summary)
    if args.nsight:
        torch.cuda.profiler.stop()
