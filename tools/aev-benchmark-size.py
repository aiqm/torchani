import time
import torch
import torchani
import pynvml
import gc
import os
from ase.io import read
import argparse


def checkgpu(device=None):
    i = device if device else torch.cuda.current_device()
    real_i = int(os.environ['CUDA_VISIBLE_DEVICES'][0]) if 'CUDA_VISIBLE_DEVICES' in os.environ else i
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(real_i)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    name = pynvml.nvmlDeviceGetName(h)
    print('  GPU Memory Used (nvidia-smi): {:7.1f}MB / {:.1f}MB ({})'.format(info.used / 1024 / 1024, info.total / 1024 / 1024, name.decode()))


def alert(text):
    print('\033[91m{}\33[0m'.format(text))  # red


def info(text):
    print('\033[32m{}\33[0m'.format(text))  # green


def benchmark(speciesPositions, aev_comp, N, check_gpu_mem):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    start = time.time()

    aev = None
    for i in range(N):
        aev = aev_comp(speciesPositions).aevs
        if i == 2 and check_gpu_mem:
            checkgpu()

    torch.cuda.synchronize()
    delta = time.time() - start
    print(f'  Duration: {delta:.2f} s')
    print(f'  Speed: {delta/N*1000:.2f} ms/it')
    return aev, delta


def check_speedup_error(aev, aev_ref, speed, speed_ref):
    speedUP = speed_ref / speed
    if speedUP > 1:
        info(f'  Speed up: {speedUP:.2f} X\n')
    else:
        alert(f'  Speed up (slower): {speedUP:.2f} X\n')

    aev_error = torch.max(torch.abs(aev - aev_ref))
    assert aev_error < 0.02, f'  Error: {aev_error:.1e}\n'


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--check_gpu_mem',
                        dest='check_gpu_mem',
                        action='store_const',
                        const=1)
    parser.add_argument('--nsight',
                        action='store_true',
                        help='use nsight profile')
    parser.set_defaults(check_gpu_mem=0)
    args = parser.parse_args()
    path = os.path.dirname(os.path.realpath(__file__))

    check_gpu_mem = args.check_gpu_mem
    device = torch.device('cuda')
    files = ['small.pdb', '1hz5.pdb', '6W8H.pdb']

    N = 500
    if args.nsight:
        N = 3
        torch.cuda.profiler.start()

    for file in files:
        datafile = os.path.join(path, f'../dataset/pdb/{file}')
        mol = read(datafile)
        species = torch.tensor([mol.get_atomic_numbers()], device=device)
        positions = torch.tensor([mol.get_positions()], dtype=torch.float32, requires_grad=False, device=device)
        print(f'File: {file}, Molecule size: {species.shape[-1]}\n')

        nnp = torchani.models.ANI2x(periodic_table_index=True, model_index=None).to(device)
        speciesPositions = nnp.species_converter((species, positions))
        aev_computer = nnp.aev_computer

        if args.nsight:
            torch.cuda.nvtx.range_push(file)
        print('Original TorchANI:')
        aev_ref, delta_ref = benchmark(speciesPositions, aev_computer, N, check_gpu_mem)
        print()

        print('CUaev:')
        nnp.aev_computer.use_cuda_extension = True
        cuaev_computer = nnp.aev_computer
        aev, delta = benchmark(speciesPositions, cuaev_computer, N, check_gpu_mem)
        if args.nsight:
            torch.cuda.nvtx.range_pop()

        check_speedup_error(aev, aev_ref, delta, delta_ref)
        print('-' * 70 + '\n')

    if args.nsight:
        torch.cuda.profiler.stop()
