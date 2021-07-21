import torch
import torchani
import time
import argparse
import pkbar
import gc
import pynvml
import os
import pickle
from torchani.units import hartree2kcalmol
from typing import Dict
from tool_utils import time_functions_in_model

summary = ''
runcounter = 0


def build_network():
    H_network = torch.nn.Sequential(
        torch.nn.Linear(384, 160),
        torch.nn.CELU(0.1),
        torch.nn.Linear(160, 128),
        torch.nn.CELU(0.1),
        torch.nn.Linear(128, 96),
        torch.nn.CELU(0.1),
        torch.nn.Linear(96, 1)
    )

    C_network = torch.nn.Sequential(
        torch.nn.Linear(384, 144),
        torch.nn.CELU(0.1),
        torch.nn.Linear(144, 112),
        torch.nn.CELU(0.1),
        torch.nn.Linear(112, 96),
        torch.nn.CELU(0.1),
        torch.nn.Linear(96, 1)
    )

    N_network = torch.nn.Sequential(
        torch.nn.Linear(384, 128),
        torch.nn.CELU(0.1),
        torch.nn.Linear(128, 112),
        torch.nn.CELU(0.1),
        torch.nn.Linear(112, 96),
        torch.nn.CELU(0.1),
        torch.nn.Linear(96, 1)
    )

    O_network = torch.nn.Sequential(
        torch.nn.Linear(384, 128),
        torch.nn.CELU(0.1),
        torch.nn.Linear(128, 112),
        torch.nn.CELU(0.1),
        torch.nn.Linear(112, 96),
        torch.nn.CELU(0.1),
        torch.nn.Linear(96, 1)
    )
    nets = [H_network, C_network, N_network, O_network]

    for net in nets:
        net.apply(init_normal)

    return nets


def init_normal(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)


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


def sync_cuda(sync):
    if sync:
        torch.cuda.synchronize()


def print_timer(label, t):
    if t < 1:
        t = f'{t * 1000:.1f} ms'
    else:
        t = f'{t:.3f} sec'
    print(f'{label} - {t}')


def format_time(t):
    if t < 1:
        t = f'{t * 1000:.1f} ms'
    else:
        t = f'{t:.3f} sec'
    return t


def benchmark(args, dataset, use_cuda_extension, force_train=False):
    global summary
    global runcounter

    if args.nsight and runcounter >= 0:
        torch.cuda.nvtx.range_push(args.runname)
    synchronize = True
    timers = {}

    aev_computer = torchani.AEVComputer.like_1x(use_cuda_extension=use_cuda_extension)

    nn = torchani.ANIModel(build_network())
    model = torch.nn.Sequential(aev_computer, nn).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    mse = torch.nn.MSELoss(reduction='none')

    # enable timers
    timers: Dict[str, int] = dict()
    fn_to_time_aev = ['_compute_radial_aev', '_compute_angular_aev',
                             '_compute_aev', '_triple_by_molecule', 'forward']
    fn_to_time_neighborlist = ['forward']
    fn_to_time_nn = ['forward']
    fn_to_time_opt = ['step']
    fn_to_time_model = ['forward']

    time_functions_in_model(model, fn_to_time_model, timers, synchronize)
    time_functions_in_model(aev_computer, fn_to_time_aev, timers, synchronize)
    time_functions_in_model(aev_computer.neighborlist, fn_to_time_neighborlist, timers, synchronize)
    time_functions_in_model(nn, fn_to_time_nn, timers, synchronize)
    time_functions_in_model(optimizer, fn_to_time_opt, timers, synchronize)

    print('=> start training')
    start = time.time()
    loss_time = 0
    force_time = 0

    for epoch in range(0, args.num_epochs):

        print('Epoch: %d/%d' % (epoch + 1, args.num_epochs))
        progbar = pkbar.Kbar(target=len(dataset) - 1, width=8)

        for i, properties in enumerate(dataset):
            species = properties['species'].to(args.device)
            coordinates = properties['coordinates'].to(args.device).float().requires_grad_(force_train)
            true_energies = properties['energies'].to(args.device).float()
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            _, predicted_energies = model((species, coordinates))
            # TODO add sync after aev is done
            sync_cuda(synchronize)
            energy_loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
            if force_train:
                sync_cuda(synchronize)
                force_coefficient = 0.1
                true_forces = properties['forces'].to(args.device).float()
                force_start = time.time()
                try:
                    sync_cuda(synchronize)
                    forces = -torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
                    sync_cuda(synchronize)
                except Exception as e:
                    alert('Error: {}'.format(e))
                    return
                force_time += time.time() - force_start
                force_loss = (mse(true_forces, forces).sum(dim=(1, 2)) / num_atoms).mean()
                loss = energy_loss + force_coefficient * force_loss
                sync_cuda(synchronize)
            else:
                loss = energy_loss
            rmse = hartree2kcalmol((mse(predicted_energies, true_energies)).mean()).detach().cpu().numpy()
            progbar.update(i, values=[("rmse", rmse)])
            sync_cuda(synchronize)
            loss_start = time.time()
            loss.backward()
            sync_cuda(synchronize)
            loss_stop = time.time()
            loss_time += loss_stop - loss_start
            optimizer.step()
            sync_cuda(synchronize)

        gpumem = checkgpu()
    sync_cuda(synchronize)
    stop = time.time()

    if args.nsight and runcounter >= 0:
        torch.cuda.nvtx.range_pop()
    print('=> More detail about benchmark PER EPOCH')
    total_time = (stop - start) / args.num_epochs
    loss_time = loss_time / args.num_epochs
    force_time = force_time / args.num_epochs
    opti_time = timers['Adam.step'] / args.num_epochs
    nn_time = timers['ANIModel.forward'] / args.num_epochs
    aev_time = timers['AEVComputer.forward'] / args.num_epochs
    model_time = timers['Sequential.forward'] / args.num_epochs
    print_timer('   Full Model forward', model_time)
    print_timer('   AEV forward', aev_time)
    print_timer('   NN forward', nn_time)
    print_timer('   Backward', loss_time)
    print_timer('   Force', force_time)
    print_timer('   Optimizer', opti_time)
    others_time = total_time - loss_time - aev_time - nn_time - opti_time - force_time
    print_timer('   Others', others_time)
    print_timer('   Epoch time', total_time)

    if runcounter == 0:
        summary += '\n' + 'RUN'.ljust(27) + 'Total AEV'.ljust(13) + 'NN Forward'.ljust(13) + 'Backward'.ljust(13) + 'Force'.ljust(13) + \
            'Optimizer'.ljust(13) + 'Others'.ljust(13) + 'Epoch time'.ljust(13) + 'GPU'.ljust(13) + '\n'
    if runcounter >= 0:
        summary += f'{runcounter} {args.runname}'.ljust(27) + f'{format_time(aev_time)}'.ljust(13) + f'{format_time(nn_time)}'.ljust(13) + f'{format_time(loss_time)}'.ljust(13) + f'{format_time(force_time)}'.ljust(13) + \
            f'{format_time(opti_time)}'.ljust(13) + f'{format_time(others_time)}'.ljust(13) + f'{format_time(total_time)}'.ljust(13) + f'{gpumem}'.ljust(13) + '\n'
    runcounter += 1


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path',
                        help='Path of the dataset, can a hdf5 file \
                            or a directory containing hdf5 files')
    parser.add_argument('-d', '--device',
                        help='Device of modules and tensors',
                        default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('-b', '--batch_size',
                        help='Number of conformations of each batch',
                        default=2560, type=int)
    parser.add_argument('-p', '--pickle',
                        action='store_true',
                        help='Dataset is pickled or not')
    parser.add_argument('--nsight',
                        action='store_true',
                        help='use nsight profile')
    parser.add_argument('-n', '--num_epochs',
                        help='epochs',
                        default=1, type=int)
    args = parser.parse_args()

    print('=> loading dataset...')
    if args.pickle:
        f = open(args.dataset_path, 'rb')
        dataset_shuffled = pickle.load(f)
        f.close()
    else:
        shifter = torchani.EnergyShifter(None)
        dataset = torchani.data.load(args.dataset_path, additional_properties=('forces',)).subtract_self_energies(shifter).species_to_indices()
        print('=> Caching shuffled dataset...')
        dataset_shuffled = list(dataset.shuffle().collate(args.batch_size))
        f = open(f'{args.dataset_path}.pickle', 'wb')
        pickle.dump(dataset_shuffled, f)
        f.close()

    print("=> CUDA info:")
    devices = torch.cuda.device_count()
    print('Total devices: {}'.format(devices))
    for i in range(devices):
        d = 'cuda:{}'.format(i)
        print('{}: {}'.format(i, torch.cuda.get_device_name(d)))
        print('   {}'.format(torch.cuda.get_device_properties(i)))
        checkgpu(i)

    # Warming UP
    if len(dataset_shuffled) < 100:
        runcounter = -1
        args.runname = 'Warning UP'
        print(f"\n\n=> Test 0: {args.runname}")
        torch.cuda.empty_cache()
        gc.collect()
        benchmark(args, dataset_shuffled, use_cuda_extension=True, force_train=False)

    if args.nsight:
        torch.cuda.profiler.start()

    args.runname = 'cu Energy train'
    print(f"\n\n=> Test 1: {args.runname}")
    torch.cuda.empty_cache()
    gc.collect()
    benchmark(args, dataset_shuffled, use_cuda_extension=True, force_train=False)

    args.runname = 'py Energy train'
    print(f"\n\n=> Test 2: {args.runname}")
    torch.cuda.empty_cache()
    gc.collect()
    benchmark(args, dataset_shuffled, use_cuda_extension=False, force_train=False)
    try:
        args.runname = 'cu Energy + Force train'
        print(f"\n\n=> Test 3: {args.runname}")
        torch.cuda.empty_cache()
        gc.collect()
        benchmark(args, dataset_shuffled, use_cuda_extension=True, force_train=True)

        args.runname = 'py Energy + Force train'
        print(f"\n\n=> Test 4: {args.runname}")
        torch.cuda.empty_cache()
        gc.collect()
        benchmark(args, dataset_shuffled, use_cuda_extension=False, force_train=True)
    except AttributeError:
        print('Skipping force training benchmark since a dataset without forces was provided')
        print('Please provide a dataset with forces for this benchmark')

    print(summary)

    if args.nsight:
        torch.cuda.profiler.stop()
