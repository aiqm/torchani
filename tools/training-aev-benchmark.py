import torch
import torchani
import time
import timeit
import argparse
import pkbar
import gc
import pynvml
import os
from torchani.units import hartree2kcalmol


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
    return [H_network, C_network, N_network, O_network]


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


def alert(text):
    print('\033[91m{}\33[0m'.format(text))  # red


def sync_cuda(sync):
    if sync:
        torch.cuda.synchronize()


def benchmark(parser, dataset, use_cuda_extension, force_training=False):
    synchronize = True if parser.synchronize else False
    timers = {}

    def time_func(key, func):
        timers[key] = 0

        def wrapper(*args, **kwargs):
            start = timeit.default_timer()
            ret = func(*args, **kwargs)
            if synchronize:
                torch.cuda.synchronize()
            end = timeit.default_timer()
            timers[key] += end - start
            return ret

        return wrapper

    Rcr = 5.2000e+00
    Rca = 3.5000e+00
    EtaR = torch.tensor([1.6000000e+01], device=parser.device)
    ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=parser.device)
    Zeta = torch.tensor([3.2000000e+01], device=parser.device)
    ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=parser.device)
    EtaA = torch.tensor([8.0000000e+00], device=parser.device)
    ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=parser.device)
    num_species = 4
    aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species, use_cuda_extension)

    nn = torchani.ANIModel(build_network())
    model = torch.nn.Sequential(aev_computer, nn).to(parser.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    mse = torch.nn.MSELoss(reduction='none')

    # enable timers
    torchani.aev.cutoff_cosine = time_func('torchani.aev.cutoff_cosine', torchani.aev.cutoff_cosine)
    torchani.aev.radial_terms = time_func('torchani.aev.radial_terms', torchani.aev.radial_terms)
    torchani.aev.angular_terms = time_func('torchani.aev.angular_terms', torchani.aev.angular_terms)
    torchani.aev.compute_shifts = time_func('torchani.aev.compute_shifts', torchani.aev.compute_shifts)
    torchani.aev.neighbor_pairs = time_func('torchani.aev.neighbor_pairs', torchani.aev.neighbor_pairs)
    torchani.aev.neighbor_pairs_nopbc = time_func('torchani.aev.neighbor_pairs_nopbc', torchani.aev.neighbor_pairs_nopbc)
    torchani.aev.triu_index = time_func('torchani.aev.triu_index', torchani.aev.triu_index)
    torchani.aev.cumsum_from_zero = time_func('torchani.aev.cumsum_from_zero', torchani.aev.cumsum_from_zero)
    torchani.aev.triple_by_molecule = time_func('torchani.aev.triple_by_molecule', torchani.aev.triple_by_molecule)
    torchani.aev.compute_aev = time_func('torchani.aev.compute_aev', torchani.aev.compute_aev)
    model[0].forward = time_func('total', model[0].forward)
    model[1].forward = time_func('forward', model[1].forward)
    optimizer.step = time_func('optimizer.step', optimizer.step)

    print('=> start training')
    start = time.time()
    loss_time = 0

    for epoch in range(0, parser.num_epochs):

        print('Epoch: %d/%d' % (epoch + 1, parser.num_epochs))
        progbar = pkbar.Kbar(target=len(dataset) - 1, width=8)

        for i, properties in enumerate(dataset):
            species = properties['species'].to(parser.device)
            coordinates = properties['coordinates'].to(parser.device).float().requires_grad_(force_training)
            true_energies = properties['energies'].to(parser.device).float()
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            _, predicted_energies = model((species, coordinates))
            # TODO add sync after aev is done
            sync_cuda(synchronize)
            energy_loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
            if force_training:
                force_coefficient = 0.1
                true_forces = properties['forces'].to(parser.device).float()
                try:
                    forces = -torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
                except Exception as e:
                    alert('Error: {}'.format(e))
                    return
                force_loss = (mse(true_forces, forces).sum(dim=(1, 2)) / num_atoms).mean()
                loss = energy_loss + force_coefficient * force_loss
            else:
                loss = energy_loss
            rmse = hartree2kcalmol((mse(predicted_energies, true_energies)).mean()).detach().cpu().numpy()
            sync_cuda(synchronize)
            loss_start = time.time()
            loss.backward()
            sync_cuda(synchronize)
            loss_stop = time.time()
            loss_time += loss_stop - loss_start
            optimizer.step()
            sync_cuda(synchronize)
            progbar.update(i, values=[("rmse", rmse)])

        checkgpu()
    sync_cuda(synchronize)
    stop = time.time()

    print('=> More detail about benchmark PER EPOCH')
    for k in timers:
        if k.startswith('torchani.'):
            print('   {} - {:.1f}s'.format(k, timers[k] / parser.num_epochs))
    total_time = (stop - start) / parser.num_epochs
    loss_time = loss_time / parser.num_epochs
    opti_time = timers['optimizer.step'] / parser.num_epochs
    forward_time = timers['forward'] / parser.num_epochs
    aev_time = timers['total'] / parser.num_epochs
    print('Total AEV - {:.1f}s'.format(aev_time))
    print('Forward - {:.1f}s'.format(forward_time))
    print('Backward - {:.1f}s'.format(loss_time))
    print('Optimizer - {:.1f}s'.format(opti_time))
    print('Others - {:.1f}s'.format(total_time - loss_time - aev_time - forward_time - opti_time))
    print('Epoch time - {:.1f}s'.format(total_time))


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
    parser.add_argument('-y', '--synchronize',
                        action='store_true',
                        help='whether to insert torch.cuda.synchronize() at the end of each function')
    parser.add_argument('-n', '--num_epochs',
                        help='epochs',
                        default=1, type=int)
    parser = parser.parse_args()

    print('=> loading dataset...')
    shifter = torchani.EnergyShifter(None)
    # parser.batch_size = 1280
    dataset = torchani.data.load(parser.dataset_path, additional_properties=('forces',)).subtract_self_energies(shifter).species_to_indices()
    print('=> Caching shuffled dataset...')
    dataset_shuffled = list(dataset.shuffle().collate(parser.batch_size))
    print('=> Caching non-shuffled dataset...')
    dataset = list(dataset.collate(parser.batch_size))

    print("=> CUDA info:")
    devices = torch.cuda.device_count()
    print('Total devices: {}'.format(devices))
    for i in range(devices):
        d = 'cuda:{}'.format(i)
        print('{}: {}'.format(i, torch.cuda.get_device_name(d)))
        print('   {}'.format(torch.cuda.get_device_properties(i)))
        checkgpu(i)

    print("\n\n=> Test 1/8: Shuffled Dataset, USE cuda extension, Energy training")
    torch.cuda.empty_cache()
    gc.collect()
    benchmark(parser, dataset_shuffled, use_cuda_extension=True, force_training=False)
    print("\n\n=> Test 2/8: Shuffled Dataset, NO cuda extension, Energy training")
    torch.cuda.empty_cache()
    gc.collect()
    benchmark(parser, dataset_shuffled, use_cuda_extension=False, force_training=False)
    print("\n\n=> Test 3/8: Non-Shuffled Dataset, USE cuda extension, Energy training")
    torch.cuda.empty_cache()
    gc.collect()
    benchmark(parser, dataset, use_cuda_extension=True, force_training=False)
    print("\n\n=> Test 4/8: Non-Shuffled Dataset, NO cuda extension, Energy training")
    torch.cuda.empty_cache()
    gc.collect()
    benchmark(parser, dataset, use_cuda_extension=False, force_training=False)

    print("\n\n=> Test 5/8: Shuffled Dataset, USE cuda extension, Force and Energy training")
    torch.cuda.empty_cache()
    gc.collect()
    benchmark(parser, dataset_shuffled, use_cuda_extension=True, force_training=True)
    print("\n\n=> Test 6/8: Shuffled Dataset, NO cuda extension, Force and Energy training")
    torch.cuda.empty_cache()
    gc.collect()
    benchmark(parser, dataset_shuffled, use_cuda_extension=False, force_training=True)
    print("\n\n=> Test 7/8: Non-Shuffled Dataset, USE cuda extension, Force and Energy training")
    torch.cuda.empty_cache()
    gc.collect()
    benchmark(parser, dataset, use_cuda_extension=True, force_training=True)
    print("\n\n=> Test 8/8: Non-Shuffled Dataset, NO cuda extension, Force and Energy training")
    torch.cuda.empty_cache()
    gc.collect()
    benchmark(parser, dataset, use_cuda_extension=False, force_training=True)
