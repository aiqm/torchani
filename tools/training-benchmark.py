import torch
import torchani
import time
import timeit
import argparse
import pkbar
from torchani.units import hartree2kcalmol
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


def time_func(key, func):
    timers[key] = 0

    def wrapper(*args, **kwargs):
        if synchronize:
            torch.cuda.synchronize()
        start = timeit.default_timer()
        ret = func(*args, **kwargs)
        if synchronize:
            torch.cuda.synchronize()
        end = timeit.default_timer()
        timers[key] += end - start
        return ret

    return wrapper


def time_functions_in_module(module, function_names_list):
    # Wrap all the functions from "function_names_list" from the module
    # "module" with a timer
    for n in function_names_list:
        setattr(module, n, time_func(f'{module.__name__}.{n}', getattr(module, n)))


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
                        help='whether to insert torch.cuda.synchronize() at the start and end of each function')
    parser.add_argument('-n', '--num_epochs',
                        help='epochs',
                        default=1, type=int)
    args = parser.parse_args()

    if args.synchronize:
        synchronize = True
    else:
        synchronize = False
        print('WARNING: Synchronization creates some small overhead but if CUDA'
              ' streams are not synchronized the timings before and after a'
              ' function do not reflect the actual calculation load that'
              ' function is performing. Only run this benchmark without'
              ' synchronization if you know very well what you are doing')

    Rcr = 5.2000e+00
    Rca = 3.5000e+00
    EtaR = torch.tensor([1.6000000e+01], device=args.device)
    ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=args.device)
    Zeta = torch.tensor([3.2000000e+01], device=args.device)
    ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=args.device)
    EtaA = torch.tensor([8.0000000e+00], device=args.device)
    ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=args.device)
    num_species = 4
    aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)

    nn = torchani.ANIModel([H_network, C_network, N_network, O_network])
    model = torch.nn.Sequential(aev_computer, nn).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    mse = torch.nn.MSELoss(reduction='none')

    # enable timers
    functions_to_time = ['cutoff_cosine', 'radial_terms', 'angular_terms',
                         'compute_shifts', 'neighbor_pairs',
                         'neighbor_pairs_nopbc', 'cumsum_from_zero',
                         'triple_by_molecule', 'compute_aev']

    timers = {fn: 0.0 for fn in functions_to_time}

    time_functions_in_module(torchani.aev, functions_to_time)

    model[0].forward = time_func('total', model[0].forward)
    model[1].forward = time_func('forward', model[1].forward)

    print('=> loading dataset...')
    shifter = torchani.EnergyShifter(None)
    dataset = torchani.data.load(args.dataset_path).subtract_self_energies(shifter).species_to_indices().shuffle().collate(args.batch_size).cache()

    print('=> start training')
    start = time.time()

    for epoch in range(0, args.num_epochs):

        print('Epoch: %d/%d' % (epoch + 1, args.num_epochs))
        progbar = pkbar.Kbar(target=len(dataset) - 1, width=8)

        for i, properties in enumerate(dataset):
            species = properties['species'].to(args.device)
            coordinates = properties['coordinates'].to(args.device).float()
            true_energies = properties['energies'].to(args.device).float()
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            _, predicted_energies = model((species, coordinates))
            loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
            rmse = hartree2kcalmol((mse(predicted_energies, true_energies)).mean()).detach().cpu().numpy()
            loss.backward()
            optimizer.step()

            progbar.update(i, values=[("rmse", rmse)])
    if synchronize:
        torch.cuda.synchronize()
    stop = time.time()

    print('=> more detail about benchmark')
    for k in timers:
        if k.startswith('torchani.'):
            print('{} - {:.2f}s'.format(k, timers[k]))
    print('Total AEV - {:.2f}s'.format(timers['total']))
    print('NN - {:.2f}s'.format(timers['forward']))
    print('Epoch time - {:.2f}s'.format(stop - start))
