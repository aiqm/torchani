import torch
import torchani
import time
import timeit
import argparse
import pkbar


def atomic():
    model = torch.nn.Sequential(
        torch.nn.Linear(384, 128),
        torch.nn.CELU(0.1),
        torch.nn.Linear(128, 128),
        torch.nn.CELU(0.1),
        torch.nn.Linear(128, 64),
        torch.nn.CELU(0.1),
        torch.nn.Linear(64, 1)
    )
    return model


def time_func(key, func):
    timers[key] = 0

    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        ret = func(*args, **kwargs)
        end = timeit.default_timer()
        timers[key] += end - start
        return ret

    return wrapper


def hartree2kcal(x):
    return 627.509 * x


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
    parser.add_argument('-o', '--original_dataset_api',
                        help='use original dataset api',
                        dest='dataset',
                        action='store_const',
                        const='original')
    parser.add_argument('-s', '--shuffle_dataset_api',
                        help='use shuffle dataset api',
                        dest='dataset',
                        action='store_const',
                        const='shuffle')
    parser.add_argument('-c', '--cache_dataset_api',
                        help='use cache dataset api',
                        dest='dataset',
                        action='store_const',
                        const='cache')
    parser.set_defaults(dataset='shuffle')
    parser.add_argument('-n', '--num_epochs',
                        help='epochs',
                        default=1, type=int)
    parser = parser.parse_args()

    Rcr = 5.2000e+00
    Rca = 3.5000e+00
    EtaR = torch.tensor([1.6000000e+01], device=parser.device)
    ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=parser.device)
    Zeta = torch.tensor([3.2000000e+01], device=parser.device)
    ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=parser.device)
    EtaA = torch.tensor([8.0000000e+00], device=parser.device)
    ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=parser.device)
    num_species = 4
    aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)

    nn = torchani.ANIModel([atomic() for _ in range(4)])
    model = torch.nn.Sequential(aev_computer, nn).to(parser.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    mse = torch.nn.MSELoss(reduction='none')
    timers = {}

    # enable timers
    torchani.aev.cutoff_cosine = time_func('torchani.aev.cutoff_cosine', torchani.aev.cutoff_cosine)
    torchani.aev.radial_terms = time_func('torchani.aev.radial_terms', torchani.aev.radial_terms)
    torchani.aev.angular_terms = time_func('torchani.aev.angular_terms', torchani.aev.angular_terms)
    torchani.aev.compute_shifts = time_func('torchani.aev.compute_shifts', torchani.aev.compute_shifts)
    torchani.aev.neighbor_pairs = time_func('torchani.aev.neighbor_pairs', torchani.aev.neighbor_pairs)
    torchani.aev.triu_index = time_func('torchani.aev.triu_index', torchani.aev.triu_index)
    torchani.aev.cumsum_from_zero = time_func('torchani.aev.cumsum_from_zero', torchani.aev.cumsum_from_zero)
    torchani.aev.triple_by_molecule = time_func('torchani.aev.triple_by_molecule', torchani.aev.triple_by_molecule)
    torchani.aev.compute_aev = time_func('torchani.aev.compute_aev', torchani.aev.compute_aev)
    model[0].forward = time_func('total', model[0].forward)
    model[1].forward = time_func('forward', model[1].forward)

    if parser.dataset == 'shuffle':
        torchani.data.ShuffledDataset = time_func('data_loading', torchani.data.ShuffledDataset)
        print('using shuffle dataset API')
        print('=> loading dataset...')
        dataset = torchani.data.ShuffledDataset(file_path=parser.dataset_path,
                                                species_order=['H', 'C', 'N', 'O'],
                                                subtract_self_energies=True,
                                                batch_size=parser.batch_size,
                                                num_workers=2)
        print('=> the first batch is ([chunk1, chunk2, ...], {"energies", "force", ...}) in which chunk1=(species, coordinates)')
        chunks, properties = iter(dataset).next()
    elif parser.dataset == 'original':
        torchani.data.load_ani_dataset = time_func('data_loading', torchani.data.load_ani_dataset)
        print('using original dataset API')
        print('=> loading dataset...')
        energy_shifter = torchani.utils.EnergyShifter(None)
        species_to_tensor = torchani.utils.ChemicalSymbolsToInts('HCNO')
        dataset = torchani.data.load_ani_dataset(parser.dataset_path, species_to_tensor,
                                                 parser.batch_size, device=parser.device,
                                                 transform=[energy_shifter.subtract_from_dataset])
        print('=> the first batch is ([chunk1, chunk2, ...], {"energies", "force", ...}) in which chunk1=(species, coordinates)')
        chunks, properties = dataset[0]
    elif parser.dataset == 'cache':
        torchani.data.CachedDataset = time_func('data_loading', torchani.data.CachedDataset)
        print('using cache dataset API')
        print('=> loading dataset...')
        dataset = torchani.data.CachedDataset(file_path=parser.dataset_path,
                                              species_order=['H', 'C', 'N', 'O'],
                                              subtract_self_energies=True,
                                              batch_size=parser.batch_size)
        print('=> caching all dataset into cpu')
        pbar = pkbar.Pbar('loading and processing dataset into cpu memory, total '
                          + 'batches: {}, batch_size: {}'.format(len(dataset), parser.batch_size),
                          len(dataset))
        for i, t in enumerate(dataset):
            pbar.update(i)
        print('=> the first batch is ([chunk1, chunk2, ...], {"energies", "force", ...}) in which chunk1=(species, coordinates)')
        chunks, properties = dataset[0]

    for i, chunk in enumerate(chunks):
        print('chunk{}'.format(i + 1), list(chunk[0].size()), list(chunk[1].size()))
    print('energies', list(properties['energies'].size()))

    print('=> start training')
    start = time.time()

    for epoch in range(0, parser.num_epochs):

        print('Epoch: %d/%d' % (epoch + 1, parser.num_epochs))
        progbar = pkbar.Kbar(target=len(dataset) - 1, width=8)

        for i, (batch_x, batch_y) in enumerate(dataset):

            true_energies = batch_y['energies'].to(parser.device)
            predicted_energies = []
            num_atoms = []

            for chunk_species, chunk_coordinates in batch_x:
                chunk_species = chunk_species.to(parser.device)
                chunk_coordinates = chunk_coordinates.to(parser.device)
                num_atoms.append((chunk_species >= 0).to(true_energies.dtype).sum(dim=1))
                _, chunk_energies = model((chunk_species, chunk_coordinates))
                predicted_energies.append(chunk_energies)

            num_atoms = torch.cat(num_atoms)
            predicted_energies = torch.cat(predicted_energies).to(true_energies.dtype)
            loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
            rmse = hartree2kcal((mse(predicted_energies, true_energies)).mean()).detach().cpu().numpy()
            loss.backward()
            optimizer.step()

            progbar.update(i, values=[("rmse", rmse)])
    stop = time.time()

    print('=> more detail about benchmark')
    for k in timers:
        if k.startswith('torchani.'):
            print('{} - {:.1f}s'.format(k, timers[k]))
    print('Total AEV - {:.1f}s'.format(timers['total']))
    print('Data Loading - {:.1f}s'.format(timers['data_loading']))
    print('NN - {:.1f}s'.format(timers['forward']))
    print('Epoch time - {:.1f}s'.format(stop - start))
