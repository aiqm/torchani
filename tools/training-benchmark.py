import torch
import torchani
import time
import argparse
import pkbar
from typing import Dict
from torchani.units import hartree2kcalmol
from tool_utils import time_functions_in_model

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

    aev_computer = torchani.AEVComputer.like_1x()

    nn = torchani.ANIModel([H_network, C_network, N_network, O_network])
    model = torch.nn.Sequential(aev_computer, nn).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    mse = torch.nn.MSELoss(reduction='none')

    # enable timers
    timers: Dict[str, int] = dict()

    # time these functions

    fn_to_time_aev = ['_compute_radial_aev', '_compute_angular_aev',
                             '_compute_aev', '_triple_by_molecule', 'forward']
    fn_to_time_neighborlist = ['forward']
    fn_to_time_nn = ['forward']
    fn_to_time_angular = ['forward']
    fn_to_time_radial = ['forward']

    time_functions_in_model(aev_computer.angular_terms, fn_to_time_angular, timers, synchronize)
    time_functions_in_model(aev_computer.radial_terms, fn_to_time_radial, timers, synchronize)
    time_functions_in_model(aev_computer, fn_to_time_aev, timers, synchronize)
    time_functions_in_model(aev_computer.neighborlist, fn_to_time_neighborlist, timers, synchronize)
    time_functions_in_model(nn, fn_to_time_nn, timers, synchronize)

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

    for k in timers.keys():
        timers[k] = timers[k] / args.num_epochs

    print('=> more detail about benchmark')
    for k in timers:
        if k not in ['AEVComputer.forward', 'ANIModel.forward']:
            print('{} - {:.3f}s'.format(k, timers[k]))
    print('Total AEV forward - {:.3f}s'.format(timers['AEVComputer.forward']))
    print('Total NN forward - {:.3f}s'.format(timers['ANIModel.forward']))
    print('Total epoch time - {:.3f}s'.format((stop - start) / args.num_epochs))
