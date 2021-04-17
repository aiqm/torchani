import torch
import torchani
import argparse
import pkbar
from torchani.units import hartree2kcalmol
from tool_utils import time_functions_in_model


WARM_UP_BATCHES = 50
PROFILE_BATCHES = 10


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


def enable_timers(model):
    # enable timers
    aev_computer = model[0]
    nn = model[1]
    nl = aev_computer.neighborlist
    fn_to_time_aev = ['_compute_radial_aev', '_compute_angular_aev',
                             '_compute_aev', '_triple_by_molecule']

    fn_to_time_neighborlist = ['forward']
    fn_to_time_nn = ['forward']
    time_functions_in_model(nl, fn_to_time_neighborlist, nvtx=True)
    time_functions_in_model(nn, fn_to_time_nn, nvtx=True)
    time_functions_in_model(aev_computer, fn_to_time_aev, nvtx=True)


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path',
                        help='Path of the dataset, can a hdf5 file \
                            or a directory containing hdf5 files')
    parser.add_argument('-b', '--batch_size',
                        help='Number of conformations of each batch',
                        default=2560, type=int)
    parser.add_argument('-d', '--dry-run',
                        help='just run it in a CI without GPU',
                        action='store_true')
    args = parser.parse_args()
    args.device = torch.device('cpu' if args.dry_run else 'cuda')

    num_species = 4
    aev_computer = torchani.AEVComputer.like_1x()

    nn = torchani.ANIModel([atomic() for _ in range(4)])
    model = torch.nn.Sequential(aev_computer, nn).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    mse = torch.nn.MSELoss(reduction='none')

    print('=> loading dataset...')
    shifter = torchani.EnergyShifter(None)
    dataset = list(torchani.data.load(args.dataset_path).subtract_self_energies(shifter).species_to_indices().shuffle().collate(args.batch_size))

    print('=> start warming up')
    total_batch_counter = 0
    for epoch in range(0, WARM_UP_BATCHES + 1):

        print('Epoch: %d/inf' % (epoch + 1,))
        progbar = pkbar.Kbar(target=len(dataset) - 1, width=8)

        for i, properties in enumerate(dataset):

            if not args.dry_run and total_batch_counter == WARM_UP_BATCHES:
                print('=> warm up finished, start profiling')
                enable_timers(model)
                torch.cuda.cudart().cudaProfilerStart()

            PROFILING_STARTED = not args.dry_run and (total_batch_counter >= WARM_UP_BATCHES)

            if PROFILING_STARTED:
                torch.cuda.nvtx.range_push("batch{}".format(total_batch_counter))

            species = properties['species'].to(args.device)
            coordinates = properties['coordinates'].to(args.device).float()
            true_energies = properties['energies'].to(args.device).float()
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            with torch.autograd.profiler.emit_nvtx(enabled=PROFILING_STARTED, record_shapes=True):
                _, predicted_energies = model((species, coordinates))
            loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
            rmse = hartree2kcalmol((mse(predicted_energies, true_energies)).mean()).detach().cpu().numpy()

            if PROFILING_STARTED:
                torch.cuda.nvtx.range_push("backward")
            with torch.autograd.profiler.emit_nvtx(enabled=PROFILING_STARTED, record_shapes=True):
                loss.backward()
            if PROFILING_STARTED:
                torch.cuda.nvtx.range_pop()

            if PROFILING_STARTED:
                torch.cuda.nvtx.range_push("optimizer.step()")
            with torch.autograd.profiler.emit_nvtx(enabled=PROFILING_STARTED, record_shapes=True):
                optimizer.step()
            if PROFILING_STARTED:
                torch.cuda.nvtx.range_pop()

            progbar.update(i, values=[("rmse", rmse)])

            if PROFILING_STARTED:
                torch.cuda.nvtx.range_pop()

            total_batch_counter += 1
            if total_batch_counter > WARM_UP_BATCHES + PROFILE_BATCHES:
                break

        if total_batch_counter > WARM_UP_BATCHES + PROFILE_BATCHES:
            print('=> profiling terminate after {} batches'.format(PROFILE_BATCHES))
            break
