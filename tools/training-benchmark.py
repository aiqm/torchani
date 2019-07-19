import torch
import ignite
import torchani
import timeit
import tqdm
import argparse

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset_path',
                    help='Path of the dataset, can a hdf5 file \
                          or a directory containing hdf5 files')
parser.add_argument('-d', '--device',
                    help='Device of modules and tensors',
                    default=('cuda' if torch.cuda.is_available() else 'cpu'))
parser.add_argument('--batch_size',
                    help='Number of conformations of each batch',
                    default=1024, type=int)
parser = parser.parse_args()

# set up benchmark
device = torch.device(parser.device)
ani1x = torchani.models.ANI1x()
consts = ani1x.consts
aev_computer = ani1x.aev_computer
shift_energy = ani1x.energy_shifter


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


model = torchani.ANIModel([atomic() for _ in range(4)])


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x[0], x[1].flatten()


nnp = torch.nn.Sequential(aev_computer, model, Flatten()).to(device)

dataset = torchani.data.BatchedANIDataset(
    parser.dataset_path, consts.species_to_tensor,
    parser.batch_size, device=device,
    transform=[shift_energy.subtract_from_dataset])
container = torchani.ignite.Container({'energies': nnp})
optimizer = torch.optim.Adam(nnp.parameters())

trainer = ignite.engine.create_supervised_trainer(
    container, optimizer, torchani.ignite.MSELoss('energies'))


@trainer.on(ignite.engine.Events.EPOCH_STARTED)
def init_tqdm(trainer):
    trainer.state.tqdm = tqdm.tqdm(total=len(dataset), desc='epoch')


@trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
def update_tqdm(trainer):
    trainer.state.tqdm.update(1)


@trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
def finalize_tqdm(trainer):
    trainer.state.tqdm.close()


timers = {}


def time_func(key, func):
    timers[key] = 0

    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        ret = func(*args, **kwargs)
        end = timeit.default_timer()
        timers[key] += end - start
        return ret

    return wrapper


# enable timers
torchani.aev.cutoff_cosine = time_func('torchani.aev.cutoff_cosine', torchani.aev.cutoff_cosine)
torchani.aev.radial_terms = time_func('torchani.aev.radial_terms', torchani.aev.radial_terms)
torchani.aev.angular_terms = time_func('torchani.aev.angular_terms', torchani.aev.angular_terms)
torchani.aev.compute_shifts = time_func('torchani.aev.compute_shifts', torchani.aev.compute_shifts)
torchani.aev.neighbor_pairs = time_func('torchani.aev.neighbor_pairs', torchani.aev.neighbor_pairs)
torchani.aev.triu_index = time_func('torchani.aev.triu_index', torchani.aev.triu_index)
torchani.aev.convert_pair_index = time_func('torchani.aev.convert_pair_index', torchani.aev.convert_pair_index)
torchani.aev.cumsum_from_zero = time_func('torchani.aev.cumsum_from_zero', torchani.aev.cumsum_from_zero)
torchani.aev.triple_by_molecule = time_func('torchani.aev.triple_by_molecule', torchani.aev.triple_by_molecule)
torchani.aev.compute_aev = time_func('torchani.aev.compute_aev', torchani.aev.compute_aev)
nnp[0].forward = time_func('total', nnp[0].forward)
nnp[1].forward = time_func('forward', nnp[1].forward)

# run it!
start = timeit.default_timer()
trainer.run(dataset, max_epochs=1)
elapsed = round(timeit.default_timer() - start, 2)
for k in timers:
    if k.startswith('torchani.'):
        print(k, timers[k])
print('Total AEV:', timers['total'])
print('NN:', timers['forward'])
print('Epoch time:', elapsed)
