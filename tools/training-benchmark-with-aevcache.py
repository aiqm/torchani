import torch
import ignite
import torchani
import timeit
import tqdm
import argparse

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('cache_path',
                    help='Path of the aev cache')
parser.add_argument('-d', '--device',
                    help='Device of modules and tensors',
                    default=('cuda' if torch.cuda.is_available() else 'cpu'))
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


nnp = torch.nn.Sequential(model, Flatten()).to(device)

dataset = torchani.data.AEVCacheLoader(parser.cache_path)
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
nnp[0].forward = time_func('forward', nnp[0].forward)

# run it!
start = timeit.default_timer()
trainer.run(dataset, max_epochs=1)
elapsed = round(timeit.default_timer() - start, 2)
print('NN:', timers['forward'])
print('Epoch time:', elapsed)
