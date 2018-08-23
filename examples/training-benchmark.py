import torch
import ignite
import torchani
import timeit
import model
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
nnp = model.get_or_create_model('/tmp/model.pt', device=device)
shift_energy = torchani.buildins.energy_shifter
dataset = torchani.data.BatchedANIDataset(
    parser.dataset_path, model.consts.species_to_tensor,
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
nnp[0]._radial_subaev_terms = time_func('radial terms',
                                        nnp[0]._radial_subaev_terms)
nnp[0]._angular_subaev_terms = time_func('angular terms',
                                         nnp[0]._angular_subaev_terms)
nnp[0]._terms_and_indices = time_func('terms and indices',
                                      nnp[0]._terms_and_indices)
nnp[0]._combinations = time_func('combinations', nnp[0]._combinations)
nnp[0]._compute_mask_r = time_func('mask_r', nnp[0]._compute_mask_r)
nnp[0]._compute_mask_a = time_func('mask_a', nnp[0]._compute_mask_a)
nnp[0]._assemble = time_func('assemble', nnp[0]._assemble)
nnp[0].forward = time_func('total', nnp[0].forward)
nnp[1].forward = time_func('forward', nnp[1].forward)

# run it!
start = timeit.default_timer()
trainer.run(dataset, max_epochs=1)
elapsed = round(timeit.default_timer() - start, 2)
print('Radial terms:', timers['radial terms'])
print('Angular terms:', timers['angular terms'])
print('Terms and indices:', timers['terms and indices'])
print('Combinations:', timers['combinations'])
print('Mask R:', timers['mask_r'])
print('Mask A:', timers['mask_a'])
print('Assemble:', timers['assemble'])
print('Total AEV:', timers['total'])
print('NN:', timers['forward'])
print('Epoch time:', elapsed)
