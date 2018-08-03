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
parser.add_argument('--chunk_size',
                    help='Number of conformations of each chunk',
                    default=256, type=int)
parser.add_argument('--batch_chunks',
                    help='Number of chunks in each minibatch',
                    default=4, type=int)
parser = parser.parse_args()

# set up benchmark
device = torch.device(parser.device)
nnp, shift_energy = model.get_or_create_model('/tmp/model.pt',
                                              True, device=device)
dataset = torchani.data.ANIDataset(
    parser.dataset_path, parser.chunk_size, device=device,
    transform=[shift_energy.subtract_from_dataset])
dataloader = torchani.data.dataloader(dataset, parser.batch_chunks)
container = torchani.ignite.Container({'energies': nnp})
optimizer = torch.optim.Adam(nnp.parameters())

trainer = ignite.engine.create_supervised_trainer(
    container, optimizer, torchani.ignite.MSELoss('energies'))


@trainer.on(ignite.engine.Events.EPOCH_STARTED)
def init_tqdm(trainer):
    trainer.state.tqdm = tqdm.tqdm(total=len(dataloader), desc='epoch')


@trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
def update_tqdm(trainer):
    trainer.state.tqdm.update(1)


@trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
def finalize_tqdm(trainer):
    trainer.state.tqdm.close()


# run it!
start = timeit.default_timer()
trainer.run(dataloader, max_epochs=1)
elapsed = round(timeit.default_timer() - start, 2)
print('Radial terms:', nnp[1].timers['radial terms'])
print('Angular terms:', nnp[1].timers['angular terms'])
print('Terms and indices:', nnp[1].timers['terms and indices'])
print('Combinations:', nnp[1].timers['combinations'])
print('Mask R:', nnp[1].timers['mask_r'])
print('Mask A:', nnp[1].timers['mask_a'])
print('Assemble:', nnp[1].timers['assemble'])
print('Total AEV:', nnp[1].timers['total'])
print('NN:', nnp[2].timers['forward'])
print('Epoch time:', elapsed)
