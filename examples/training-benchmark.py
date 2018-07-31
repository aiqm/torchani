import sys
import torch
import ignite
import torchani
import timeit
import model
import tqdm

chunk_size = 256
batch_chunks = 4
dataset_path = sys.argv[1]
shift_energy = torchani.EnergyShifter()
dataset = torchani.data.ANIDataset(
    dataset_path, chunk_size,
    transform=[shift_energy.dataset_subtract_sae])
dataloader = torchani.data.dataloader(dataset, batch_chunks)
nnp = model.get_or_create_model('/tmp/model.pt', True)
batch_nnp = torchani.models.BatchModel(nnp)
container = torchani.ignite.Container({'energies': batch_nnp})
optimizer = torch.optim.Adam(nnp.parameters())

trainer = ignite.engine.create_supervised_trainer(
    container, optimizer, torchani.ignite.energy_mse_loss)


@trainer.on(ignite.engine.Events.EPOCH_STARTED)
def init_tqdm(trainer):
    trainer.state.tqdm = tqdm.tqdm(total=len(dataloader), desc='epoch')


@trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
def update_tqdm(trainer):
    trainer.state.tqdm.update(1)


@trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
def finalize_tqdm(trainer):
    trainer.state.tqdm.close()


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
