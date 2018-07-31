import sys
import torch
import ignite
import torchani
import model
import tqdm
import timeit
import tensorboardX
import math

chunk_size = 256
batch_chunks = 4
dataset_path = sys.argv[1]
dataset_checkpoint = 'dataset-checkpoint.dat'
model_checkpoint = 'checkpoint.pt'
max_epochs = 10

writer = tensorboardX.SummaryWriter()
start = timeit.default_timer()

shift_energy = torchani.EnergyShifter()
training, validation, testing = torchani.data.load_or_create(
    dataset_checkpoint, dataset_path, chunk_size,
    transform=[shift_energy.dataset_subtract_sae])
training = torchani.data.dataloader(training, batch_chunks)
validation = torchani.data.dataloader(validation, batch_chunks)
nnp = model.get_or_create_model(model_checkpoint)
batch_nnp = torchani.models.BatchModel(nnp)
container = torchani.ignite.Container({'energies': batch_nnp})
optimizer = torch.optim.Adam(nnp.parameters())

trainer = ignite.engine.create_supervised_trainer(
    container, optimizer, torchani.ignite.energy_mse_loss)
evaluator = ignite.engine.create_supervised_evaluator(container, metrics={
        'RMSE': torchani.ignite.energy_rmse_metric
    })


def hartree2kcal(x):
    return 627.509 * x


@trainer.on(ignite.engine.Events.EPOCH_STARTED)
def init_tqdm(trainer):
    trainer.state.tqdm = tqdm.tqdm(total=len(training), desc='epoch')


@trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
def update_tqdm(trainer):
    trainer.state.tqdm.update(1)


@trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
def finalize_tqdm(trainer):
    trainer.state.tqdm.close()


@trainer.on(ignite.engine.Events.EPOCH_STARTED)
def log_validation_results(trainer):
    evaluator.run(validation)
    metrics = evaluator.state.metrics
    rmse = hartree2kcal(metrics['RMSE'])
    writer.add_scalar('validation_rmse_vs_epoch', rmse, trainer.state.epoch)


@trainer.on(ignite.engine.Events.EPOCH_STARTED)
def log_time(trainer):
    elapsed = round(timeit.default_timer() - start, 2)
    writer.add_scalar('time_vs_epoch', elapsed, trainer.state.epoch)


@trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
def log_loss_and_time(trainer):
    iteration = trainer.state.iteration
    rmse = hartree2kcal(math.sqrt(trainer.state.output))
    writer.add_scalar('training_rmse_vs_iteration', rmse, iteration)


trainer.run(training, max_epochs=max_epochs)
