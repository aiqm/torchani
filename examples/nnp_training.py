import torch
import ignite
import torchani
import model
import tqdm
import timeit
import tensorboardX
import math
import argparse
import json

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset_path',
                    help='Path of the dataset, can a hdf5 file \
                          or a directory containing hdf5 files')
parser.add_argument('--dataset_checkpoint',
                    help='Checkpoint file for datasets',
                    default='dataset-checkpoint.dat')
parser.add_argument('--model_checkpoint',
                    help='Checkpoint file for model',
                    default='model.pt')
parser.add_argument('-m', '--max_epochs',
                    help='Maximum number of epoches',
                    default=10, type=int)
parser.add_argument('-d', '--device',
                    help='Device of modules and tensors',
                    default=('cuda' if torch.cuda.is_available() else 'cpu'))
parser.add_argument('--chunk_size',
                    help='Number of conformations of each chunk',
                    default=256, type=int)
parser.add_argument('--batch_chunks',
                    help='Number of chunks in each minibatch',
                    default=4, type=int)
parser.add_argument('--log',
                    help='Log directory for tensorboardX',
                    default=None)
parser.add_argument('--optimizer',
                    help='Optimizer used to train the model',
                    default='Adam')
parser.add_argument('--optim_args',
                    help='Arguments to optimizers, in the format of json',
                    default='{}')
parser = parser.parse_args()

# set up the training
device = torch.device(parser.device)
writer = tensorboardX.SummaryWriter(log_dir=parser.log)
start = timeit.default_timer()

nnp, shift_energy = model.get_or_create_model('/tmp/model.pt',
                                              True, device=device)
training, validation, testing = torchani.data.load_or_create(
    parser.dataset_checkpoint, parser.dataset_path, parser.chunk_size,
    device=device, transform=[shift_energy.subtract_from_dataset])
training = torchani.data.dataloader(training, parser.batch_chunks)
validation = torchani.data.dataloader(validation, parser.batch_chunks)
container = torchani.ignite.Container({'energies': nnp})

parser.optim_args = json.loads(parser.optim_args)
optimizer = getattr(torch.optim, parser.optimizer)
optimizer = optimizer(nnp.parameters(), **parser.optim_args)

trainer = ignite.engine.create_supervised_trainer(
    container, optimizer, torchani.ignite.MSELoss('energies'))
evaluator = ignite.engine.create_supervised_evaluator(container, metrics={
        'RMSE': torchani.ignite.RMSEMetric('energies')
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


trainer.run(training, max_epochs=parser.max_epochs)
