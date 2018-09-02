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
parser.add_argument('training_path',
                    help='Path of the training set, can be a hdf5 file \
                          or a directory containing hdf5 files')
parser.add_argument('validation_path',
                    help='Path of the validation set, can be a hdf5 file \
                          or a directory containing hdf5 files')
parser.add_argument('--model_checkpoint',
                    help='Checkpoint file for model',
                    default='model.pt')
parser.add_argument('-m', '--max_epochs',
                    help='Maximum number of epoches',
                    default=300, type=int)
parser.add_argument('--training_rmse_every',
                    help='Compute training RMSE every epoches',
                    default=20, type=int)
parser.add_argument('-d', '--device',
                    help='Device of modules and tensors',
                    default=('cuda' if torch.cuda.is_available() else 'cpu'))
parser.add_argument('--batch_size',
                    help='Number of conformations of each batch',
                    default=1024, type=int)
parser.add_argument('--log',
                    help='Log directory for tensorboardX',
                    default=None)
parser.add_argument('--optimizer',
                    help='Optimizer used to train the model',
                    default='Adam')
parser.add_argument('--optim_args',
                    help='Arguments to optimizers, in the format of json',
                    default='{}')
parser.add_argument('--early_stopping',
                    help='Stop after epoches of no improvements',
                    default=math.inf, type=int)
parser = parser.parse_args()

# set up the training
device = torch.device(parser.device)
writer = tensorboardX.SummaryWriter(log_dir=parser.log)
start = timeit.default_timer()

nnp = model.get_or_create_model(parser.model_checkpoint, device=device)
training = torchani.data.BatchedANIDataset(
    parser.training_path, model.consts.species_to_tensor,
    parser.batch_size, device=device,
    transform=[model.shift_energy.subtract_from_dataset])
validation = torchani.data.BatchedANIDataset(
    parser.validation_path, model.consts.species_to_tensor,
    parser.batch_size, device=device,
    transform=[model.shift_energy.subtract_from_dataset])
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


@trainer.on(ignite.engine.Events.STARTED)
def initialize(trainer):
    trainer.state.best_validation_rmse = math.inf
    trainer.state.no_improve_count = 0


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
def validation_and_checkpoint(trainer):
    def evaluate(dataset, name):
        evaluator = ignite.engine.create_supervised_evaluator(
            container,
            metrics={
                'RMSE': torchani.ignite.RMSEMetric('energies')
            }
        )
        evaluator.run(dataset)
        metrics = evaluator.state.metrics
        rmse = hartree2kcal(metrics['RMSE'])
        writer.add_scalar(name, rmse, trainer.state.epoch)
        return rmse

    # compute validation RMSE
    rmse = evaluate(validation, 'validation_rmse_vs_epoch')

    # compute training RMSE
    if trainer.state.epoch % parser.training_rmse_every == 1:
        evaluate(training, 'training_rmse_vs_epoch')

    # handle best validation RMSE
    if rmse < trainer.state.best_validation_rmse:
        trainer.state.no_improve_count = 0
        trainer.state.best_validation_rmse = rmse
        writer.add_scalar('best_validation_rmse_vs_epoch', rmse,
                          trainer.state.epoch)
        torch.save(nnp.state_dict(), parser.model_checkpoint)
    else:
        trainer.state.no_improve_count += 1
    writer.add_scalar('no_improve_count_vs_epoch',
                      trainer.state.no_improve_count,
                      trainer.state.epoch)

    if trainer.state.no_improve_count > parser.early_stopping:
            trainer.terminate()


@trainer.on(ignite.engine.Events.EPOCH_STARTED)
def log_time(trainer):
    elapsed = round(timeit.default_timer() - start, 2)
    writer.add_scalar('time_vs_epoch', elapsed, trainer.state.epoch)


@trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
def log_loss_and_time(trainer):
    iteration = trainer.state.iteration
    writer.add_scalar('loss_vs_iteration', trainer.state.output, iteration)


trainer.run(training, max_epochs=parser.max_epochs)
