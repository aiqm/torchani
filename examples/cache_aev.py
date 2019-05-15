# -*- coding: utf-8 -*-
"""
Use Disk Cache of AEV to Boost Training
=======================================

In the previous :ref:`training-example` example, AEVs are computed everytime
when needed. This is not very efficient because the AEVs actually never change
during training. If one has a good SSD, it would be beneficial to cache these
AEVs.  This example shows how to use disk cache to boost training
"""

###############################################################################
# Most part of the codes in this example are line by line copy of
# :ref:`training-example`.
import torch
import ignite
import torchani
import timeit
import os
import ignite.contrib.handlers
import torch.utils.tensorboard


# training and validation set
try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()
training_path = os.path.join(path, '../dataset/ani1-up_to_gdb4/ani_gdb_s01.h5')
validation_path = os.path.join(path, '../dataset/ani1-up_to_gdb4/ani_gdb_s01.h5')  # noqa: E501

# checkpoint file to save model when validation RMSE improves
model_checkpoint = 'model.pt'

# max epochs to run the training
max_epochs = 20

# Compute training RMSE every this steps. Since the training set is usually
# huge and the loss funcition does not directly gives us RMSE, we need to
# check the training RMSE to see overfitting.
training_rmse_every = 5

# device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# batch size
batch_size = 1024

# log directory for tensorboardX
log = 'runs'

###############################################################################
# Here, there is no need to manually construct aev computer and energy shifter,
# but we do need to generate a disk cache for datasets
const_file = os.path.join(path, '../torchani/resources/ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params')
sae_file = os.path.join(path, '../torchani/resources/ani-1x_8x/sae_linfit.dat')
training_cache = './training_cache'
validation_cache = './validation_cache'

# If the cache dirs already exists, then we assume these data has already been
# cached and skip the generation part.
if not os.path.exists(training_cache):
    torchani.data.cache_aev(training_cache, training_path, batch_size, device,
                            const_file, True, sae_file)
if not os.path.exists(validation_cache):
    torchani.data.cache_aev(validation_cache, validation_path, batch_size,
                            device, const_file, True, sae_file)


###############################################################################
# The codes that define the network are also the same
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


nn = torchani.ANIModel([atomic() for _ in range(4)])
print(nn)

if os.path.isfile(model_checkpoint):
    nn.load_state_dict(torch.load(model_checkpoint))
else:
    torch.save(nn.state_dict(), model_checkpoint)


###############################################################################
# Except that at here we do not include aev computer into our pipeline, because
# the cache loader will load computed AEVs from disk.
model = nn.to(device)

###############################################################################
# This part is also a line by line copy
writer = torch.utils.tensorboard.SummaryWriter(log_dir=log)

###############################################################################
# Here we don't need to construct :class:`torchani.data.BatchedANIDataset`
# object, but instead an object of :class:`torchani.data.AEVCacheLoader`
training = torchani.data.AEVCacheLoader(training_cache)
validation = torchani.data.AEVCacheLoader(validation_cache)

###############################################################################
# The rest of the code are again the same
training = torchani.data.AEVCacheLoader(training_cache)
container = torchani.ignite.Container({'energies': model})
optimizer = torch.optim.Adam(model.parameters())
trainer = ignite.engine.create_supervised_trainer(
    container, optimizer, torchani.ignite.MSELoss('energies'))
evaluator = ignite.engine.create_supervised_evaluator(
    container,
    metrics={
        'RMSE': torchani.ignite.RMSEMetric('energies')
    })


###############################################################################
# Let's add a progress bar for the trainer
pbar = ignite.contrib.handlers.ProgressBar()
pbar.attach(trainer)


def hartree2kcal(x):
    return 627.509 * x


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

    # compute validation RMSE
    evaluate(validation, 'validation_rmse_vs_epoch')

    # compute training RMSE
    if trainer.state.epoch % training_rmse_every == 1:
        evaluate(training, 'training_rmse_vs_epoch')

    # checkpoint model
    torch.save(nn.state_dict(), model_checkpoint)


start = timeit.default_timer()


@trainer.on(ignite.engine.Events.EPOCH_STARTED)
def log_time(trainer):
    elapsed = round(timeit.default_timer() - start, 2)
    writer.add_scalar('time_vs_epoch', elapsed, trainer.state.epoch)


@trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
def log_loss(trainer):
    iteration = trainer.state.iteration
    writer.add_scalar('loss_vs_iteration', trainer.state.output, iteration)


trainer.run(training, max_epochs)
