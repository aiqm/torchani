# -*- coding: utf-8 -*-
"""
.. _training-example-ignite:

Train Your Own Neural Network Potential, Using PyTorch-Ignite
=============================================================

We have seen how to train a neural network potential by manually writing
training loop in :ref:`training-example`. TorchANI provide tools to work
with PyTorch-Ignite to simplify the writing of training code. This tutorial
shows how to use these tools to train a demo model.

This tutorial assumes readers have read :ref:`training-example`.
"""

###############################################################################
# To begin with, let's first import the modules we will use:
import torch
import ignite
import torchani
import timeit
import os
import ignite.contrib.handlers
import torch.utils.tensorboard


###############################################################################
# Now let's setup training hyperparameters and dataset.

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

# log directory for tensorboard
log = 'runs'


###############################################################################
# Instead of manually specifying hyperparameters as in :ref:`training-example`,
# here we will load them from files.
const_file = os.path.join(path, '../torchani/resources/ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params')  # noqa: E501
sae_file = os.path.join(path, '../torchani/resources/ani-1x_8x/sae_linfit.dat')  # noqa: E501
consts = torchani.neurochem.Constants(const_file)
aev_computer = torchani.AEVComputer(**consts)
energy_shifter = torchani.neurochem.load_sae(sae_file)


###############################################################################
# Now let's define atomic neural networks. Here in this demo, we use the same
# size of neural network for all atom types, but this is not necessary.
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

###############################################################################
# If checkpoint from previous training exists, then load it.
if os.path.isfile(model_checkpoint):
    nn.load_state_dict(torch.load(model_checkpoint))
else:
    torch.save(nn.state_dict(), model_checkpoint)

###############################################################################
# Let's now create a pipeline of AEV Computer --> Neural Networks.
model = torch.nn.Sequential(aev_computer, nn).to(device)

###############################################################################
# Now setup tensorboard
writer = torch.utils.tensorboard.SummaryWriter(log_dir=log)

###############################################################################
# Now load training and validation datasets into memory.
training = torchani.data.BatchedANIDataset(
    training_path, consts.species_to_tensor, batch_size, device=device,
    transform=[energy_shifter.subtract_from_dataset])

validation = torchani.data.BatchedANIDataset(
    validation_path, consts.species_to_tensor, batch_size, device=device,
    transform=[energy_shifter.subtract_from_dataset])

###############################################################################
# We have tools to deal with the chunking (see :ref:`training-example`). These
# tools can be used as follows:
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


###############################################################################
# And some event handlers to compute validation and training metrics:
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


###############################################################################
# Also some to log elapsed time:
start = timeit.default_timer()


@trainer.on(ignite.engine.Events.EPOCH_STARTED)
def log_time(trainer):
    elapsed = round(timeit.default_timer() - start, 2)
    writer.add_scalar('time_vs_epoch', elapsed, trainer.state.epoch)


###############################################################################
# Also log the loss per iteration:
@trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
def log_loss(trainer):
    iteration = trainer.state.iteration
    writer.add_scalar('loss_vs_iteration', trainer.state.output, iteration)


###############################################################################
# And finally, we are ready to run:
trainer.run(training, max_epochs)
