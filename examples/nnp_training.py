# -*- coding: utf-8 -*-
"""
.. _training-example:

Train Your Own Neural Network Potential
=======================================

This example shows how to use TorchANI to train a neural network potential. We
will use the same configuration as specified as in `inputtrain.ipt`_

.. _`inputtrain.ipt`:
    https://github.com/aiqm/torchani/blob/master/torchani/resources/ani-1x_8x/inputtrain.ipt

.. note::
    TorchANI provide tools to run NeuroChem training config file `inputtrain.ipt`.
    See: :ref:`neurochem-training`.
"""

###############################################################################
# To begin with, let's first import the modules we will use:
import torch
import torchani
import os
import math
import torch.utils.tensorboard

# device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###############################################################################
# Now let's setup constants and construct an AEV computer. These numbers could
# be found in `rHCNO-5.2R_16-3.5A_a4-8.params`_ and `sae_linfit.dat`_
#
# .. _rHCNO-5.2R_16-3.5A_a4-8.params:
#   https://github.com/aiqm/torchani/blob/master/torchani/resources/ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params
# .. _sae_linfit.dat:
#   https://github.com/aiqm/torchani/blob/master/torchani/resources/ani-1x_8x/sae_linfit.dat
Rcr = 5.2000e+00
Rca = 3.5000e+00
EtaR = torch.tensor([1.6000000e+01], device=device)
ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
Zeta = torch.tensor([3.2000000e+01], device=device)
ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
EtaA = torch.tensor([8.0000000e+00], device=device)
ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=device)
num_species = 4
aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
energy_shifter = torchani.utils.EnergyShifter([
    -0.600952980000,  # H
    -38.08316124000,  # C
    -54.70775770000,  # N
    -75.19446356000,  # O
])
species_to_tensor = torchani.utils.ChemicalSymbolsToInts('HCNO')


###############################################################################
# Now let's setup datasets. Note that here for our demo purpose, we set both
# training set and validation set the ``ani_gdb_s01.h5`` in TorchANI's
# repository. This allows this program to finish very quick, because that
# dataset is very small. But this is wrong and should be avoided for any
# serious training. These paths assumes the user run this script under the
# ``examples`` directory of TorchANI's repository. If you download this script,
# you should manually set the path of these files in your system before this
# script can run successfully.
#
# Now load training and validation datasets into memory. Note that we need to
# subtracting energies by the self energies of all atoms for each molecule.
# This makes the range of energies in a reasonable range. The second argument
# defines how to convert species as a list of string to tensor, that is, for
# all supported chemical symbols, which is correspond to ``0``, which
# correspond to ``1``, etc.

try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()
training_path = os.path.join(path, '../dataset/ani1-up_to_gdb4/ani_gdb_s01.h5')
validation_path = os.path.join(path, '../dataset/ani1-up_to_gdb4/ani_gdb_s01.h5')

batch_size = 2560

training = torchani.data.BatchedANIDataset(
    training_path, species_to_tensor, batch_size, device=device,
    transform=[energy_shifter.subtract_from_dataset])

validation = torchani.data.BatchedANIDataset(
    validation_path, species_to_tensor, batch_size, device=device,
    transform=[energy_shifter.subtract_from_dataset])

###############################################################################
# When iterating the dataset, we will get pairs of input and output
# ``(species_coordinates, properties)``, where ``species_coordinates`` is the
# input and ``properties`` is the output.
#
# ``species_coordinates`` is a list of species-coordinate pairs, with shape
# ``(N, Na)`` and ``(N, Na, 3)``. The reason for getting this type is, when
# loading the dataset and generating minibatches, the whole dataset are
# shuffled and each minibatch contains structures of molecules with a wide
# range of number of atoms. Molecules of different number of atoms are batched
# into single by padding. The way padding works is: adding ghost atoms, with
# species 'X', and do computations as if they were normal atoms. But when
# computing AEVs, atoms with species `X` would be ignored. To avoid computation
# wasting on padding atoms, minibatches are further splitted into chunks. Each
# chunk contains structures of molecules of similar size, which minimize the
# total number of padding atoms required to add. The input list
# ``species_coordinates`` contains chunks of that minibatch we are getting. The
# batching and chunking happens automatically, so the user does not need to
# worry how to construct chunks, but the user need to compute the energies for
# each chunk and concat them into single tensor.
#
# The output, i.e. ``properties`` is a dictionary holding each property. This
# allows us to extend TorchANI in the future to training forces and properties.


###############################################################################
# Now let's define atomic neural networks. Here in this demo, we use the same
# size of neural network for all atom types, but this is not necessary.

H_network = torch.nn.Sequential(
    torch.nn.Linear(384, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

C_network = torch.nn.Sequential(
    torch.nn.Linear(384, 144),
    torch.nn.CELU(0.1),
    torch.nn.Linear(144, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

N_network = torch.nn.Sequential(
    torch.nn.Linear(384, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

O_network = torch.nn.Sequential(
    torch.nn.Linear(384, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

nn = torchani.ANIModel([H_network, C_network, N_network, O_network])
print(nn)

###############################################################################
# Let's now create a pipeline of AEV Computer --> Neural Networks.
model = torch.nn.Sequential(aev_computer, nn).to(device)

###############################################################################
# Now let's setup the optimizer. We need to specify different weight decay rate
# for different parameters.
# .. note::
#   The weight decay in `inputtrain.ipt`_ is named "l2", but it is actually not
#   L2 regularization. The confusion between L2 and weight decay is a common
#   mistake in deep learning.  See: `Decoupled Weight Decay Regularization`_
#
# .. _Decoupled Weight Decay Regularization:
#   https://arxiv.org/abs/1711.05101
optimizer = torch.optim.Adam([
    {'params': H_network[0].parameters(), 'weight_decay': 0.0001},
    {'params': H_network[2].parameters(), 'weight_decay': 0.00001},
    {'params': H_network[4].parameters(), 'weight_decay': 0.000001},
    {'params': H_network[6].parameters()},
    {'params': C_network[0].parameters(), 'weight_decay': 0.0001},
    {'params': C_network[2].parameters(), 'weight_decay': 0.00001},
    {'params': C_network[4].parameters(), 'weight_decay': 0.000001},
    {'params': C_network[6].parameters()},
    {'params': N_network[0].parameters(), 'weight_decay': 0.0001},
    {'params': N_network[2].parameters(), 'weight_decay': 0.00001},
    {'params': N_network[4].parameters(), 'weight_decay': 0.000001},
    {'params': N_network[6].parameters()},
    {'params': O_network[0].parameters(), 'weight_decay': 0.0001},
    {'params': O_network[2].parameters(), 'weight_decay': 0.00001},
    {'params': O_network[4].parameters(), 'weight_decay': 0.000001},
    {'params': O_network[6].parameters()},
])

###############################################################################
# The way ANI trains a neural network potential looks like this:
# Phase 1: Pretrain the model by minimizing MSE loss
# Phase 2: Train the model by minimizing the exponential loss, until validation
#   RMSE no longer improves for a certain steps, decay the learning rate and
#   repeat the same process, stop until the learning rate is smaller than a
#   certain number.
#
# We first read the checkpoint files to find where we are. We use `latest.pt`
# to store current training state. If `latest.pt` does not exist, this
# this means the pretraining has not been finished yet.
latest_checkpoint = 'latest.pt'
pretrained = os.path.isfile(latest_checkpoint)

###############################################################################
# If the model is already pretrained, then we just load the lastest checkpoint
# file and nothing else should be done. Otherwise we need to run the pretrain.
pretrain_epoches = 10
mse = torch.nn.MSELoss()

if not pretrained:
    epoch = 0
    optimizer = optimizer
    for i in range(pretrain_epoches):
        for batch_x, batch_y in training:
            true_energies = batch_y['energies']
            predicted_energies = []
            for chunk_species, chunk_coordinates in batch_x:
                _, chunk_energies = model((chunk_species, chunk_coordinates))
                predicted_energies.append(chunk_energies)
            predicted_energies = torch.cat(predicted_energies)
            loss = mse(predicted_energies, true_energies)
            loss.backward()
            optimizer.step()
    torch.save({
        'nn': nn.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, latest_checkpoint)

###############################################################################
# For phase 2, we need a learning rate scheduler to do learning rate decay
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100)

###############################################################################
# We will also use TensorBoard to visualize our training process
tensorboard = torch.utils.tensorboard.SummaryWriter()

###############################################################################
# If pretrain phase has been done, then resume normal training
if pretrained:
    checkpoint = torch.load(latest_checkpoint)
    nn.load_state_dict(checkpoint['nn'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

###############################################################################
# During training, we need to validate on validation set and if validation error
# is better than the best, then save the new best model to a checkpoint


# helper function to convert energy unit from Hartree to kcal/mol
def hartree2kcal(x):
    return 627.509 * x


def validate():
    # run validation
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0
    for batch_x, batch_y in validation:
        true_energies = batch_y['energies']
        predicted_energies = []
        for chunk_species, chunk_coordinates in batch_x:
            _, chunk_energies = model((chunk_species, chunk_coordinates))
            predicted_energies.append(chunk_energies)
        predicted_energies = torch.cat(predicted_energies)
        total_mse += mse_sum(predicted_energies, true_energies)
        count += predicted_energies.shape[0]
    return hartree2kcal(math.sqrt(total_mse / count))


###############################################################################
# Finally, we come to the training loop
# In this tutorial, we are setting the maximum epoch to a very small number,
# only to make this demo terminate fast. For serious training, this should be
# set to a much larger value or even `math.inf`
max_epochs = 20
early_stopping_learning_rate = 1.0E-5
best_model_checkpoint = 'best.pt'

for _ in range(scheduler.last_epoch + 1, max_epochs):
    rmse = validate()
    scheduler.step(rmse)
    learning_rate = optimizer.param_groups[0]['lr']

    tensorboard.add_scalar('validation_rmse', rmse, scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse', scheduler.best, scheduler.last_epoch)
    tensorboard.add_scalar('learning_rate', learning_rate, scheduler.last_epoch)

    # checkpoint
    if scheduler.is_better(rmse, scheduler.best):
        torch.save(nn.state_dict(), best_model_checkpoint)

    for i, (batch_x, batch_y) in enumerate(training):
        true_energies = batch_y['energies']
        predicted_energies = []
        for chunk_species, chunk_coordinates in batch_x:
            _, chunk_energies = model((chunk_species, chunk_coordinates))
            predicted_energies.append(chunk_energies)
        predicted_energies = torch.cat(predicted_energies)
        loss = mse(predicted_energies, true_energies)
        loss = 0.5 * (torch.exp(2 * loss) - 1)
        loss.backward()
        optimizer.step()

        # write current batch loss to TensorBoard
        tensorboard.add_scalar('batch_loss', loss, scheduler.last_epoch * len(training) + i)

    torch.save({
        'nn': nn.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, latest_checkpoint)
