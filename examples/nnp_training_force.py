# -*- coding: utf-8 -*-
"""
.. _force-training-example:

Train To Force
==============

We have seen how to train a neural network potential by manually writing
training loop in :ref:`training-example`. This tutorial shows how to modify
that script to train to force.
"""

###############################################################################
# Most part of the script are the same as :ref:`training-example`, we will omit
# the comments for these parts. Please refer to :ref:`training-example` for more
# information
import torch
import torchani
import os
import math
import torch.utils.tensorboard
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()
training_path = os.path.join(path, '../dataset/ani-1x/sample.h5')
validation_path = os.path.join(path, '../dataset/ani-1x/sample.h5')

batch_size = 2560


###############################################################################
# The code to create the dataset is a bit different: we need to manually
# specify that ``atomic_properties=['forces']`` so that forces will be read
# from hdf5 files.
training = torchani.data.BatchedANIDataset(
    training_path, species_to_tensor, batch_size, device=device,
    atomic_properties=['forces'],
    transform=[energy_shifter.subtract_from_dataset])

validation = torchani.data.BatchedANIDataset(
    validation_path, species_to_tensor, batch_size, device=device,
    atomic_properties=['forces'],
    transform=[energy_shifter.subtract_from_dataset])

###############################################################################
# When iterating the dataset, we will get pairs of input and output
# ``(species_coordinates, properties)``, in this case, ``properties`` would
# contain a key ``'atomic'`` where ``properties['atomic']`` is also a dict
# containing forces:

print(training[0]['atomic'])


###############################################################################
# Now let's define atomic neural networks.

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
# for different parameters. Since PyTorch does not have correct implementation
# of weight decay right now, we provide the correct implementation at TorchANI.
#
# .. note::
#
#   The weight decay in `inputtrain.ipt`_ is named "l2", but it is actually not
#   L2 regularization. The confusion between L2 and weight decay is a common
#   mistake in deep learning.  See: `Decoupled Weight Decay Regularization`_
#   Also note that the weight decay only applies to weight in the training
#   of ANI models, not bias.
#
# .. warning::
#
#   Currently TorchANI training with weight decay can not reproduce the training
#   result of NeuroChem with the same training setup. If you really want to use
#   weight decay, consider smaller rates and and make sure you do enough validation
#   to check if you get expected result.
#
# .. _Decoupled Weight Decay Regularization:
#   https://arxiv.org/abs/1711.05101
optimizer = torchani.optim.AdamW([
    # H networks
    {'params': [H_network[0].weight], 'weight_decay': 0.0001},
    {'params': [H_network[0].bias]},
    {'params': [H_network[2].weight], 'weight_decay': 0.00001},
    {'params': [H_network[2].bias]},
    {'params': [H_network[4].weight], 'weight_decay': 0.000001},
    {'params': [H_network[4].bias]},
    {'params': H_network[6].parameters()},
    # C networks
    {'params': [C_network[0].weight], 'weight_decay': 0.0001},
    {'params': [C_network[0].bias]},
    {'params': [C_network[2].weight], 'weight_decay': 0.00001},
    {'params': [C_network[2].bias]},
    {'params': [C_network[4].weight], 'weight_decay': 0.000001},
    {'params': [C_network[4].bias]},
    {'params': C_network[6].parameters()},
    # N networks
    {'params': [N_network[0].weight], 'weight_decay': 0.0001},
    {'params': [N_network[0].bias]},
    {'params': [N_network[2].weight], 'weight_decay': 0.00001},
    {'params': [N_network[2].bias]},
    {'params': [N_network[4].weight], 'weight_decay': 0.000001},
    {'params': [N_network[4].bias]},
    {'params': N_network[6].parameters()},
    # O networks
    {'params': [O_network[0].weight], 'weight_decay': 0.0001},
    {'params': [O_network[0].bias]},
    {'params': [O_network[2].weight], 'weight_decay': 0.00001},
    {'params': [O_network[2].bias]},
    {'params': [O_network[4].weight], 'weight_decay': 0.000001},
    {'params': [O_network[4].bias]},
    {'params': O_network[6].parameters()},
])

###############################################################################
# The way ANI trains a neural network potential looks like this:
#
# Phase 1: Pretrain the model by minimizing MSE loss
#
# Phase 2: Train the model by minimizing the exponential loss, until validation
# RMSE no longer improves for a certain steps, decay the learning rate and repeat
# the same process, stop until the learning rate is smaller than a certain number.
#
# We first read the checkpoint files to find where we are. We use `latest.pt`
# to store current training state. If `latest.pt` does not exist, this
# this means the pretraining has not been finished yet.
latest_checkpoint = 'latest.pt'
pretrained = os.path.isfile(latest_checkpoint)

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
        total_mse += mse_sum(predicted_energies, true_energies).item()
        count += predicted_energies.shape[0]
    return hartree2kcal(math.sqrt(total_mse / count))


###############################################################################
# If the model is not pretrained yet, we need to run the pretrain.
pretrain_criterion = 10  # kcal/mol
mse = torch.nn.MSELoss(reduction='none')

if not pretrained:
    print("pre-training...")
    epoch = 0
    rmse = math.inf
    pretrain_optimizer = torch.optim.Adam(nn.parameters())
    while rmse > pretrain_criterion:
        for batch_x, batch_y in tqdm.tqdm(training):
            true_energies = batch_y['energies']
            predicted_energies = []
            num_atoms = []
            for chunk_species, chunk_coordinates in batch_x:
                num_atoms.append((chunk_species >= 0).sum(dim=1))
                _, chunk_energies = model((chunk_species, chunk_coordinates))
                predicted_energies.append(chunk_energies)
            num_atoms = torch.cat(num_atoms).to(true_energies.dtype)
            predicted_energies = torch.cat(predicted_energies)
            loss = (mse(predicted_energies, true_energies) / num_atoms).mean()
            pretrain_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        rmse = validate()
        print('RMSE:', rmse, 'Target RMSE:', pretrain_criterion)
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
# Resume training from previously saved checkpoints:
checkpoint = torch.load(latest_checkpoint)
nn.load_state_dict(checkpoint['nn'])
optimizer.load_state_dict(checkpoint['optimizer'])
if 'scheduler' in checkpoint:
    scheduler.load_state_dict(checkpoint['scheduler'])


###############################################################################
# Finally, we come to the training loop.
#
# In this tutorial, we are setting the maximum epoch to a very small number,
# only to make this demo terminate fast. For serious training, this should be
# set to a much larger value
print("training starting from epoch", scheduler.last_epoch + 1)
max_epochs = 200
early_stopping_learning_rate = 1.0E-5
best_model_checkpoint = 'best.pt'

for _ in range(scheduler.last_epoch + 1, max_epochs):
    rmse = validate()
    learning_rate = optimizer.param_groups[0]['lr']

    if learning_rate < early_stopping_learning_rate:
        break

    tensorboard.add_scalar('validation_rmse', rmse, scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse', scheduler.best, scheduler.last_epoch)
    tensorboard.add_scalar('learning_rate', learning_rate, scheduler.last_epoch)

    # checkpoint
    if scheduler.is_better(rmse, scheduler.best):
        torch.save(nn.state_dict(), best_model_checkpoint)

    scheduler.step(rmse)

    for i, (batch_x, batch_y) in tqdm.tqdm(enumerate(training), total=len(training)):
        true_energies = batch_y['energies']
        predicted_energies = []
        num_atoms = []
        for chunk_species, chunk_coordinates in batch_x:
            num_atoms.append((chunk_species >= 0).sum(dim=1))
            _, chunk_energies = model((chunk_species, chunk_coordinates))
            predicted_energies.append(chunk_energies)
        num_atoms = torch.cat(num_atoms).to(true_energies.dtype)
        predicted_energies = torch.cat(predicted_energies)
        loss = (mse(predicted_energies, true_energies) / num_atoms).mean()
        loss = 0.5 * (torch.exp(2 * loss) - 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # write current batch loss to TensorBoard
        tensorboard.add_scalar('batch_loss', loss, scheduler.last_epoch * len(training) + i)

    torch.save({
        'nn': nn.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, latest_checkpoint)
