# -*- coding: utf-8 -*-
"""
.. _force-training-example:

Train Neural Network Potential To Both Energies and Forces
==========================================================

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
# contain a key ``'atomic'`` where ``properties['atomic']`` is a list of dict
# containing forces:

data = training[0]
properties = data[1]
atomic_properties = properties['atomic']
print(type(atomic_properties))
print(list(atomic_properties[0].keys()))

###############################################################################
# Due to padding, part of the forces might be 0
print(atomic_properties[0]['forces'][0])


###############################################################################
# The code to define networks, optimizers, are mostly the same

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
model = torch.nn.Sequential(aev_computer, nn).to(device)

###############################################################################
# Here we will turn off weight decay
optimizer = torch.optim.Adam(nn.parameters())

###############################################################################
# This part of the code is also the same
latest_checkpoint = 'force-training-latest.pt'
pretrained = os.path.isfile(latest_checkpoint)


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


pretrain_criterion = 10  # kcal/mol
mse = torch.nn.MSELoss(reduction='none')

###############################################################################
# For simplicity, we don't train to force during pretraining
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

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100)
tensorboard = torch.utils.tensorboard.SummaryWriter()

checkpoint = torch.load(latest_checkpoint)
nn.load_state_dict(checkpoint['nn'])
optimizer.load_state_dict(checkpoint['optimizer'])
if 'scheduler' in checkpoint:
    scheduler.load_state_dict(checkpoint['scheduler'])


###############################################################################
# In the training loop, we need to compute force, and loss for forces
print("training starting from epoch", scheduler.last_epoch + 1)
max_epochs = 200
early_stopping_learning_rate = 1.0E-5
force_coefficient = 1  # controls the importance of energy loss vs force loss
best_model_checkpoint = 'force-training-best.pt'

for _ in range(scheduler.last_epoch + 1, max_epochs):
    rmse = validate()
    print('RMSE:', rmse, 'at epoch', scheduler.last_epoch)

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

    # Besides being stored in x, species and coordinates are also stored in y.
    # So here, for simplicity, we just ignore the x and use y for everything.
    for i, (_, batch_y) in tqdm.tqdm(enumerate(training), total=len(training)):
        true_energies = batch_y['energies']
        predicted_energies = []
        num_atoms = []
        force_loss = []

        for chunk in batch_y['atomic']:
            chunk_species = chunk['species']
            chunk_coordinates = chunk['coordinates']
            chunk_true_forces = chunk['forces']
            chunk_num_atoms = (chunk_species >= 0).sum(dim=1).to(true_energies.dtype)
            num_atoms.append(chunk_num_atoms)

            # We must set `chunk_coordinates` to make it requires grad, so
            # that we could compute force from it
            chunk_coordinates.requires_grad_(True)

            _, chunk_energies = model((chunk_species, chunk_coordinates))

            # We can use torch.autograd.grad to compute force. Remember to
            # create graph so that the loss of the force can contribute to
            # the gradient of parameters, and also to retain graph so that
            # we can backward through it a second time when computing gradient
            # w.r.t. parameters.
            chunk_forces = -torch.autograd.grad(chunk_energies.sum(), chunk_coordinates, create_graph=True, retain_graph=True)[0]

            # Now let's compute loss for force of this chunk
            chunk_force_loss = mse(chunk_true_forces, chunk_forces).sum(dim=(1, 2)) / chunk_num_atoms

            predicted_energies.append(chunk_energies)
            force_loss.append(chunk_force_loss)

        num_atoms = torch.cat(num_atoms)
        predicted_energies = torch.cat(predicted_energies)

        # Now the total loss has two parts, energy loss and force loss
        energy_loss = (mse(predicted_energies, true_energies) / num_atoms).mean()
        energy_loss = 0.5 * (torch.exp(2 * energy_loss) - 1)
        force_loss = torch.cat(force_loss).mean()
        loss = energy_loss + force_coefficient * force_loss

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
