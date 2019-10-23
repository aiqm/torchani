"""
.. _dipole-training-example:
Train Neural Network Potential To Both Energies and Dipoles
==========================================================
We have seen how to train a neural network potential by manually writing
training loop in :ref:`training-example`. This tutorial shows how to modify
that script to train to dipoles.
"""

#-*- coding: utf-8 -*-
import torch
import torchani
import os
import math
import tqdm

###############################################################################
# Most part of the script are the same as :ref:`training-example`, we will omit
# the comments for these parts. Please refer to :ref:`training-example` for more
# information


class ANIModelDipole(torch.nn.ModuleList):
    """ANI model that computes dipoles and energies from species and AEVs.
    Different atom types might have different modules, when computing
    properties, for each atom, the module for its corresponding atom type will
    be applied to its AEV, after that, outputs of modules will be reduced along
    different atoms to obtain molecular properties.
    Arguments:
        modules (:class:`collections.abc.Sequence`): Modules for each atom
            types. Atom types are distinguished by their order in
            :attr:`modules`, which means, for example ``modules[i]`` must be
            the module for atom type ``i``. Different atom types can share a
            module by putting the same reference in :attr:`modules`.
        aev_computer(:class:'torchani.AEVComputer'): Class for aev calculations.
            Species and coordinaates are passed to the aev_computer
            which returns the aevs used as input for the neural network.
        padding_fill (float): The value to fill output of padding atoms.
            Padding values will participate in reducing, so this value should
            be appropriately chosen so that it has no effect on the result. For
            example, if the reducer is :func:`torch.sum`, then
            :attr:`padding_fill` should be 0, and if the reducer is
            :func:`torch.min`, then :attr:`padding_fill` should be
            :obj:`math.inf`.
    """

    def __init__(self, modules, aev_computer, padding_fill=0):
        super(ANIModelDipole, self).__init__(modules)
        self.padding_fill = padding_fill
        self.aev_computer = aev_computer

    def get_atom_mask(self, species):
        padding_mask = (species.ne(-1)).float()
        assert padding_mask.sum() > 1.e-6
        padding_mask = padding_mask.unsqueeze(-1)
        return padding_mask

    def get_atom_neighbor_mask(self, atom_mask):
        atom_neighbor_mask = atom_mask.unsqueeze(1)*atom_mask.unsqueeze(2)
        assert atom_neighbor_mask.sum() > 1.e-6
        return atom_neighbor_mask

    def get_coulomb(self, charges, coordinates, species):
        dist=coordinates.unsqueeze(1) - coordinates.unsqueeze(2)
        #add 1e-6 to prevent sqrt(0) errors
        distances=torch.sqrt(torch.sum(dist**2,dim=-1)+1e-6).unsqueeze(-1)
        # Mask for padding atoms in distance matrix.
        distance_matrix_mask = self.get_atom_neighbor_mask(self.get_atom_mask(species))
        charges = charges.unsqueeze(2)
        charge_products = charges.unsqueeze(1)*charges.unsqueeze(2)
        coulomb = charge_products/distances
        coulomb = coulomb * distance_matrix_mask
        coulomb = coulomb.squeeze(-1)
        coulomb = torch.triu(coulomb, diagonal=1)
        coulomb = torch.sum(coulomb, dim=(1,2))
        return coulomb

    def get_dipole(self, xyz, charge):
        charge = charge.unsqueeze(1)
        xyz = xyz.permute(0,2,1)
        dipole = charge*xyz
        dipole = dipole.permute(0,2,1)
        dipole = torch.sum(dipole,dim=1)
        return dipole

    def forward(self, species_coordinates, total_charge=0):

        species, coordinates = species_coordinates
        species, aev = self.aev_computer(species_coordinates)
        species_ = species.flatten()
        present_species = torchani.utils.present_species(species)
        aev = aev.flatten(0, 1)
        output = torch.full_like(species_, self.padding_fill,
                                 dtype=aev.dtype)
        output_c = torch.full_like(species_, self.padding_fill,
                                 dtype=aev.dtype)
        for i in present_species:
            # Check that none of the weights are nan.
            for parameter in self[i].parameters():
                assert not (torch.isnan(parameter)).any()
            mask = (species_ == i)
            input_ = aev.index_select(0, mask.nonzero().squeeze())
            res = self[i](input_)
            output.masked_scatter_(mask, res[:,0].squeeze())
            output_c.masked_scatter_(mask, res[:,1].squeeze())
        output = output.view_as(species)
        output_c = output_c.view_as(species)

        #Maintain conservation of charge
        excess_charge = (torch.full_like(output_c[:,0],total_charge)-torch.sum(output_c,dim=1))/output_c.shape[1]
        excess_charge = excess_charge.unsqueeze(1)
        output_c+=excess_charge
        
        coulomb = self.get_coulomb(output_c, coordinates, species)

        output=torch.sum(output,dim=1)
        output+=coulomb
        dipole=self.get_dipole(coordinates, output_c)
        return species, output, dipole



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
energy_shifter = torchani.utils.EnergyShifter(None)
species_to_tensor = torchani.utils.ChemicalSymbolsToInts('HCNO')

try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()
dspath = os.path.join(path, '../datasets/mini_tz_dipoles.h5')


batch_size=1280

training, validation = torchani.data.load_ani_dataset(
    dspath,
    species_to_tensor,
    batch_size,
    device=device,
    properties=['energies', 'dipoles'],
    rm_outlier=True,
    transform=[energy_shifter.subtract_from_dataset],
    split=[0.8,None])



###############################################################################
# The neural network now outputs two values for each atom instead of one. 
# An atomic contribution to energy, and a partial atomic charge which will
# be used to determine the dipole moment of the molecule and coulombic 
# contribution to energy.

H_network = torch.nn.Sequential(
    torch.nn.Linear(384, 60),
    torch.nn.CELU(0.1),
    torch.nn.Linear(60, 40),
    torch.nn.CELU(0.1),
    torch.nn.Linear(40, 20),
    torch.nn.CELU(0.1),
    torch.nn.Linear(20, 2)
)

C_network = torch.nn.Sequential(
    torch.nn.Linear(384, 60),
    torch.nn.CELU(0.1),
    torch.nn.Linear(60, 40),
    torch.nn.CELU(0.1),
    torch.nn.Linear(40, 20),
    torch.nn.CELU(0.1),
    torch.nn.Linear(20, 2)
)

N_network = torch.nn.Sequential(
    torch.nn.Linear(384, 60),
    torch.nn.CELU(0.1),
    torch.nn.Linear(60, 40),
    torch.nn.CELU(0.1),
    torch.nn.Linear(40, 20),
    torch.nn.CELU(0.1),
    torch.nn.Linear(20, 2)
)

O_network = torch.nn.Sequential(
    torch.nn.Linear(384, 60),
    torch.nn.CELU(0.1),
    torch.nn.Linear(60, 40),
    torch.nn.CELU(0.1),
    torch.nn.Linear(40, 20),
    torch.nn.CELU(0.1),
    torch.nn.Linear(20, 2)
)


nn = ANIModelDipole([H_network, C_network, N_network, O_network], aev_computer)
print(nn)

###############################################################################
# Initialize the weights and biases.
#
# .. note::
#   Pytorch default initialization for the weights and biases in linear layers
#   is Kaiming uniform. See: `TORCH.NN.MODULES.LINEAR`_
#   We initialize the weights similarly but from the normal distribution.
#   The biases were initialized to zero.
#
# .. _TORCH.NN.MODULES.LINEAR:
#   https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear


def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)


nn.apply(init_params)


model = torch.nn.Sequential(nn).to(device)

###############################################################################
# Here we will use Adam with weight decay for the weights and Stochastic Gradient
# Descent for biases.

AdamW = torchani.optim.AdamW([
    # H networks
    {'params': [H_network[0].weight]},
    {'params': [H_network[2].weight], 'weight_decay': 0.00001},
    {'params': [H_network[4].weight], 'weight_decay': 0.000001},
    {'params': [H_network[6].weight]},
    # C networks
    {'params': [C_network[0].weight]},
    {'params': [C_network[2].weight], 'weight_decay': 0.00001},
    {'params': [C_network[4].weight], 'weight_decay': 0.000001},
    {'params': [C_network[6].weight]},
    # N networks
    {'params': [N_network[0].weight]},
    {'params': [N_network[2].weight], 'weight_decay': 0.00001},
    {'params': [N_network[4].weight], 'weight_decay': 0.000001},
    {'params': [N_network[6].weight]},
    # O networks
    {'params': [O_network[0].weight]},
    {'params': [O_network[2].weight], 'weight_decay': 0.00001},
    {'params': [O_network[4].weight], 'weight_decay': 0.000001},
    {'params': [O_network[6].weight]},
])

SGD = torch.optim.SGD([
    # H networks
    {'params': [H_network[0].bias]},
    {'params': [H_network[2].bias]},
    {'params': [H_network[4].bias]},
    {'params': [H_network[6].bias]},
    # C networks
    {'params': [C_network[0].bias]},
    {'params': [C_network[2].bias]},
    {'params': [C_network[4].bias]},
    {'params': [C_network[6].bias]},
    # N networks
    {'params': [N_network[0].bias]},
    {'params': [N_network[2].bias]},
    {'params': [N_network[4].bias]},
    {'params': [N_network[6].bias]},
    # O networks
    {'params': [O_network[0].bias]},
    {'params': [O_network[2].bias]},
    {'params': [O_network[4].bias]},
    {'params': [O_network[6].bias]},
], lr=1e-3)

AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0)
SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.5, patience=100, threshold=0)

###############################################################################
# This part of the code is also the same
latest_checkpoint = 'dipole-training-latest.pt'

###############################################################################
# Resume training from previously saved checkpoints:
if os.path.isfile(latest_checkpoint):
    checkpoint = torch.load(latest_checkpoint)
    nn.load_state_dict(checkpoint['nn'])
    AdamW.load_state_dict(checkpoint['AdamW'])
    SGD.load_state_dict(checkpoint['SGD'])
    AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
    SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])

###############################################################################
# During training, we need to validate on validation set and if validation error
# is better than the best, then save the new best model to a checkpoint

# helper function to convert energy unit from Hartree to kcal/mol
def hartree2kcal(x):
    return 627.509 * x


def validate():
    # run validation
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_energy_mse = 0.0
    total_dipole_mse = 0.0
    count = 0
    for batch_x, batch_y in validation:
        true_energies = batch_y['energies']
        true_dipoles = batch_y['dipoles']
        predicted_energies = []
        predicted_dipoles = []
        for chunk_species, chunk_coordinates in batch_x:
            s, chunk_energies, chunk_dipoles = model((chunk_species, chunk_coordinates))
            predicted_energies.append(chunk_energies)
            predicted_dipoles.append(chunk_dipoles)
        predicted_energies = torch.cat(predicted_energies)
        predicted_dipoles = torch.cat(predicted_dipoles)
        total_dipole_mse += mse_sum(predicted_dipoles, true_dipoles).item()
        total_energy_mse += mse_sum(predicted_energies, true_energies).item()
        count += predicted_energies.shape[0]
    return hartree2kcal(math.sqrt(total_energy_mse / count)), math.sqrt(total_dipole_mse/count)

###############################################################################
# In the training loop, we need to compute dipoles, and loss for dipoles
mse = torch.nn.MSELoss(reduction='none')

print("training starting from epoch", AdamW_scheduler.last_epoch + 1)
# We only train 3 epoches here in able to generate the docs quickly.
# Real training should take much more than 3 epoches.
max_epochs = 3
early_stopping_learning_rate = 1.0E-5
best_model_checkpoint = 'dipole-training-best.pt'

dipole_coefficient = 1.0
energy_coefficient = 1.0

for _ in range(AdamW_scheduler.last_epoch + 1, max_epochs):
    energy_rmse, dipole_rmse = validate()
    total_rmse = energy_rmse+dipole_rmse
    print('Epoch:', AdamW_scheduler.last_epoch+1)
    print('Energy RMSE:', energy_rmse)
    print('Dipole RMSE:', dipole_rmse)
    print('Total RMSE:', total_rmse)

    learning_rate = AdamW.param_groups[0]['lr']
    
    if learning_rate < early_stopping_learning_rate:
        break

    # checkpoint
    if AdamW_scheduler.is_better(total_rmse, AdamW_scheduler.best):
        torch.save(nn.state_dict(), best_model_checkpoint)

    AdamW_scheduler.step(total_rmse)
    SGD_scheduler.step(total_rmse)

    for i, (batch_x, batch_y) in tqdm.tqdm(enumerate(training), total=len(training)):
        true_energies = batch_y['energies']
        true_dipoles = batch_y['dipoles']
        predicted_energies = []
        predicted_dipoles = []
        num_atoms = []
        force_loss = []
        for chunk in batch_y['atomic']:
            chunk_species = chunk['species']
            chunk_coordinates = chunk['coordinates']
            chunk_num_atoms = (chunk_species >=0).sum(dim=1).to(true_energies.dtype)
            num_atoms.append(chunk_num_atoms)
            chunk_coordinates.requires_grad_(True)
            s, chunk_energies, chunk_dipoles = model((chunk_species, chunk_coordinates))
            predicted_energies.append(chunk_energies)
            predicted_dipoles.append(chunk_dipoles)
            
        num_atoms = torch.cat(num_atoms).to(true_energies.dtype)
        predicted_energies = torch.cat(predicted_energies)
        predicted_dipoles = torch.cat(predicted_dipoles)

        # Now the total loss has two parts, energy loss and dipole loss
        loss=0
        energy_loss = (mse(predicted_energies, true_energies) / num_atoms).mean()
        loss += energy_coefficient * energy_loss
        dipole_loss = (torch.sum(torch.abs(predicted_dipoles-true_dipoles),dim=1) / num_atoms).mean()
        loss += dipole_coefficient * dipole_loss

        AdamW.zero_grad()
        SGD.zero_grad()
        loss.backward()
        AdamW.step()
        SGD.step()

    torch.save({
        'nn': nn.state_dict(),
        'AdamW': AdamW.state_dict(),
        'SGD': SGD.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
        'SGD_scheduler': SGD_scheduler.state_dict(),
    }, latest_checkpoint)
