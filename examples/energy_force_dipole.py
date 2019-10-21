# -*- coding: utf-8 -*-
"""
Computing Energy, Force, And Dipole Using Model
"""

###############################################################################
# To begin with, let's first import the modules we will use:
from __future__ import print_function
import torch
import torchani

###############################################################################
# Let's now manually specify the device we want TorchANI to run:
device = torch.device('cpu')

#Next we define the ANI Model we will use

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

    def __init__(self, modules, aev_computer, reducer=torch.sum, padding_fill=0):
        super(ANIModelDipole, self).__init__(modules)
        self.reducer = reducer
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

        output=self.reducer(output,dim=1)
        output+=coulomb
        dipole=self.get_dipole(coordinates, output_c)
        return species, output, dipole






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
species_to_tensor = torchani.utils.ChemicalSymbolsToInts('HCNO')

#Define the values for the energy shifter
energy_shifter = torchani.utils.EnergyShifter([
    -0.600952980000,  # H
    -38.08316124000,  # C
    -54.70775770000,  # N
    -75.19446356000,  # O
])

###############################################################################

#Define the network architecture. MAke sure this is the same as the network that will be loaded.
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


species_to_tensor = torchani.utils.ChemicalSymbolsToInts('HCNO')
latest_checkpoint = 'latest.pt'
checkpoint = torch.load(latest_checkpoint)
nn.load_state_dict(checkpoint['nn'])


model = torch.nn.Sequential(nn).to(device)

# Now let's define the coordinate and species. If you just want to compute the
# energy, force, and dipole for a single structure like in this example, you need to
# make the coordinate tensor has shape ``(1, Na, 3)`` and species has shape
# ``(1, Na)``, where ``Na`` is the number of atoms in the molecule, the
# preceding ``1`` in the shape is here to support batch processing like in
# training. If you have ``N`` different structures to compute, then make it
# ``N``.

coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                             [-0.83140486, 0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308, 0.20759389],
                             [0.45554739, 0.54289633, 0.81170881],
                             [0.66091919, -0.16799635, -0.91037834]]],
                           requires_grad=True, device=device)
species = species_to_tensor('CHHHH').unsqueeze(0).to(device)


###############################################################################
# Now let's compute energy, force:, and dipole
_, energy, dipole = model((species, coordinates))
_, energy = energy_shifter((species, energy))
derivative = torch.autograd.grad(energy.sum(), coordinates)[0]
force = -derivative

###############################################################################
# And print to see the result:
print('Energy:', energy.item())
print('Dipole', dipole)
print('Force:', force.squeeze())
