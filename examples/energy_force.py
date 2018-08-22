import torch
import os
import torchani

device = torch.device('cpu')
path = os.path.dirname(os.path.realpath(__file__))
const_file = os.path.join(path, '../torchani/resources/ani-1x_dft_x8ens/rHCNO-5.2R_16-3.5A_a4-8.params')  # noqa: E501
sae_file = os.path.join(path, '../torchani/resources/ani-1x_dft_x8ens/sae_linfit.dat')  # noqa: E501
network_dir = os.path.join(path, '../torchani/resources/ani-1x_dft_x8ens/train')  # noqa: E501
ensemble = 8

consts = torchani.neurochem.Constants(const_file)
aev_computer = torchani.AEVComputer(**consts)
nn = torchani.neurochem.load_model_ensemble(consts.species, network_dir,
                                            ensemble)
shift_energy = torchani.neurochem.load_sae(sae_file)
model = torch.nn.Sequential(aev_computer, nn, shift_energy)

coordinates = torch.tensor([[[0.03192167,  0.00638559,  0.01301679],
                             [-0.83140486,  0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308,  0.20759389],
                             [0.45554739,   0.54289633,  0.81170881],
                             [0.66091919,  -0.16799635, -0.91037834]]],
                           requires_grad=True)
species = consts.species_to_tensor('CHHHH', device).unsqueeze(0)

_, energy = model((species, coordinates))
derivative = torch.autograd.grad(energy.sum(), coordinates)[0]
force = -derivative

print('Energy:', energy.item())
print('Force:', force.squeeze().numpy())
