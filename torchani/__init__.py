import pkg_resources
import torch

buildin_const_file = pkg_resources.resource_filename(
    __name__, 'resources/ani-1x_dft_x8ens/rHCNO-5.2R_16-3.5A_a4-8.params')

buildin_sae_file = pkg_resources.resource_filename(
    __name__, 'resources/ani-1x_dft_x8ens/sae_linfit.dat')

buildin_network_dir = pkg_resources.resource_filename(
    __name__, 'resources/ani-1x_dft_x8ens/train0/networks/')

buildin_model_prefix = pkg_resources.resource_filename(
    __name__, 'resources/ani-1x_dft_x8ens/train')

default_dtype = torch.float32
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .energyshifter import EnergyShifter
from .nn import ModelOnAEV, PerSpeciesFromNeuroChem
from .aev import SortedAEV
import logging

__all__ = ['SortedAEV', 'EnergyShifter', 'ModelOnAEV', 'PerSpeciesFromNeuroChem', 'data',
           'buildin_const_file', 'buildin_sae_file', 'buildin_network_dir', 'buildin_dataset_dir',
           'default_dtype', 'default_device']