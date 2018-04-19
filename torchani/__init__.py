import pkg_resources

buildin_const_file = pkg_resources.resource_filename(
    __name__, 'data/rHCNO-4.6R_16-3.1A_a4-8_3.params')

buildin_sae_file = pkg_resources.resource_filename(
    __name__, 'data/sae_linfit.dat')

buildin_network_dir = pkg_resources.resource_filename(
    __name__, 'data/networks/')

from .torchaev import AEV
from .energyshifter import EnergyShifter
from .nn import NeuralNetworkOnAEV
from .neighboraev import NeighborAEV
import logging

__all__ = ['AEV', 'NeighborAEV', 'EnergyShifter', 'NeuralNetworkOnAEV',
           'buildin_const_file', 'buildin_sae_file', 'buildin_network_dir']

try:
    from .neurochem_aev import NeuroChemAEV
    __all__.append('NeuroChemAEV')
except ImportError:
    logging.log(logging.WARNING,
                'Unable to import NeuroChemAEV, please check your pyNeuroChem installation.')
