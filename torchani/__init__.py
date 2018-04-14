from .torchaev import AEV
from .energyshifter import EnergyShifter
from .neurochem_aev import NeuroChemAEV
from .nn import NeuralNetworkOnAEV

# TODO list
# 4. allow loading from and saving to Justin's data format
# 5. do inference using Justin's pretrained model, make it work
# 4.5 handle import error when pyNeuroChem is not present
# 6. implement a subclass of `torch.utils.data.Dataset` to allow taking subsets
# 7. implement a subclass of `torch.nn.Linear` to do parameter regularization and initialization the same way as in the paper
# 8. implement checkpoint & learning rate annealing

__all__ = [ 'AEV', 'EnergyShifter', 'NeuroChemAEV', 'NeuralNetworkOnAEV' ]