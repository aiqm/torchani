from .torchaev import AEV
from .energyshifter import EnergyShifter
from .neurochem_aev import NeuroChemAEV

# TODO list
# 3. findout the correct permutation of Justin's code, and move my code to the same permutation
# 4. implement a subclass of `torch.nn.Module` to do neural network potential, allow loading from and saving to Justin's data format
# 5. do inference using Justin's pretrained model, make it work
# 6. implement a subclass of `torch.utils.data.Dataset` to allow taking subsets
# 7. implement a subclass of `torch.nn.Linear` to do parameter regularization and initialization the same way as in the paper
# 8. implement checkpoint & learning rate annealing

__all__ = [ 'AEV', 'EnergyShifter', 'NeuroChemAEV' ]