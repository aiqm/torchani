# -*- coding: utf-8 -*-
"""The ANI model zoo that stores public ANI models.

Currently the model zoo has two models: ANI-1x and ANI-1ccx. The classes
of these two models are :class:`ANI1x` and :class:`ANI1ccx`,
these are subclasses of :class:`torch.nn.Module`.
To use the models just instantiate them and either
directly calculate energies or get an ASE calculator. For example:

.. code:: python
    ani1x = torchani.models.ANI1x()
    # compute energy using ANI-1x model ensemble
    _, energies = ani1x((species, coordinates))
    ani1x.ase()  # get ASE Calculator using this ensemble
    # convert atom species from string to long tensor
    ani1x.species_to_tensor('CHHHH')

    model0 = ani1x[0]  # get the first model in the ensemble
    # compute energy using the first model in the ANI-1x model ensemble
    _, energies = model0((species, coordinates))
    model0.ase()  # get ASE Calculator using this model
    # convert atom species from string to long tensor
    model0.species_to_tensor('CHHHH')

Note that the class BuiltinModels can be accessed but it is deprecated and
shouldn't be used anymore.
"""

import torch
import warnings
from pkg_resources import resource_filename
from . import neurochem
from .aev import AEVComputer


# Future: Delete BuiltinModels in a future release, it is DEPRECATED
class BuiltinModels(torch.nn.Module):
    """BuiltinModels class.

    .. warning::
        This class is part of an old API. It is DEPRECATED and may be deleted in a
        future version. It shouldn't be used.
    """

    def __init__(self, builtin_class):
        warnings.warn(
            "BuiltinsModels is deprecated and will be deleted in"
            "the future; use torchani.models.BuiltinNet()", DeprecationWarning)
        super(BuiltinModels, self).__init__()
        self.builtins = builtin_class()
        self.aev_computer = self.builtins.aev_computer
        self.neural_networks = self.builtins.models
        self.energy_shifter = self.builtins.energy_shifter

    def forward(self, species_coordinates):
        species_aevs = self.aev_computer(species_coordinates)
        species_energies = self.neural_networks(species_aevs)
        return self.energy_shifter(species_energies)

    def __getitem__(self, index):
        ret = torch.nn.Sequential(self.aev_computer,
                                  self.neural_networks[index],
                                  self.energy_shifter)

        def ase(**kwargs):
            from . import ase
            return ase.Calculator(self.builtins.species, self.aev_computer,
                                  self.neural_networks[index],
                                  self.energy_shifter, **kwargs)

        ret.ase = ase
        ret.species_to_tensor = self.builtins.consts.species_to_tensor
        return ret

    def __len__(self):
        return len(self.neural_networks)

    def ase(self, **kwargs):
        """Get an ASE Calculator using this model"""
        from . import ase
        return ase.Calculator(self.builtins.species, self.aev_computer,
                              self.neural_networks, self.energy_shifter,
                              **kwargs)

    def species_to_tensor(self, *args, **kwargs):
        """Convert species from strings to tensor.

        See also :method:`torchani.neurochem.Constant.species_to_tensor`"""
        return self.builtins.consts.species_to_tensor(*args, **kwargs) \
            .to(self.aev_computer.ShfR.device)


class BuiltinNet(torch.nn.Module):
    """Private template for the builtin ANI ensemble models.

    All ANI ensemble models form the ANI models zoo should inherit from this class.
    This class is a torch module that sequentially calculates
    AEVs, then energies from a torchani.Ensemble and then uses EnergyShifter
    to shift those energies. It is essentially a sequential
    'AEVComputer -> Ensemble -> EnergyShifter'.

    .. note::
        This class is for internal use only, avoid using it, use ANI1x, ANI1ccx,
        etc instead. Don't confuse this class with torchani.Ensemble, which
        is only a container for many ANIModel instances and shouldn't be used
        directly for calculations.

    Attributes:
        const_file (:class:`str`): Path to the file with the builtin constants.
        sae_file (:class:`str`): Path to the file with the Self Atomic Energies.
        ensemble_prefix (:class:`str`): Prefix of directories.
        ensemble_size (:class:`int`): Number of models in the ensemble.
        energy_shifter (:class:`torchani.EnergyShifter`): Energy shifter with
            builtin Self Atomic Energies.
        aev_computer (:class:`torchani.AEVComputer`): AEV computer with
            builtin constants
        neural_networks (:class:`torchani.Ensemble`): Ensemble of ANIModel networks
    """

    def __init__(self, parent_name, const_file_path, sae_file_path,
                 ensemble_size, ensemble_prefix_path):
        super(BuiltinNet, self).__init__()

        self.const_file = resource_filename(parent_name, const_file_path)
        self.sae_file = resource_filename(parent_name, sae_file_path)
        self.ensemble_prefix = resource_filename(parent_name,
                                                 ensemble_prefix_path)

        self.ensemble_size = ensemble_size
        self.consts = neurochem.Constants(self.const_file)
        self.species = self.consts.species
        self.aev_computer = AEVComputer(**self.consts)
        self.energy_shifter = neurochem.load_sae(self.sae_file)
        self.neural_networks = neurochem.load_model_ensemble(
            self.species, self.ensemble_prefix, self.ensemble_size)

    def forward(self, species_coordinates):
        """Calculates predicted properties for minibatch of configurations

        Args:
            species_coordinates: minibatch of configurations

        Returns:
            species_energies: energies for the given configurations
        """
        species_aevs = self.aev_computer(species_coordinates)
        species_energies = self.neural_networks(species_aevs)
        return self.energy_shifter(species_energies)

    def __getitem__(self, index):
        """Get a single 'AEVComputer -> ANIModel -> EnergyShifter' sequential model

        Indexing allows access to a single model inside the ensemble
        that can be used directly for calculations. The model consists
        of a sequence AEVComputer -> ANIModel -> EnergyShifter
        and can return an ase calculator and convert species to tensor.

        Args:
            index (:class:`int`): Index of the model

        Returns:
            ret: (:class:`torch.nn.Sequential`): Sequential model ready for
                calculations
        """
        ret = torch.nn.Sequential(
            self.aev_computer,
            self.neural_networks[index],
            self.energy_shifter
        )

        def ase(**kwargs):
            """Attach an ase calculator """
            from . import ase
            return ase.Calculator(self.species,
                                  self.aev_computer,
                                  self.neural_networks[index],
                                  self.energy_shifter,
                                  **kwargs)

        ret.ase = ase
        ret.species_to_tensor = self.consts.species_to_tensor
        return ret

    def __len__(self):
        """Get the number of networks in the ensemble

        Returns:
            length (:class:`int`): Number of networks in the ensemble
        """
        return len(self.neural_networks)

    def ase(self, **kwargs):
        """Get an ASE Calculator using this ANI model ensemble

        Arguments:
            kwargs: ase.Calculator kwargs

        Returns:
            calculator (:class:`int`): A calculator to be used with ASE
        """
        from . import ase
        return ase.Calculator(self.species, self.aev_computer,
                              self.neural_networks, self.energy_shifter,
                              **kwargs)

    def species_to_tensor(self, *args, **kwargs):
        """Convert species from strings to tensor.

        See also :method:`torchani.neurochem.Constant.species_to_tensor`

        Arguments:
            species (:class:`str`): A string of chemical symbols

        Returns:
            tensor (:class:`torch.Tensor`): A 1D tensor of integers
        """
        return self.consts.species_to_tensor(*args, **kwargs) \
            .to(self.aev_computer.ShfR.device)


class ANI1x(BuiltinNet):
    """The ANI-1x model as in `ani-1x_8x on GitHub`_ and `Active Learning Paper`_.

    The ANI-1x model is an ensemble of 8 networks that was trained using
    active learning on the ANI-1x dataset, the target level of theory is
    wB97X/6-31G(d). It predicts energies on HCNO elements exclusively, it
    shouldn't be used with other atom types.

    .. _ani-1x_8x on GitHub:
        https://github.com/isayev/ASE_ANI/tree/master/ani_models/ani-1x_8x

    .. _Active Learning Paper:
        https://aip.scitation.org/doi/abs/10.1063/1.5023802
    """

    def __init__(self):
        super(ANI1x, self).__init__(
            parent_name='.'.join(__name__.split('.')[:-1]),
            const_file_path='resources/ani-1x_8x'
            '/rHCNO-5.2R_16-3.5A_a4-8.params',
            sae_file_path='resources/ani-1x_8x/sae_linfit.dat',
            ensemble_size=8,
            ensemble_prefix_path='resources/ani-1x_8x/train')


class ANI1ccx(BuiltinNet):
    """The ANI-1ccx model as in `ani-1ccx_8x on GitHub`_ and `Transfer Learning Paper`_.

    The ANI-1ccx model is an ensemble of 8 networks that was trained
    on the ANI-1ccx dataset, using transfer learning. The target accuracy
    is CCSD(T)*/CBS (CCSD(T) using the DPLNO-CCSD(T) method). It predicts
    energies on HCNO elements exclusively, it shouldn't be used with other
    atom types.

    .. _ani-1ccx_8x on GitHub:
        https://github.com/isayev/ASE_ANI/tree/master/ani_models/ani-1ccx_8x

    .. _Transfer Learning Paper:
        https://doi.org/10.26434/chemrxiv.6744440.v1
    """

    def __init__(self):
        super(ANI1ccx, self).__init__(
            parent_name='.'.join(__name__.split('.')[:-1]),
            const_file_path='resources/ani-1ccx_8x'
            '/rHCNO-5.2R_16-3.5A_a4-8.params',
            sae_file_path='resources/ani-1ccx_8x/sae_linfit.dat',
            ensemble_size=8,
            ensemble_prefix_path='resources/ani-1ccx_8x/train')
