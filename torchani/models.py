# -*- coding: utf-8 -*-
"""The ANI model zoo that stores public ANI models.

Currently the model zoo has three models: ANI-1x, ANI-1ccx, and ANI-2x.
The parameters of these models are stored in `ani-model-zoo`_ repository and
will be automatically downloaded the first time any of these models are
instantiated. The classes of these models are :class:`ANI1x`, :class:`ANI1ccx`,
and :class:`ANI2x` these are subclasses of :class:`torch.nn.Module`.
To use the models just instantiate them and either
directly calculate energies or get an ASE calculator. For example:

.. _ani-model-zoo:
    https://github.com/aiqm/ani-model-zoo

.. code-block:: python

    ani1x = torchani.models.ANI1x()
    # compute energy using ANI-1x model ensemble
    _, energies = ani1x((species, coordinates))
    ani1x.ase()  # get ASE Calculator using this ensemble
    # convert atom species from string to long tensor
    ani1x.species_to_tensor(['C', 'H', 'H', 'H', 'H'])

    model0 = ani1x[0]  # get the first model in the ensemble
    # compute energy using the first model in the ANI-1x model ensemble
    _, energies = model0((species, coordinates))
    model0.ase()  # get ASE Calculator using this model
    # convert atom species from string to long tensor
    model0.species_to_tensor(['C', 'H', 'H', 'H', 'H'])
"""
import os
import io
import requests
import zipfile
import torch
from distutils import dir_util
from torch import Tensor
from typing import Tuple, Optional
from . import neurochem
from .nn import SpeciesConverter, SpeciesEnergies
from .aev import AEVComputer


class BuiltinModel(torch.nn.Module):
    r"""Private template for the builtin ANI models """

    def __init__(self, species_converter, aev_computer, neural_networks, energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index):
        super(BuiltinModel, self).__init__()
        self.species_converter = species_converter
        self.aev_computer = aev_computer
        self.neural_networks = neural_networks
        self.energy_shifter = energy_shifter
        self._species_to_tensor = species_to_tensor
        self.species = consts.species
        self.periodic_table_index = periodic_table_index

        # a bit useless maybe
        self.consts = consts
        self.sae_dict = sae_dict

    @classmethod
    def _from_neurochem_resources(cls, info_file_path, periodic_table_index=False, model_index=0):
        # this is used to load only 1 model (by default model 0)
        consts, sae_file, ensemble_prefix, ensemble_size = cls._parse_neurochem_resources(info_file_path)
        if (model_index >= ensemble_size):
            raise ValueError("The ensemble size is only {}, model {} can't be loaded".format(ensemble_size, model_index))

        species_converter = SpeciesConverter(consts.species)
        aev_computer = AEVComputer(**consts)
        energy_shifter, sae_dict = neurochem.load_sae(sae_file, return_dict=True)
        species_to_tensor = consts.species_to_tensor

        network_dir = os.path.join('{}{}'.format(ensemble_prefix, model_index), 'networks')
        neural_networks = neurochem.load_model(consts.species, network_dir)

        return cls(species_converter, aev_computer, neural_networks,
                   energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index)

    @staticmethod
    def _parse_neurochem_resources(info_file_path):
        def get_resource(resource_path, file_path):
            return os.path.join(resource_path, file_path)

        resource_path = os.path.join(os.path.dirname(__file__), 'resources/')
        local_dir = os.path.expanduser('~/.local/torchani/')
        repo_name = "ani-model-zoo"
        tag_name = "ani-2x"
        extracted_name = '{}-{}'.format(repo_name, tag_name)
        url = "https://github.com/aiqm/{}/archive/{}.zip".format(repo_name, tag_name)

        if os.stat(get_resource(resource_path, info_file_path)).st_size == 0:
            if not os.path.isfile(get_resource(local_dir, info_file_path)):
                print('Downloading ANI model parameters ...')
                resource_res = requests.get(url)
                resource_zip = zipfile.ZipFile(io.BytesIO(resource_res.content))
                try:
                    resource_zip.extractall(resource_path)
                except PermissionError:
                    resource_zip.extractall(local_dir)
                    resource_path = local_dir

                source = os.path.join(resource_path, extracted_name, "resources")
                dir_util.copy_tree(source, resource_path)
                dir_util.remove_tree(os.path.join(resource_path, extracted_name))

            else:
                resource_path = local_dir

        info_file = get_resource(resource_path, info_file_path)

        with open(info_file) as f:
            # const_file: Path to the file with the builtin constants.
            # sae_file: Path to the file with the Self Atomic Energies.
            # ensemble_prefix: Prefix of the neurochem resource directories.
            lines = [x.strip() for x in f.readlines()][:4]
            const_file_path, sae_file_path, ensemble_prefix_path, ensemble_size = lines
            const_file = get_resource(resource_path, const_file_path)
            sae_file = get_resource(resource_path, sae_file_path)
            ensemble_prefix = get_resource(resource_path, ensemble_prefix_path)
            ensemble_size = int(ensemble_size)
            consts = neurochem.Constants(const_file)
        return consts, sae_file, ensemble_prefix, ensemble_size

    def forward(self, species_coordinates: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        """Calculates predicted properties for minibatch of configurations

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled

        Returns:
            species_energies: energies for the given configurations

        .. note:: The coordinates, and cell are in Angstrom, and the energies
            will be in Hartree.
        """
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        species_energies = self.neural_networks(species_aevs)
        return self.energy_shifter(species_energies)

    @torch.jit.export
    def _recast_long_buffers(self):
        self.species_converter.conv_tensor = self.species_converter.conv_tensor.to(dtype=torch.long)
        self.aev_computer.triu_index = self.aev_computer.triu_index.to(dtype=torch.long)

    def species_to_tensor(self, *args, **kwargs):
        """Convert species from strings to tensor.

        See also :method:`torchani.neurochem.Constant.species_to_tensor`

        Arguments:
            species (:class:`str`): A string of chemical symbols

        Returns:
            tensor (:class:`torch.Tensor`): A 1D tensor of integers
        """
        # The only difference between this and the "raw" private version
        # _species_to_tensor is that this sends the final tensor to the model
        # device
        return self._species_to_tensor(*args, **kwargs) \
            .to(self.aev_computer.ShfR.device)

    def ase(self, **kwargs):
        """Get an ASE Calculator using this ANI model

        Arguments:
            kwargs: ase.Calculator kwargs

        Returns:
            calculator (:class:`int`): A calculator to be used with ASE
        """
        from . import ase
        return ase.Calculator(self.species, self, **kwargs)


class BuiltinEnsemble(BuiltinModel):
    """Private template for the builtin ANI ensemble models.

    ANI ensemble models form the ANI models zoo are instances of this class.
    This class is a torch module that sequentially calculates
    AEVs, then energies from a torchani.Ensemble and then uses EnergyShifter
    to shift those energies. It is essentially a sequential

    'AEVComputer -> Ensemble -> EnergyShifter'

    (periodic_table_index=False), or a sequential

    'SpeciesConverter -> AEVComputer -> Ensemble -> EnergyShifter'

    (periodic_table_index=True).

    .. note::
        This class is for internal use only, avoid relying on anything from it
        except the public methods, always use ANI1x, ANI1ccx, etc to instance
        the models.
        Also, don't confuse this class with torchani.Ensemble, which is only a
        container for many ANIModel instances and shouldn't be used directly
        for calculations.

    Attributes:
        species_converter (:class:`torchani.nn.SpeciesConverter`): Converts periodic table index to
            internal indices. Only present if periodic_table_index is `True`.
        aev_computer (:class:`torchani.AEVComputer`): AEV computer with
            builtin constants
        energy_shifter (:class:`torchani.EnergyShifter`): Energy shifter with
            builtin Self Atomic Energies.
        periodic_table_index (bool): Whether to use element number in periodic table
            to index species. If set to `False`, then indices must be `0, 1, 2, ..., N - 1`
            where `N` is the number of parametrized species.
    """

    def __init__(self, species_converter, aev_computer, neural_networks,
                 energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index):
        super(BuiltinEnsemble, self).__init__(species_converter,
                                              aev_computer,
                                              neural_networks,
                                              energy_shifter,
                                              species_to_tensor,
                                              consts,
                                              sae_dict,
                                              periodic_table_index)

    @classmethod
    def _from_neurochem_resources(cls, info_file_path, periodic_table_index=False):
        # this is used to load only 1 model (by default model 0)
        consts, sae_file, ensemble_prefix, ensemble_size = cls._parse_neurochem_resources(info_file_path)

        species_converter = SpeciesConverter(consts.species)
        aev_computer = AEVComputer(**consts)
        energy_shifter, sae_dict = neurochem.load_sae(sae_file, return_dict=True)
        species_to_tensor = consts.species_to_tensor
        neural_networks = neurochem.load_model_ensemble(consts.species,
                                                        ensemble_prefix, ensemble_size)

        return cls(species_converter, aev_computer, neural_networks,
                   energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index)

    def __getitem__(self, index):
        """Get a single 'AEVComputer -> ANIModel -> EnergyShifter' sequential model

        Get a single 'AEVComputer -> ANIModel -> EnergyShifter' sequential model
        or
        Indexing allows access to a single model inside the ensemble
        that can be used directly for calculations. The model consists
        of a sequence AEVComputer -> ANIModel -> EnergyShifter
        and can return an ase calculator and convert species to tensor.

        Args:
            index (:class:`int`): Index of the model

        Returns:
            ret: (:class:`torchani.models.BuiltinModel`) Model ready for
                calculations
        """
        ret = BuiltinModel(self.species_converter, self.aev_computer,
                           self.neural_networks[index], self.energy_shifter,
                           self._species_to_tensor, self.consts, self.sae_dict,
                           self.periodic_table_index)
        return ret

    def __len__(self):
        """Get the number of networks in the ensemble

        Returns:
            length (:class:`int`): Number of networks in the ensemble
        """
        return len(self.neural_networks)


def ANI1x(periodic_table_index=False, model_index=None):
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
    info_file = 'ani-1x_8x.info'
    if model_index is None:
        return BuiltinEnsemble._from_neurochem_resources(info_file, periodic_table_index)
    return BuiltinModel._from_neurochem_resources(info_file, periodic_table_index, model_index)


def ANI1ccx(periodic_table_index=False, model_index=None):
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
    info_file = 'ani-1ccx_8x.info'
    if model_index is None:
        return BuiltinEnsemble._from_neurochem_resources(info_file, periodic_table_index)
    return BuiltinModel._from_neurochem_resources(info_file, periodic_table_index, model_index)


def ANI2x(periodic_table_index=False, model_index=None):
    """The ANI-2x model as in `ANI2x Paper`_ and `ANI2x Results on GitHub`_.

    The ANI-2x model is an ensemble of 8 networks that was trained on the
    ANI-2x dataset. The target level of theory is wB97X/6-31G(d). It predicts
    energies on HCNOFSCl elements exclusively it shouldn't be used with other
    atom types.

    .. _ANI2x Results on GitHub:
        https://github.com/cdever01/ani-2x_results

    .. _ANI2x Paper:
        https://doi.org/10.26434/chemrxiv.11819268.v1
    """
    info_file = 'ani-2x_8x.info'
    if model_index is None:
        return BuiltinEnsemble._from_neurochem_resources(info_file, periodic_table_index)
    return BuiltinModel._from_neurochem_resources(info_file, periodic_table_index, model_index)
