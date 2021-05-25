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
from copy import deepcopy
from pathlib import Path
from collections import OrderedDict
import torch
from torch import Tensor
from torch.nn import Module
from typing import Tuple, Optional, NamedTuple, Sequence, Union, Type
from .nn import SpeciesConverter, SpeciesEnergies, Ensemble, ANIModel
from .utils import ChemicalSymbolsToInts, PERIODIC_TABLE, EnergyShifter, path_is_writable
from .aev import AEVComputer
from .compat import Final
from . import atomics


class SpeciesEnergiesQBC(NamedTuple):
    species: Tensor
    energies: Tensor
    qbcs: Tensor


class BuiltinModel(Module):
    r"""Private template for the builtin ANI models """

    atomic_numbers: Tensor
    periodic_table_index: Final[bool]

    def __init__(self,
                 aev_computer: AEVComputer,
                 neural_networks,
                 energy_shifter,
                 elements: Sequence[str],
                 periodic_table_index: bool = False):

        super().__init__()

        self.aev_computer = aev_computer
        self.neural_networks = neural_networks
        self.energy_shifter = energy_shifter
        self.species_to_tensor = ChemicalSymbolsToInts(elements)
        self.species_converter = SpeciesConverter(elements)

        self.periodic_table_index = periodic_table_index
        numbers = torch.tensor([PERIODIC_TABLE.index(e) for e in elements], dtype=torch.long)
        self.register_buffer('atomic_numbers', numbers)

        # checks are performed to make sure all modules passed support the
        # correct number of species
        if energy_shifter.fit_intercept:
            assert len(energy_shifter.self_energies) == len(self.atomic_numbers) + 1
        else:
            assert len(energy_shifter.self_energies) == len(self.atomic_numbers)

        assert len(self.atomic_numbers) == self.aev_computer.num_species

        if isinstance(self.neural_networks, Ensemble):
            for nnp in self.neural_networks:
                assert len(nnp) == len(self.atomic_numbers)
        else:
            assert len(self.neural_networks) == len(self.atomic_numbers)

    @torch.jit.unused
    def get_chemical_symbols(self) -> Tuple[str, ...]:
        return tuple(PERIODIC_TABLE[z] for z in self.atomic_numbers)

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

        # check if unknown species are included
        if species_coordinates[0].ge(self.aev_computer.num_species).any():
            raise ValueError(f'Unknown species found in {species_coordinates[0]}')

        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        species_energies = self.neural_networks(species_aevs)
        return self.energy_shifter(species_energies)

    @torch.jit.export
    def atomic_energies(self, species_coordinates: Tuple[Tensor, Tensor],
                        cell: Optional[Tensor] = None,
                        pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        """Calculates predicted atomic energies of all atoms in a molecule

        ..warning::
            Since this function does not call ``__call__`` directly,
            hooks are not registered and profiling is not done correctly by
            pytorch on it. It is meant as a convenience function for analysis
             and active learning.

        .. note:: The coordinates, and cell are in Angstrom, and the energies
            will be in Hartree.

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled

        Returns:
            species_atomic_energies: species and energies for the given configurations
                note that the shape of species is (C, A), where C is
                the number of configurations and A the number of atoms, and
                the shape of energies is (C, A) for a BuiltinModel.
        """
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        species, aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        atomic_energies = self.neural_networks._atomic_energies((species, aevs))
        self_energies = self.energy_shifter.self_energies.clone().to(species.device)
        self_energies = self_energies[species]
        self_energies[species == torch.tensor(-1, device=species.device)] = torch.tensor(0, device=species.device, dtype=torch.double)
        # shift all atomic energies individually
        assert self_energies.shape == atomic_energies.shape
        atomic_energies += self_energies
        return SpeciesEnergies(species, atomic_energies)

    @torch.jit.export
    def _recast_long_buffers(self):
        self.species_converter.conv_tensor = self.species_converter.conv_tensor.to(dtype=torch.long)
        self.aev_computer.triu_index = self.aev_computer.triu_index.to(dtype=torch.long)

    def ase(self, **kwargs):
        """Get an ASE Calculator using this ANI model

        Arguments:
            kwargs: ase.Calculator kwargs

        Returns:
            calculator (:class:`int`): A calculator to be used with ASE
        """
        from . import ase
        return ase.Calculator(self.get_chemical_symbols(), self, **kwargs)


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.jit.export
    def atomic_energies(self, species_coordinates: Tuple[Tensor, Tensor],
                        cell: Optional[Tensor] = None,
                        pbc: Optional[Tensor] = None, average: bool = True) -> SpeciesEnergies:
        """Calculates predicted atomic energies of all atoms in a molecule

        see `:method:torchani.BuiltinModel.atomic_energies`

        If average is True (the default) it returns the average over all models
        (shape (C, A)), otherwise it returns one atomic energy per model (shape
        (M, C, A))
        """
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        species, aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        members_list = []
        for nnp in self.neural_networks:
            members_list.append(nnp._atomic_energies((species, aevs)).unsqueeze(0))
        member_atomic_energies = torch.cat(members_list, dim=0)

        self_energies = self.energy_shifter.self_energies.clone().to(species.device)
        self_energies = self_energies[species]
        self_energies[species == torch.tensor(-1, device=species.device)] = torch.tensor(0, device=species.device, dtype=torch.double)
        # shift all atomic energies individually
        assert self_energies.shape == member_atomic_energies.shape[1:]
        member_atomic_energies += self_energies
        if average:
            return SpeciesEnergies(species, member_atomic_energies.mean(dim=0))
        return SpeciesEnergies(species, member_atomic_energies)

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
        ret = BuiltinModel(self.aev_computer,
                           self.neural_networks[index],
                           self.energy_shifter,
                           self.get_chemical_symbols(),
                           self.periodic_table_index)
        return ret

    @torch.jit.export
    def members_energies(self, species_coordinates: Tuple[Tensor, Tensor],
                         cell: Optional[Tensor] = None,
                         pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        """Calculates predicted energies of all member modules

        ..warning::
            Since this function does not call ``__call__`` directly,
            hooks are not registered and profiling is not done correctly by
            pytorch on it. It is meant as a convenience function for analysis
             and active learning.

        .. note:: The coordinates, and cell are in Angstrom, and the energies
            will be in Hartree.

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled

        Returns:
            species_energies: species and energies for the given configurations
                note that the shape of species is (C, A), where C is
                the number of configurations and A the number of atoms, and
                the shape of energies is (M, C), where M is the number
                of modules in the ensemble

        """
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        species, aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        member_outputs = []
        for nnp in self.neural_networks:
            unshifted_energies = nnp((species, aevs)).energies
            shifted_energies = self.energy_shifter((species, unshifted_energies)).energies
            member_outputs.append(shifted_energies.unsqueeze(0))
        return SpeciesEnergies(species, torch.cat(member_outputs, dim=0))

    @torch.jit.export
    def energies_qbcs(self, species_coordinates: Tuple[Tensor, Tensor],
                      cell: Optional[Tensor] = None,
                      pbc: Optional[Tensor] = None, unbiased: bool = True) -> SpeciesEnergiesQBC:
        """Calculates predicted predicted energies and qbc factors

        QBC factors are used for query-by-committee (QBC) based active learning
        (as described in the ANI-1x paper `less-is-more`_ ).

        .. _less-is-more:
            https://aip.scitation.org/doi/10.1063/1.5023802

        ..warning::
            Since this function does not call ``__call__`` directly,
            hooks are not registered and profiling is not done correctly by
            pytorch on it. It is meant as a convenience function for analysis
             and active learning.

        .. note:: The coordinates, and cell are in Angstrom, and the energies
            and qbc factors will be in Hartree.

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not
                enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set
                to None if PBC is not enabled
            unbiased: if `True` then Bessel's correction is applied to the
                standard deviation over the ensemble member's. If `False` Bessel's
                correction is not applied, True by default.

        Returns:
            species_energies_qbcs: species, energies and qbc factors for the
                given configurations note that the shape of species is (C, A),
                where C is the number of configurations and A the number of
                atoms, the shape of energies is (C,) and the shape of qbc
                factors is also (C,).
        """
        species, energies = self.members_energies(species_coordinates, cell, pbc)

        # standard deviation is taken across ensemble members
        qbc_factors = energies.std(0, unbiased=unbiased)

        # rho's (qbc factors) are weighted by dividing by the square root of
        # the number of atoms in each molecule
        num_atoms = (species >= 0).sum(dim=1, dtype=energies.dtype)
        qbc_factors = qbc_factors / num_atoms.sqrt()
        energies = energies.mean(dim=0)
        assert qbc_factors.shape == energies.shape
        return SpeciesEnergiesQBC(species, energies, qbc_factors)

    def __len__(self):
        """Get the number of networks in the ensemble

        Returns:
            length (:class:`int`): Number of networks in the ensemble
        """
        return len(self.neural_networks)


def _get_component_modules(state_dict_file: str,
                           model_index: Optional[int] = None,
                           use_cuda_extension: bool = False,
                           ensemble_size: int = 8) -> Tuple[AEVComputer, Module, EnergyShifter, Sequence[str]]:
    # This generates ani-style architectures without neurochem
    name = state_dict_file.split('_')[0]
    elements: Tuple[str, ...]
    if name == 'ani1x':
        aev_maker = AEVComputer.like_1x
        atomic_maker = atomics.like_1x
        elements = ('H', 'C', 'N', 'O')
    elif name == 'ani1ccx':
        aev_maker = AEVComputer.like_1ccx
        atomic_maker = atomics.like_1ccx
        elements = ('H', 'C', 'N', 'O')
    elif name == 'ani2x':
        aev_maker = AEVComputer.like_2x
        atomic_maker = atomics.like_2x
        elements = ('H', 'C', 'N', 'O', 'S', 'F', 'Cl')
    else:
        raise ValueError(f'{name} is not a supported model')
    aev_computer = aev_maker(use_cuda_extension=use_cuda_extension)
    atomic_networks = OrderedDict([(e, atomic_maker(e)) for e in elements])

    neural_networks: Union[ANIModel, Ensemble]
    if model_index is None:
        neural_networks = Ensemble([ANIModel(deepcopy(atomic_networks)) for _ in range(ensemble_size)])
    else:
        neural_networks = ANIModel(atomic_networks)
    return aev_computer, neural_networks, EnergyShifter([0.0 for _ in elements]), elements


def _fetch_state_dict(state_dict_file: str,
                      model_index: Optional[int] = None,
                      local: bool = False) -> 'OrderedDict[str, Tensor]':
    # if we want a pretrained model then we load the state dict from a
    # remote url or a local path
    # NOTE: torch.hub caches remote state_dicts after they have been downloaded
    if local:
        return torch.load(state_dict_file)

    model_dir = Path(__file__).parent.joinpath('resources/state_dicts').as_posix()
    if not path_is_writable(model_dir):
        model_dir = os.path.expanduser('~/.local/torchani/')

    # NOTE: we need some private url for in-development models of the
    # group, this url is for public models
    tag = 'v0.1'
    url = f'https://github.com/roitberg-group/torchani_model_zoo/releases/download/{tag}/{state_dict_file}'
    # for now for simplicity we load a state dict for the ensemble directly and
    # then parse if needed
    state_dict = torch.hub.load_state_dict_from_url(url, model_dir=model_dir)

    if model_index is not None:
        new_state_dict = OrderedDict()
        # Parse the state dict and rename/select only useful keys to build
        # the individual model
        for k, v in state_dict.items():
            tkns = k.split('.')
            if tkns[0] == 'neural_networks':
                # rename or discard the key
                if int(tkns[1]) == model_index:
                    tkns.pop(1)
                    k = '.'.join(tkns)
                else:
                    continue
            new_state_dict[k] = v
        state_dict = new_state_dict

    return state_dict


def _load_ani_model(state_dict_file: Optional[str] = None,
                    info_file: Optional[str] = None,
                    **model_kwargs) -> Union[BuiltinModel, BuiltinEnsemble]:
    # Helper function to toggle if the loading is done from an NC file or
    # directly using torchani and state_dicts
    use_neurochem_source = model_kwargs.pop('use_neurochem_source', False)
    use_cuda_extension = model_kwargs.pop('use_cuda_extension', False)
    model_index = model_kwargs.pop('model_index', None)
    pretrained = model_kwargs.pop('pretrained', True)

    if use_neurochem_source:
        assert info_file is not None, "Info file is needed to load from a neurochem source"
        assert pretrained, "Non pretrained models not available from neurochem source"
        from . import neurochem  # noqa
        components = neurochem.parse_resources._get_component_modules(info_file, model_index, use_cuda_extension)
    else:
        assert state_dict_file is not None
        components = _get_component_modules(state_dict_file, model_index, use_cuda_extension)

    aev_computer, neural_networks, energy_shifter, elements = components

    model_class: Union[Type[BuiltinEnsemble], Type[BuiltinModel]]
    if model_index is None:
        model_class = BuiltinEnsemble
    else:
        model_class = BuiltinModel

    model = model_class(aev_computer, neural_networks, energy_shifter, elements, **model_kwargs)

    if pretrained and not use_neurochem_source:
        assert state_dict_file is not None
        model.load_state_dict(_fetch_state_dict(state_dict_file, model_index))
    return model


def ANI1x(**kwargs):
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
    state_dict_file = 'ani1x_state_dict.pt'
    return _load_ani_model(state_dict_file, info_file, **kwargs)


def ANI1ccx(**kwargs):
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
    state_dict_file = 'ani1ccx_state_dict.pt'
    return _load_ani_model(state_dict_file, info_file, **kwargs)


def ANI2x(**kwargs):
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
    state_dict_file = 'ani2x_state_dict.pt'
    return _load_ani_model(state_dict_file, info_file, **kwargs)
