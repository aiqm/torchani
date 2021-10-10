# -*- coding: utf-8 -*-
"""The ANI model zoo that stores public ANI models.

Currently the model zoo has three models: ANI-1x, ANI-1ccx, and ANI-2x.  The
parameters of these models are stored in `ani-model-zoo`_ repository and will
be automatically downloaded the first time any of these models are
instantiated.

To use the models just instantiate them and either directly calculate energies
by calling them, or cast them to an ASE calculator.

If the models have many ensembled neural networks they can be indexed to access
individual members of the ensemble, and len() can be used to get the number of
networks in the ensemble

The models also have three extra entry points for more specific use cases:
members_energies, atomic_energies and energies_qbcs.

All entrypoints expect a tuple of tensors `(species, coordinates)` as input,
together with two optional tensors, `cell` and `pbc`.
`coordinates` and `cell` should be in units of Angstroms,
and the output energies are always in Hartrees

example usage:

.. _ani-model-zoo:
    https://github.com/aiqm/ani-model-zoo

.. code-block:: python

    ani1x = torchani.models.ANI1x()

    # compute energy using ANI-1x model ensemble
    _, energies = ani1x((species, coordinates))

    # get an ASE Calculator using this ensemble
    ani1x_calc = ani1x.ase()

    # convert atom species from string to long tensor
    ani1x.species_to_tensor(['C', 'H', 'H', 'H', 'H'])

    # output shape of energies is (M, C), where M is the number of ensemble members
    _, members_energies = ani1x.members_energies((species, coordinates))

    # output shape of energies is (A, C) where A is the number of atoms in the minibatch
    # atomic energies are averaged over all models by default
    _, atomic_energies = ani1x.atomic_energies((species, coordinates))

    # qbc factors are used for active learning, shape of qbc factors is equal to energies
    _, energies, qbcs = ani1x.energies_qbcs((species, coordinates))

    # individual models of the ensemble can be obtained by indexing,
    # and they have the same functionality as the ensembled model
    model0 = ani1x[0]
    _, energies = model0((species, coordinates))
"""
import os
from copy import deepcopy
from pathlib import Path
from collections import OrderedDict
import torch
from torch import Tensor
from torch.nn import Module
from typing import Tuple, Optional, NamedTuple, Sequence, Union, Dict, Any
from .nn import SpeciesConverter, SpeciesEnergies, Ensemble, ANIModel
from .utils import ChemicalSymbolsToInts, PERIODIC_TABLE, EnergyShifter, path_is_writable
from .aev import AEVComputer
from .compat import Final
from . import atomics


NN = Union[ANIModel, Ensemble]


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
                 neural_networks: NN,
                 energy_shifter: EnergyShifter,
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
        self._register_types_for_jit()

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

    def _register_types_for_jit(self):
        # Register dummy modules so that JIT knows their types and can
        # perform isinstance checks correctly
        self._dummy_model = ANIModel([torch.nn.Sequential(torch.nn.Identity())])
        self._dummy_ensemble = Ensemble([self._dummy_model])

    def to_infer_model(self, *args, **kwargs) -> 'BuiltinModel':
        """ Convert the neural networks module of the model into a module
            optimized for inference.

            Currently this function assumes that the atomic networks consist of
            an MLP with CELU activation functions, all with the same alpha.
        """
        self.neural_networks = self.neural_networks.to_infer_model(*args, **kwargs)
        return self

    @torch.jit.unused
    def get_chemical_symbols(self) -> Tuple[str, ...]:
        return tuple(PERIODIC_TABLE[z] for z in self.atomic_numbers)

    def forward(self, species_coordinates: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        """Calculates predicted energies for minibatch of configurations

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled

        Returns:
            species_energies: tuple of tensors, species and energies for the given configurations
        """
        species_coordinates = self._maybe_convert_species(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        species_energies = self.neural_networks(species_aevs)
        return self.energy_shifter(species_energies)

    @torch.jit.export
    def _maybe_convert_species(self, species_coordinates: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        if (species_coordinates[0] >= self.aev_computer.num_species).any():
            raise ValueError(f'Unknown species found in {species_coordinates[0]}')
        return species_coordinates

    @torch.jit.export
    def atomic_energies(self, species_coordinates: Tuple[Tensor, Tensor],
                        cell: Optional[Tensor] = None,
                        pbc: Optional[Tensor] = None, average: bool = True) -> SpeciesEnergies:
        """Calculates predicted atomic energies of all atoms in a molecule

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled
            average: If True (the default) it returns the average over all models
                in the ensemble, should there be more than one (output shape (C, A)),
                otherwise it returns one atomic energy per model (output shape (M, C, A)).

        Returns:
            species_energies: tuple of tensors, species and atomic energies
        """
        assert isinstance(self.neural_networks, (Ensemble, ANIModel))
        species_coordinates = self._maybe_convert_species(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        atomic_energies = self.neural_networks._atomic_energies(species_aevs)
        atomic_energies += self.energy_shifter._atomic_saes(species_coordinates[0])

        if atomic_energies.dim() == 2:
            atomic_energies = atomic_energies.unsqueeze(0)
        if average:
            atomic_energies = atomic_energies.mean(dim=0)
        return SpeciesEnergies(species_coordinates[0], atomic_energies)

    # unfortunately this is an UGLY workaround to a torchscript bug
    @torch.jit.export
    def _recast_long_buffers(self):
        self.species_converter.conv_tensor = self.species_converter.conv_tensor.to(dtype=torch.long)
        self.aev_computer.triu_index = self.aev_computer.triu_index.to(dtype=torch.long)
        self.aev_computer.neighborlist._recast_long_buffers()

    def ase(self, **kwargs):
        """Get an ASE Calculator using this ANI model

        Arguments:
            kwargs: ase.Calculator kwargs

        Returns:
            calculator (:class:`ase.Calculator`): A calculator to be used with ASE
        """
        from . import ase
        return ase.Calculator(self.get_chemical_symbols(), self, **kwargs)

    def __getitem__(self, index: int) -> 'BuiltinModel':
        assert isinstance(self.neural_networks, Ensemble), "Your model doesn't have an ensemble of networks"
        return BuiltinModel(self.aev_computer,
                           self.neural_networks[index],
                           self.energy_shifter,
                           self.get_chemical_symbols(),
                           self.periodic_table_index)

    @torch.jit.export
    def members_energies(self, species_coordinates: Tuple[Tensor, Tensor],
                         cell: Optional[Tensor] = None,
                         pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        """Calculates predicted energies of all member modules

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled

        Returns:
            species_energies: species and members energies for the given configurations
                shape of energies is (M, C), where M is the number of modules in the ensemble.
        """
        assert isinstance(self.neural_networks, Ensemble), "Your model doesn't have an ensemble of networks"
        species, members_energies = self.atomic_energies(species_coordinates, cell=cell, pbc=pbc, average=False)
        return SpeciesEnergies(species, members_energies.sum(-1))

    @torch.jit.export
    def energies_qbcs(self, species_coordinates: Tuple[Tensor, Tensor],
                      cell: Optional[Tensor] = None,
                      pbc: Optional[Tensor] = None, unbiased: bool = True) -> SpeciesEnergiesQBC:
        """Calculates predicted predicted energies and qbc factors

        QBC factors are used for query-by-committee (QBC) based active learning
        (as described in the ANI-1x paper `less-is-more`_ ).

        .. _less-is-more:
            https://aip.scitation.org/doi/10.1063/1.5023802

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled
            unbiased: Whether to take an unbiased standard deviation over the ensemble's members.

        Returns:
            species_energies_qbcs: tuple of tensors, species, energies and qbc
                factors for the given configurations. The shapes of qbcs and
                energies are equal.
        """
        assert isinstance(self.neural_networks, Ensemble), "Your model doesn't have an ensemble of networks"
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
        assert isinstance(self.neural_networks, Ensemble), "Your model doesn't have an ensemble of networks"
        return self.neural_networks.size


def _get_component_modules(state_dict_file: str,
                           model_index: Optional[int] = None,
                           aev_computer_kwargs: Optional[Dict[str, Any]] = None,
                           ensemble_size: int = 8) -> Tuple[AEVComputer, NN, EnergyShifter, Sequence[str]]:
    if aev_computer_kwargs is None:
        aev_computer_kwargs = dict()
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
    aev_computer = aev_maker(**aev_computer_kwargs)
    atomic_networks = OrderedDict([(e, atomic_maker(e)) for e in elements])

    neural_networks: NN
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
                    **model_kwargs) -> BuiltinModel:
    # Helper function to toggle if the loading is done from an NC file or
    # directly using torchani and state_dicts
    use_neurochem_source = model_kwargs.pop('use_neurochem_source', False)
    model_index = model_kwargs.pop('model_index', None)
    pretrained = model_kwargs.pop('pretrained', True)

    # aev computer args
    if model_kwargs.pop('cell_list', False):
        neighborlist = 'cell_list'
    elif model_kwargs.pop('verlet_cell_list', False):
        neighborlist = 'verlet_cell_list'
    else:
        neighborlist = 'full_pairwise'
    aev_computer_kwargs = {'neighborlist': neighborlist,
                           'use_cuda_extension': model_kwargs.pop('use_cuda_extension', False)}

    if use_neurochem_source:
        assert info_file is not None, "Info file is needed to load from a neurochem source"
        assert pretrained, "Non pretrained models not available from neurochem source"
        from . import neurochem  # noqa
        components = neurochem.parse_resources._get_component_modules(info_file, model_index, aev_computer_kwargs)
    else:
        assert state_dict_file is not None
        components = _get_component_modules(state_dict_file, model_index, aev_computer_kwargs)

    aev_computer, neural_networks, energy_shifter, elements = components
    model = BuiltinModel(aev_computer, neural_networks, energy_shifter, elements, **model_kwargs)

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
