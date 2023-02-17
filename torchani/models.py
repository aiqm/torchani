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
import warnings
from copy import deepcopy
from pathlib import Path
from collections import OrderedDict
import torch
from torch import Tensor
from torch.nn import Module
from typing import Tuple, Optional, NamedTuple, Sequence, Union, Dict, Any, Type, Callable, List
from .nn import SpeciesConverter, SpeciesEnergies, Ensemble, ANIModel
from .utils import ChemicalSymbolsToInts, PERIODIC_TABLE, EnergyShifter, path_is_writable
from .aev import AEVComputer
from .repulsion import RepulsionXTB
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
                 periodic_table_index: bool = True):

        super().__init__()
        # if periodic table index is True then it has been
        # set by the user, so lets output a warning that this is the default
        warnings.warn("The default is now to accept atomic numbers as indexes,"
                      " do not set periodic_table_index=True."
                      " if you need to accept raw indices set periodic_table_index=False")

        self.aev_computer = aev_computer
        self.neural_networks = neural_networks
        self.energy_shifter = energy_shifter
        self.species_to_tensor = ChemicalSymbolsToInts(elements)
        device = energy_shifter.self_energies.device
        self.species_converter = SpeciesConverter(elements).to(device)

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
        return ase.Calculator(self, **kwargs)

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


# Adaptor to use the aev computer as a three body potential
class AEVPotential(torch.nn.Module):
    cutoff: Final[float]

    def __init__(self, aev_computer: AEVComputer, neural_networks: NN):
        super().__init__()
        self.aev_computer = aev_computer
        self.neural_networks = neural_networks
        self.cutoff = aev_computer.radial_terms.cutoff

    def forward(
        self,
        element_idxs: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
        diff_vectors: Tensor,
    ) -> Tensor:
        assert diff_vectors is not None, "AEV potential needs diff vectors always"
        aevs = self.aev_computer._compute_aev(
            element_idxs=element_idxs,
            neighbor_idxs=neighbor_idxs,
            distances=distances,
            diff_vectors=diff_vectors,
        )
        energies = self.neural_networks((element_idxs, aevs)).energies
        return energies

    def members_energies(
        self,
        element_idxs: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
        diff_vectors: Tensor,
    ) -> Tensor:
        """
        Returns: members energies for the given configurations with shape (M, C),
            where M is the number of modules in the ensemble.
        """
        assert isinstance(self.neural_networks, Ensemble), "Your model doesn't have an ensemble of networks"
        aevs = self.aev_computer._compute_aev(
            element_idxs=element_idxs,
            neighbor_idxs=neighbor_idxs,
            distances=distances,
            diff_vectors=diff_vectors,
        )
        atomic_energies = self.neural_networks._atomic_energies((element_idxs, aevs))
        return atomic_energies.sum(-1)


class BuiltinModelPairInteractions(BuiltinModel):
    # NOTE: contribution of pairwise interactions to atomic energies is not
    # implemented yet

    def __init__(self, *args, **kwargs):
        potentials = kwargs.pop('pairwise_potentials', list())
        super().__init__(*args, **kwargs)
        assert isinstance(potentials, (tuple, list))
        potentials = list(potentials)
        potentials.append(AEVPotential(self.aev_computer, self.neural_networks))

        # We want to check the cutoffs of the potentials, and the cutoff of the
        # aev computer, and sort the "aev energy" and the "pairwise energies"
        # in order of decreasing cutoffs. this way the energy with the LARGEST
        # cutoff is computed first, then sequentially things that need smaller
        # cutoffs are computed.
        #
        # e.g. if the aev-potential has cutoff 10, and we have SRB with cutoff
        # 5 and repulsion with cutoff 3, we want to calculate:
        #
        # coords -> screen r<10 -> aev-energy -> screen r<5 -> SRB -> screen r<3 -> rep
        potentials = sorted(potentials, key=lambda x: x.cutoff, reverse=True)
        self.potentials = torch.nn.ModuleList(potentials)

        # Set the neighborlist cutoff to the largest cutoff in existence
        self.aev_computer.neighborlist.cutoff = self.potentials[0].cutoff

    def forward(
        self,
        species_coordinates: Tuple[Tensor, Tensor],
        cell: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None
    ) -> SpeciesEnergies:
        element_idxs, coordinates = self._maybe_convert_species(species_coordinates)
        neighbor_data = self.aev_computer.neighborlist(element_idxs, coordinates, cell, pbc)
        energies = torch.zeros(element_idxs.shape[0], device=element_idxs.device, dtype=coordinates.dtype)
        previous_cutoff = self.aev_computer.neighborlist.cutoff
        rescreen = self.aev_computer.neighborlist._rescreen_with_cutoff
        for pot in self.potentials:
            if pot.cutoff < previous_cutoff:
                neighbor_data = rescreen(
                    cutoff=pot.cutoff,
                    neighbor_idxs=neighbor_data.indices,
                    distances=neighbor_data.distances,
                    diff_vectors=neighbor_data.diff_vectors,
                )
                previous_cutoff = pot.cutoff
            energies = energies + pot(
                element_idxs=element_idxs,
                neighbor_idxs=neighbor_data.indices,
                distances=neighbor_data.distances,
                diff_vectors=neighbor_data.diff_vectors,
            )
        return self.energy_shifter((element_idxs, energies))

    def __getitem__(self, index: int) -> 'BuiltinModel':
        assert isinstance(self.neural_networks, Ensemble), "Your model doesn't have an ensemble of networks"
        non_aev_potentials = [p for p in self.potentials if not isinstance(p, AEVPotential)]
        return BuiltinModelPairInteractions(
            aev_computer=self.aev_computer,
            neural_networks=self.neural_networks[index],
            energy_shifter=self.energy_shifter,
            elements=self.get_chemical_symbols(),
            periodic_table_index=self.periodic_table_index,
            pairwise_potentials=non_aev_potentials,
        )

    @torch.jit.export
    def members_energies(self, species_coordinates: Tuple[Tensor, Tensor],
                         cell: Optional[Tensor] = None,
                         pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        assert isinstance(self.neural_networks, Ensemble), "Your model doesn't have an ensemble of networks"
        element_idxs, coordinates = self._maybe_convert_species(species_coordinates)
        neighbor_data = self.aev_computer.neighborlist(element_idxs, coordinates, cell, pbc)
        energies = torch.zeros(element_idxs.shape[0], device=element_idxs.device, dtype=coordinates.dtype)
        previous_cutoff = self.aev_computer.neighborlist.cutoff
        members_energies: Optional[Tensor] = None
        rescreen = self.aev_computer.neighborlist._rescreen_with_cutoff
        for pot in self.potentials:
            if pot.cutoff < previous_cutoff:
                neighbor_data = rescreen(
                    pot.cutoff,
                    neighbor_idxs=neighbor_data.indices,
                    distances=neighbor_data.distances,
                    diff_vectors=neighbor_data.diff_vectors,
                )
                previous_cutoff = pot.cutoff
            if isinstance(pot, AEVPotential):
                members_energies = pot.members_energies(
                    element_idxs,
                    neighbor_idxs=neighbor_data.indices,
                    distances=neighbor_data.distances,
                    diff_vectors=neighbor_data.diff_vectors,
                )
            else:
                energies += pot(
                    element_idxs,
                    neighbor_idxs=neighbor_data.indices,
                    distances=neighbor_data.distances,
                    diff_vectors=neighbor_data.diff_vectors,
                )
        assert members_energies is not None
        energies = self.energy_shifter((element_idxs, energies)).energies
        members_energies += energies.unsqueeze(0)
        return SpeciesEnergies(element_idxs, members_energies)


def _get_component_modules(
    state_dict_file: str,
    model_index: Optional[int] = None,
    aev_computer_kwargs: Optional[Dict[str, Any]] = None,
    ensemble_size: int = 8,
    atomic_maker: Optional[Callable[[str], torch.nn.Module]] = None,
    aev_maker: Optional[Callable[..., AEVComputer]] = None,
    elements: Optional[Sequence[str]] = None,
    self_energies: Optional[Sequence[float]] = None
) -> Tuple[AEVComputer, NN, EnergyShifter, Sequence[str]]:
    # Component modules are obtained by default from the name of the state_dict_file,
    # but they can be overriden by passing specific parameters

    if aev_computer_kwargs is None:
        aev_computer_kwargs = dict()
    # This generates ani-style architectures without neurochem
    name = state_dict_file.split('_')[0]
    _elements: Sequence[str]
    if name == 'ani1x':
        _aev_maker = aev_maker or AEVComputer.like_1x
        _atomic_maker = atomic_maker or atomics.like_1x
        _elements = elements or ('H', 'C', 'N', 'O')
    elif name == 'ani1ccx':
        _aev_maker = aev_maker or AEVComputer.like_1ccx
        _atomic_maker = atomic_maker or atomics.like_1ccx
        _elements = elements or ('H', 'C', 'N', 'O')
    elif name == 'ani2x':
        _aev_maker = aev_maker or AEVComputer.like_2x
        _atomic_maker = atomic_maker or atomics.like_2x
        _elements = elements or ('H', 'C', 'N', 'O', 'S', 'F', 'Cl')
    else:
        raise ValueError(f'{name} is not a supported model')

    aev_computer = _aev_maker(**aev_computer_kwargs)
    atomic_networks = OrderedDict([(e, _atomic_maker(e)) for e in _elements])
    energy_shifter = EnergyShifter(self_energies or [0.0] * len(_elements))
    neural_networks: NN
    if model_index is None:
        neural_networks = Ensemble([ANIModel(deepcopy(atomic_networks)) for _ in range(ensemble_size)])
    else:
        neural_networks = ANIModel(atomic_networks)

    return aev_computer, neural_networks, energy_shifter, _elements


def _fetch_state_dict(state_dict_file: str,
                      model_index: Optional[int] = None,
                      local: bool = False,
                      private: bool = False) -> 'OrderedDict[str, Tensor]':
    # if we want a pretrained model then we load the state dict from a
    # remote url or a local path
    # NOTE: torch.hub caches remote state_dicts after they have been downloaded
    if local:
        return torch.load(state_dict_file)

    model_dir = Path(__file__).parent.joinpath('resources/state_dicts').as_posix()
    if not path_is_writable(model_dir):
        model_dir = os.path.expanduser('~/.local/torchani/')
    if private:
        url = f'http://moria.chem.ufl.edu/animodel/private/{state_dict_file}'
    else:
        tag = 'v0.1'
        url = f'https://github.com/roitberg-group/torchani_model_zoo/releases/download/{tag}/{state_dict_file}'

    # for now for simplicity we load a state dict for the ensemble directly and
    # then parse if needed
    # The argument to map_location is OK but the function is incorrectly typed
    # in the pytorch stubs
    state_dict = torch.hub.load_state_dict_from_url(url, model_dir=model_dir, map_location=torch.device('cpu'))  # type: ignore
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
    else:
        new_state_dict = OrderedDict(state_dict)
    return new_state_dict


def _load_ani_model(state_dict_file: Optional[str] = None,
                    info_file: Optional[str] = None,
                    repulsion_kwargs: Optional[Dict[str, Any]] = None,
                    repulsion: bool = False,
                    cutoff_fn: Union[str, torch.nn.Module] = 'cosine',  # for aev
                    pretrained: bool = True,
                    model_index: int = None,
                    use_neurochem_source: bool = False,
                    ensemble_size: int = 8,
                    **model_kwargs) -> BuiltinModel:
    # Helper function to toggle if the loading is done from an NC file or
    # directly using torchani and state_dicts
    aev_maker = model_kwargs.pop('aev_maker', None)
    atomic_maker = model_kwargs.pop('atomic_maker', None)
    elements = model_kwargs.pop('elements', None)

    if pretrained:
        assert ensemble_size == 8
        assert not repulsion, "No pretrained model with those characteristics exists"
        assert cutoff_fn == 'cosine', "No pretrained model with those characteristics exists"

    # aev computer args
    cell_list = model_kwargs.pop('cell_list', False)
    verlet_cell_list = model_kwargs.pop('verlet_cell_list', False)
    if cell_list:
        neighborlist = 'cell_list'
    elif verlet_cell_list:
        neighborlist = 'verlet_cell_list'
    else:
        neighborlist = 'full_pairwise'
    aev_computer_kwargs = {'neighborlist': neighborlist,
                           'cutoff_fn': cutoff_fn,
                           'use_cuda_extension': model_kwargs.pop('use_cuda_extension', False)}

    if use_neurochem_source:
        assert info_file is not None, "Info file is needed to load from a neurochem source"
        assert pretrained, "Non pretrained models not available from neurochem source"
        # neurochem is legacy and not type-checked
        from . import neurochem  # noqa
        components = neurochem.parse_resources._get_component_modules(info_file, model_index, aev_computer_kwargs)  # type: ignore
    else:
        assert state_dict_file is not None
        components = _get_component_modules(
            state_dict_file,
            model_index,
            aev_computer_kwargs,
            atomic_maker=atomic_maker,
            aev_maker=aev_maker,
            elements=elements,
            ensemble_size=ensemble_size)

    aev_computer, neural_networks, energy_shifter, elements = components

    model_class: Type[BuiltinModel]
    if repulsion:
        cutoff = aev_computer.radial_terms.cutoff
        pairwise_potentials: List[torch.nn.Module] = []
        base_repulsion_kwargs = {'symbols': elements, 'cutoff': cutoff}
        if repulsion_kwargs is not None:
            base_repulsion_kwargs.update(repulsion_kwargs)
        pairwise_potentials.append(RepulsionXTB(**base_repulsion_kwargs))
        model_kwargs.update({'pairwise_potentials': pairwise_potentials})
        model_class = BuiltinModelPairInteractions
    else:
        model_class = BuiltinModel

    model = model_class(aev_computer, neural_networks, energy_shifter, elements, **model_kwargs)

    if pretrained and not use_neurochem_source:
        assert state_dict_file is not None
        model.load_state_dict(_fetch_state_dict(state_dict_file, model_index))
    return model


def ANI1x(**kwargs) -> BuiltinModel:
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


def ANI1ccx(**kwargs) -> BuiltinModel:
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


def ANI2x(**kwargs) -> BuiltinModel:
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
