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
import typing as tp
import os
import warnings
from copy import deepcopy
from pathlib import Path
from collections import OrderedDict

import torch
from torch import Tensor
from torch.nn import Module
from torch.jit import Final

from torchani import atomics
from torchani.tuples import (
    SpeciesEnergies,
    SpeciesEnergiesQBC,
    AtomicStdev,
    SpeciesForces,
    ForceStdev,
    ForceMagnitudes
)
from torchani.nn import SpeciesConverter, Ensemble, ANIModel
from torchani.utils import ChemicalSymbolsToInts, PERIODIC_TABLE, EnergyShifter, path_is_writable
from torchani.aev import AEVComputer
from torchani.potentials import (
    AEVPotential,
    RepulsionXTB,
    TwoBodyDispersionD3,
    Potential,
    PairwisePotential,
)
from torchani.neighbors import rescreen


NN = tp.Union[ANIModel, Ensemble]


class BuiltinModel(Module):
    r"""Private template for the builtin ANI models """

    atomic_numbers: Tensor
    periodic_table_index: Final[bool]

    def __init__(self,
                 aev_computer: AEVComputer,
                 neural_networks: NN,
                 energy_shifter: EnergyShifter,
                 elements: tp.Sequence[str],
                 periodic_table_index: bool = True):

        super().__init__()
        if not periodic_table_index:
            # if periodic table index is True then it has been
            # set by the user, so lets output a warning that this is the default
            warnings.warn("The default is now to accept atomic numbers as indices,"
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
    def get_chemical_symbols(self) -> tp.Tuple[str, ...]:
        return tuple(PERIODIC_TABLE[z] for z in self.atomic_numbers)

    def forward(self, species_coordinates: tp.Tuple[Tensor, Tensor],
                cell: tp.Optional[Tensor] = None,
                pbc: tp.Optional[Tensor] = None) -> SpeciesEnergies:
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
    def _maybe_convert_species(self, species_coordinates: tp.Tuple[Tensor, Tensor]) -> tp.Tuple[Tensor, Tensor]:
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        if (species_coordinates[0] >= self.aev_computer.num_species).any():
            raise ValueError(f'Unknown species found in {species_coordinates[0]}')
        return species_coordinates

    @torch.jit.export
    def atomic_energies(self, species_coordinates: tp.Tuple[Tensor, Tensor],
                        cell: tp.Optional[Tensor] = None,
                        pbc: tp.Optional[Tensor] = None,
                        average: bool = True,
                        shift_energy: bool = True) -> SpeciesEnergies:
        """Calculates predicted atomic energies of all atoms in a molecule

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled
            average: If True (the default) it returns the average over all models
                     in the ensemble, should there be more than one (output shape (C, A)),
                     otherwise it returns one atomic energy per model (output shape (M, C, A)).
            shift_energy: returns atomic energies shifted with ground state atomic energies.
                          Set to True by default

        Returns:
            species_energies: tuple of tensors, species and atomic energies
        """
        assert isinstance(self.neural_networks, (Ensemble, ANIModel))
        species_coordinates = self._maybe_convert_species(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        atomic_energies = self.neural_networks._atomic_energies(species_aevs)

        if atomic_energies.dim() == 2:
            atomic_energies = atomic_energies.unsqueeze(0)

        if shift_energy:
            atomic_energies += self.energy_shifter._atomic_saes(species_coordinates[0])

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
    def members_energies(self, species_coordinates: tp.Tuple[Tensor, Tensor],
                         cell: tp.Optional[Tensor] = None,
                         pbc: tp.Optional[Tensor] = None) -> SpeciesEnergies:
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
        species, members_energies = self.atomic_energies(species_coordinates, cell=cell, pbc=pbc,
                                                         shift_energy=True, average=False)
        return SpeciesEnergies(species, members_energies.sum(-1))

    def members_forces(self, species_coordinates: tp.Tuple[Tensor, Tensor],
                       cell: tp.Optional[Tensor] = None,
                       pbc: tp.Optional[Tensor] = None,
                       average: bool = False) -> SpeciesForces:
        """Calculates predicted forces from ensemble members, can return the average prediction

        Args:
            species_coordinates: minibatch of configurations
            average: boolean value which determines whether to return the predicted forces from each model or the ensemble average
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to none if PBC is not enabled

        Returns:
            SpeciesForces: species, molecular energies, and atomic forces predicted by an ensemble of neural network models
        """
        assert isinstance(self.neural_networks, Ensemble), "Your model doesn't have an ensemble of networks"
        coordinates = species_coordinates[1].requires_grad_()
        members_energies = self.members_energies(species_coordinates, cell, pbc).energies
        forces_list = []
        for energy in members_energies:
            derivative = torch.autograd.grad(energy.sum(), coordinates, retain_graph=True)[0]
            force = -derivative
            forces_list.append(force)
        forces = torch.stack(forces_list, dim=0)
        if average:
            forces = forces.mean(0)
        return SpeciesForces(species_coordinates[0], members_energies, forces)

    @torch.jit.export
    def energies_qbcs(self, species_coordinates: tp.Tuple[Tensor, Tensor],
                      cell: tp.Optional[Tensor] = None,
                      pbc: tp.Optional[Tensor] = None, unbiased: bool = True) -> SpeciesEnergiesQBC:
        """Calculates predicted predicted energies and qbc factors

        QBC factors are used for query-by-committee (QBC) based active learning
        (as described in the ANI-1x paper `less-is-more`_ ).

        .. _less-is-more:
            https://aip.scitation.org/doi/10.1063/1.5023802

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled,
                    set to None if PBC is not enabled
            unbiased: Whether to unbias the standard deviation over ensemble predictions

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

    def atomic_stdev(self, species_coordinates: tp.Tuple[Tensor, Tensor],
                    cell: tp.Optional[Tensor] = None,
                    pbc: tp.Optional[Tensor] = None,
                    average: bool = False,
                    shift_energy: bool = False,
                    unbiased: bool = True) -> AtomicStdev:
        """
        Largely does the same thing as the atomic_energies function, but with a different set of default inputs.
        Returns standard deviation in atomic energy predictions across the ensemble.

        shift_energy returns the shifted atomic energies according to the model used
        """
        assert isinstance(self.neural_networks, Ensemble), "Your model doesn't have an ensemble of networks"
        species_coordinates = self._maybe_convert_species(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        atomic_energies = self.neural_networks._atomic_energies(species_aevs)

        if atomic_energies.dim() == 2:
            atomic_energies = atomic_energies.unsqueeze(0)

        stdev_atomic_energies = atomic_energies.std(0, unbiased=unbiased)

        if average:
            atomic_energies = atomic_energies.mean(0)

        if shift_energy:
            atomic_energies += self.energy_shifter._atomic_saes(species_coordinates[0])

        return AtomicStdev(species_coordinates[0], atomic_energies, stdev_atomic_energies)

    def force_magnitudes(self, species_coordinates: tp.Tuple[Tensor, Tensor],
                         cell: tp.Optional[Tensor] = None,
                         pbc: tp.Optional[Tensor] = None,
                         average: bool = True) -> ForceMagnitudes:
        '''
        Computes the L2 norm of predicted atomic force vectors, returning magnitudes,
        averaged by default.

        Args:
            species_coordinates: minibatch of configurations
            average: by default, returns the ensemble average magnitude for each atomic force vector
        '''
        assert isinstance(self.neural_networks, Ensemble), "Your model doesn't have an ensemble of networks"

        species, _, members_forces = self.members_forces(species_coordinates, cell, pbc)
        magnitudes = members_forces.norm(dim=-1)
        if average:
            magnitudes = magnitudes.mean(0)

        return ForceMagnitudes(species, magnitudes)

    def force_qbc(self, species_coordinates: tp.Tuple[Tensor, Tensor],
                   cell: tp.Optional[Tensor] = None,
                   pbc: tp.Optional[Tensor] = None,
                   average: bool = False,
                   unbiased: bool = True) -> ForceStdev:
        """
        Returns the mean force magnitudes and relative range and standard deviation
        of predicted forces across an ensemble of networks.

        Args:
            species_coordinates: minibatch of configurations
            average: returns magnitudes predicted by each model by default
            unbiased: whether or not to use Bessel's correction in computing the standard deviation, True by default
        """
        assert isinstance(self.neural_networks, Ensemble), "Your model doesn't have an ensemble of networks"
        species, magnitudes = self.force_magnitudes(species_coordinates, cell, pbc, average=False)

        max_magnitudes = magnitudes.max(dim=0).values
        min_magnitudes = magnitudes.min(dim=0).values

        mean_magnitudes = magnitudes.mean(0)
        relative_stdev = (magnitudes.std(0, unbiased=unbiased) + 1e-8) / (mean_magnitudes + 1e-8)
        relative_range = ((max_magnitudes - min_magnitudes) + 1e-8) / (mean_magnitudes + 1e-8)

        if average:
            magnitudes = mean_magnitudes

        return ForceStdev(species, magnitudes, relative_stdev, relative_range)

    def __len__(self):
        assert isinstance(self.neural_networks, Ensemble), "Your model doesn't have an ensemble of networks"
        return self.neural_networks.size


class BuiltinModelPairInteractions(BuiltinModel):
    def __init__(
        self,
        *args,
        pairwise_potentials: tp.Iterable[PairwisePotential] = tuple(),
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        potentials: tp.List[Potential] = list(pairwise_potentials)
        aev_potential = AEVPotential(self.aev_computer, self.neural_networks)
        self.size = aev_potential.size
        potentials.append(aev_potential)

        # We want to check the cutoffs of the potentials, and sort them
        # in order of decreasing cutoffs. this way the potential with the LARGEST
        # cutoff is computed first, then sequentially things that need smaller
        # cutoffs are computed.
        # e.g. if the aev-potential has cutoff 10, and we have SRB with cutoff
        # 5 and repulsion with cutoff 3, we want to calculate:
        #
        # coords -> screen r<10 -> aev-energy -> screen r<5 -> SRB -> screen r<3 -> rep
        potentials = sorted(potentials, key=lambda x: x.cutoff, reverse=True)
        self.potentials = torch.nn.ModuleList(potentials)

        # Override the neighborlist cutoff with the largest cutoff in existence
        self.aev_computer.neighborlist.cutoff = self.potentials[0].cutoff  # type: ignore

    def forward(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None
    ) -> SpeciesEnergies:
        element_idxs, coordinates = self._maybe_convert_species(species_coordinates)
        neighbor_data = self.aev_computer.neighborlist(element_idxs, coordinates, cell, pbc)
        energies = torch.zeros(element_idxs.shape[0], device=element_idxs.device, dtype=coordinates.dtype)
        previous_cutoff = self.aev_computer.neighborlist.cutoff
        for pot in self.potentials:
            if pot.cutoff < previous_cutoff:
                neighbor_data = rescreen(pot.cutoff, neighbor_data)
                previous_cutoff = pot.cutoff
            energies += pot(element_idxs, neighbor_data)
        return self.energy_shifter((element_idxs, energies))

    @torch.jit.export
    def atomic_energies(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        average: bool = True,
        shift_energy: bool = True
    ) -> SpeciesEnergies:
        assert isinstance(self.neural_networks, (Ensemble, ANIModel))
        element_idxs, coordinates = self._maybe_convert_species(species_coordinates)
        neighbor_data = self.aev_computer.neighborlist(element_idxs, coordinates, cell, pbc)
        previous_cutoff = self.aev_computer.neighborlist.cutoff

        # Here we add an extra axis to account for different models,
        # some potentials output atomic energies with shape (M, N, A), where
        # M is all models in the ensemble
        atomic_energies = torch.zeros(
            (self.size, element_idxs.shape[0], element_idxs.shape[1]),
            dtype=coordinates.dtype,
            device=coordinates.device
        )
        for pot in self.potentials:
            if pot.cutoff < previous_cutoff:
                neighbor_data = rescreen(pot.cutoff, neighbor_data)
                previous_cutoff = pot.cutoff
            atomic_energies += pot.atomic_energies(element_idxs, neighbor_data)

        if shift_energy:
            atomic_energies += self.energy_shifter._atomic_saes(element_idxs).unsqueeze(0)

        if average:
            atomic_energies = atomic_energies.mean(dim=0)
        return SpeciesEnergies(species_coordinates[0], atomic_energies)

    # NOTE: members_energies does not need to be overriden, it works correctly as is

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


def _get_component_modules(
    state_dict_file: str,
    model_index: tp.Optional[int] = None,
    aev_computer_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ensemble_size: int = 8,
    atomic_maker: tp.Optional[tp.Callable[[str], torch.nn.Module]] = None,
    aev_maker: tp.Optional[tp.Callable[..., AEVComputer]] = None,
    elements: tp.Optional[tp.Sequence[str]] = None,
    self_energies: tp.Optional[tp.Sequence[float]] = None
) -> tp.Tuple[AEVComputer, NN, EnergyShifter, tp.Sequence[str]]:
    # Component modules are obtained by default from the name of the state_dict_file,
    # but they can be overriden by passing specific parameters

    if aev_computer_kwargs is None:
        aev_computer_kwargs = dict()
    # This generates ani-style architectures without neurochem
    name = state_dict_file.split('_')[0]
    _elements: tp.Sequence[str]
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
                      model_index: tp.Optional[int] = None,
                      local: bool = False,
                      private: bool = False) -> tp.OrderedDict[str, Tensor]:
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


def _load_ani_model(state_dict_file: tp.Optional[str] = None,
                    info_file: tp.Optional[str] = None,
                    repulsion_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
                    repulsion: bool = False,
                    dispersion_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
                    dispersion: bool = False,
                    cutoff_fn: tp.Union[str, torch.nn.Module] = 'cosine',  # for aev
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
                           'use_cuda_extension': model_kwargs.pop('use_cuda_extension', False),
                           'use_cuaev_interface': model_kwargs.pop('use_cuaev_interface', False)}

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

    model_class: tp.Type[BuiltinModel]
    if repulsion or dispersion:
        model_class = BuiltinModelPairInteractions
        pairwise_potentials: tp.List[torch.nn.Module] = []
        cutoff = aev_computer.radial_terms.cutoff
        potential_kwargs = {'symbols': elements, 'cutoff': cutoff}
        if repulsion:
            base_repulsion_kwargs = deepcopy(potential_kwargs)
            base_repulsion_kwargs.update(repulsion_kwargs or {})
            pairwise_potentials.append(RepulsionXTB(**base_repulsion_kwargs))
        if dispersion:
            base_dispersion_kwargs = deepcopy(potential_kwargs)
            base_dispersion_kwargs.update(dispersion_kwargs or {})
            pairwise_potentials.append(
                TwoBodyDispersionD3.from_functional(
                    **base_dispersion_kwargs,
                ),
            )
        model_kwargs.update({'pairwise_potentials': pairwise_potentials})
    else:
        model_class = BuiltinModel

    model = model_class(
        aev_computer,
        neural_networks,
        energy_shifter,
        elements,
        **model_kwargs,
    )

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


def ANIdr(
    pretrained: bool = True,
    model_index: tp.Optional[int] = None,
    **kwargs,
):
    """ANI model trained with both dispersion and repulsion

    The level of theory is B973c, it is an ensemble of 7 models.
    It predicts
    energies on HCNOFSCl elements
    """
    # TODO: Fix this
    if model_index is not None:
        raise ValueError(
            "Currently ANIdr only supports model_index=None, "
            "to get individual models please index the ensemble"
        )

    # An ani model with dispersion
    symbols = ('H', 'C', 'N', 'O', 'S', 'F', 'Cl')
    model = ANI2x(
        pretrained=False,
        cutoff_fn='smooth',
        atomic_maker=atomics.like_dr,
        ensemble_size=7,
        dispersion=True,
        dispersion_kwargs={
            'symbols': symbols,
            'cutoff': 8.5,
            'cutoff_fn': 'smooth4',
            'functional': 'B973c'
        },
        repulsion=True,
        repulsion_kwargs={
            'symbols': symbols,
            'cutoff': 5.3,
            'cutoff_fn': 'smooth2'
        },
        model_index=model_index,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(
            _fetch_state_dict(
                'anidr_state_dict.pt',
                model_index=model_index,
                private=True,
            )
        )
    return model
