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
import torch
from torch import Tensor
from typing import Tuple, Optional, NamedTuple, Sequence
from .nn import SpeciesConverter, SpeciesEnergies, Ensemble
from .utils import ChemicalSymbolsToInts, PERIODIC_TABLE
from .aev import AEVComputer
from .compat import Final


class SpeciesEnergiesQBC(NamedTuple):
    species: Tensor
    energies: Tensor
    qbcs: Tensor


class BuiltinModel(torch.nn.Module):
    r"""Private template for the builtin ANI models """

    atomic_numbers: Tensor
    periodic_table_index: Final[bool]

    def __init__(self,
                 aev_computer,
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

    @classmethod
    def _from_neurochem_resources(cls, info_file_path, periodic_table_index=False, model_index=0):
        from . import neurochem  # noqa

        # this is used to load only 1 model (by default model 0)
        const_file, sae_file, ensemble_prefix, ensemble_size = neurochem.parse_neurochem_resources(info_file_path)
        if (model_index >= ensemble_size):
            raise ValueError("The ensemble size is only {}, model {} can't be loaded".format(ensemble_size, model_index))

        consts = neurochem.Constants(const_file)
        aev_computer = AEVComputer(**consts)
        energy_shifter = neurochem.load_sae(sae_file)

        network_dir = os.path.join('{}{}'.format(ensemble_prefix, model_index), 'networks')
        neural_networks = neurochem.load_model(consts.species, network_dir)

        return cls(aev_computer, neural_networks, energy_shifter, consts.species, periodic_table_index)

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

    @classmethod
    def _from_neurochem_resources(cls, info_file_path, periodic_table_index=False):
        from . import neurochem  # noqa
        # this is used to load only 1 model (by default model 0)
        const_file, sae_file, ensemble_prefix, ensemble_size = neurochem.parse_neurochem_resources(info_file_path)

        consts = neurochem.Constants(const_file)
        aev_computer = AEVComputer(**consts)
        energy_shifter = neurochem.load_sae(sae_file)
        neural_networks = neurochem.load_model_ensemble(consts.species,
                                                        ensemble_prefix, ensemble_size)

        return cls(aev_computer, neural_networks, energy_shifter, consts.species, periodic_table_index)

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
                           self.neural_networks[index], self.energy_shifter,
                           self.get_chemical_symbols(), self.periodic_table_index)
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
