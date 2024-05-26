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

    # output shape of energies is (M, C), where M is the number of ensemble members
    _, members_energies = ani1x.members_energies((species, coordinates))

    # output shape of energies is (A, C) where A is num atoms in the minibatch
    # atomic energies are averaged over all models by default
    _, atomic_energies = ani1x.atomic_energies((species, coordinates))

    # qbc factors are used for active learning, shape is equal to energies
    _, energies, qbcs = ani1x.energies_qbcs((species, coordinates))

    # individual models of the ensemble can be obtained by indexing,
    # and they have the same functionality as the ensembled model
    model0 = ani1x[0]
    _, energies = model0((species, coordinates))
"""
import typing as tp

import torch
from torch import Tensor
from torch.jit import Final
import typing_extensions as tpx

from torchani.tuples import (
    SpeciesEnergies,
    SpeciesEnergiesQBC,
    AtomicStdev,
    SpeciesForces,
    ForceStdev,
    ForceMagnitudes,
)
from torchani.atomics import AtomicContainer
from torchani.nn import SpeciesConverter
from torchani.utils import (
    PERIODIC_TABLE,
    ATOMIC_NUMBERS,
)
from torchani.aev import AEVComputer
from torchani.potentials import (
    AEVPotential,
    Potential,
    PairPotential,
    EnergyAdder,
)
from torchani.neighbors import rescreen


class BuiltinModel(torch.nn.Module):
    r"""Private template for the builtin ANI models"""

    atomic_numbers: Tensor
    periodic_table_index: Final[bool]

    def __init__(
        self,
        aev_computer: AEVComputer,
        neural_networks: AtomicContainer,
        energy_shifter: EnergyAdder,
        elements: tp.Sequence[str],
        periodic_table_index: bool = True,
    ):
        super().__init__()

        self.aev_computer = aev_computer
        self.neural_networks = neural_networks
        self.energy_shifter = energy_shifter
        device = self.energy_shifter.self_energies.device
        self.species_converter = SpeciesConverter(elements).to(device)

        self.periodic_table_index = periodic_table_index
        numbers = torch.tensor([ATOMIC_NUMBERS[e] for e in elements], dtype=torch.long)
        self.register_buffer("atomic_numbers", numbers)

        # checks are performed to make sure all modules passed support the
        # correct number of species
        assert len(self.energy_shifter.self_energies) == len(self.atomic_numbers)
        assert self.aev_computer.num_species == len(self.atomic_numbers)
        assert self.neural_networks.num_species == len(self.atomic_numbers)

    def to_infer_model(self, use_mnp: bool = False) -> tpx.Self:
        """Convert the neural networks module of the model into a module
        optimized for inference.

        Currently this function assumes that the atomic networks consist of
        an MLP with CELU activation functions, all with the same alpha.
        """
        self.neural_networks = self.neural_networks.to_infer_model(use_mnp=use_mnp)
        return self

    @torch.jit.unused
    def get_chemical_symbols(self) -> tp.Tuple[str, ...]:
        return tuple(PERIODIC_TABLE[z] for z in self.atomic_numbers)

    def forward(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> SpeciesEnergies:
        """Calculates predicted energies for minibatch of configurations

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set
                to None if PBC is not enabled

        Returns:
            species_energies: tuple of tensors, species and energies for the
                given configurations
        """
        species_coordinates = self._maybe_convert_species(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        species, energies = self.neural_networks(species_aevs)
        return SpeciesEnergies(species, energies + self.energy_shifter(species))

    @torch.jit.export
    def _maybe_convert_species(
        self, species_coordinates: tp.Tuple[Tensor, Tensor]
    ) -> tp.Tuple[Tensor, Tensor]:
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        if (species_coordinates[0] >= self.aev_computer.num_species).any():
            raise ValueError(f"Unknown species found in {species_coordinates[0]}")
        return species_coordinates

    @torch.jit.export
    def atomic_energies(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        average: bool = True,
        shift_energy: bool = True,
        include_non_aev_potentials: bool = True,
    ) -> SpeciesEnergies:
        """Calculates predicted atomic energies of all atoms in a molecule

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set
                to None if PBC is not enabled
            average: If True (the default) it returns the average over all
                models in the ensemble, should there be more than one (output shape
                (C, A)), otherwise it returns one atomic energy per model (output
                shape (M, C, A)).
            shift_energy: returns atomic energies shifted with ground state
                atomic energies. Set to True by default

        Returns:
            species_energies: tuple of tensors, species and atomic energies
        """
        species_coordinates = self._maybe_convert_species(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        atomic_energies = self.neural_networks._atomic_energies(species_aevs)

        if shift_energy:
            atomic_energies += self.energy_shifter.atomic_energies(
                species_coordinates[0]
            )

        if average:
            atomic_energies = atomic_energies.mean(dim=0)
        return SpeciesEnergies(species_coordinates[0], atomic_energies)

    # unfortunately this is an UGLY workaround to a torchscript bug
    @torch.jit.export
    def _recast_long_buffers(self) -> None:
        self.species_converter.conv_tensor = self.species_converter.conv_tensor.to(
            dtype=torch.long
        )
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

    def __getitem__(self, index: int) -> tpx.Self:
        return type(self)(
            self.aev_computer,
            self.neural_networks.member(index),
            self.energy_shifter,
            self.get_chemical_symbols(),
            self.periodic_table_index,
        )

    @torch.jit.export
    def members_energies(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> SpeciesEnergies:
        """Calculates predicted energies of all member modules

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set
                to None if PBC is not enabled

        Returns:
            species_energies: species and members energies for the given
                configurations shape of energies is (M, C), where M is the number
                of modules in the ensemble.
        """
        species, members_energies = self.atomic_energies(
            species_coordinates,
            cell=cell,
            pbc=pbc,
            shift_energy=True,
            average=False,
            include_non_aev_potentials=True,
        )
        return SpeciesEnergies(species, members_energies.sum(-1))

    @torch.jit.unused
    def members_forces(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        average: bool = False,
    ) -> SpeciesForces:
        """Calculates predicted forces from ensemble members

        Args:
            species_coordinates: minibatch of configurations
            average: boolean value which determines whether to return the
                predicted forces from each model or the ensemble average
            cell: the cell used in PBC computation, set to None if PBC is not
                enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set
                to none if PBC is not enabled

        Returns:
            SpeciesForces: species, molecular energies, and atomic forces
                predicted by an ensemble of neural network models
        """
        coordinates = species_coordinates[1].requires_grad_()
        members_energies = self.members_energies(
            species_coordinates, cell, pbc
        ).energies
        forces_list = []
        for energy in members_energies:
            derivative = torch.autograd.grad(
                energy.sum(), coordinates, retain_graph=True
            )[0]
            force = -derivative
            forces_list.append(force)
        forces = torch.stack(forces_list, dim=0)
        if average:
            forces = forces.mean(0)
        return SpeciesForces(species_coordinates[0], members_energies, forces)

    @torch.jit.export
    def energies_qbcs(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        unbiased: bool = True,
    ) -> SpeciesEnergiesQBC:
        """Calculates predicted predicted energies and qbc factors

        QBC factors are used for query-by-committee (QBC) based active learning
        (as described in the ANI-1x paper `less-is-more`_ ).

        If the model has only 1 network, then qbc factors are all 0.0

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
        species, energies = self.members_energies(species_coordinates, cell, pbc)

        if self.neural_networks.num_networks == 1:
            qbc_factors = torch.zeros_like(energies).squeeze(0)
        else:
            # standard deviation is taken across ensemble members
            qbc_factors = energies.std(0, unbiased=unbiased)

        # rho's (qbc factors) are weighted by dividing by the square root of
        # the number of atoms in each molecule
        num_atoms = (species >= 0).sum(dim=1, dtype=energies.dtype)
        qbc_factors = qbc_factors / num_atoms.sqrt()
        energies = energies.mean(dim=0)
        assert qbc_factors.shape == energies.shape
        return SpeciesEnergiesQBC(species, energies, qbc_factors)

    @torch.jit.export
    def atomic_stdev(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        average: bool = False,
        shift_energy: bool = False,
        unbiased: bool = True,
    ) -> AtomicStdev:
        r"""Returns standard deviation of atomic energies across an ensemble

        shift_energy returns the shifted atomic energies according to the model used

        If the model has only 1 network, a value of 0.0 is output for the stdev
        """
        species_coordinates = self._maybe_convert_species(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        atomic_energies = self.neural_networks._atomic_energies(species_aevs)

        if shift_energy:
            atomic_energies += self.energy_shifter.atomic_energies(
                species_coordinates[0]
            )

        if self.neural_networks.num_networks == 1:
            stdev_atomic_energies = torch.zeros_like(atomic_energies).squeeze(0)
        else:
            stdev_atomic_energies = atomic_energies.std(0, unbiased=unbiased)

        if average:
            atomic_energies = atomic_energies.mean(0)

        return AtomicStdev(
            species_coordinates[0], atomic_energies, stdev_atomic_energies
        )

    @torch.jit.unused
    def force_magnitudes(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        average: bool = True,
    ) -> ForceMagnitudes:
        """
        Computes the L2 norm of predicted atomic force vectors, returning magnitudes,
        averaged by default.

        Args:
            species_coordinates: minibatch of configurations
            average: by default, returns the ensemble average magnitude for
                each atomic force vector
        """
        species, _, members_forces = self.members_forces(species_coordinates, cell, pbc)
        magnitudes = members_forces.norm(dim=-1)
        if average:
            magnitudes = magnitudes.mean(0)

        return ForceMagnitudes(species, magnitudes)

    @torch.jit.unused
    def force_qbc(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        average: bool = False,
        unbiased: bool = True,
    ) -> ForceStdev:
        """
        Returns the mean force magnitudes and relative range and standard deviation
        of predicted forces across an ensemble of networks.

        Args:
            species_coordinates: minibatch of configurations
            average: returns magnitudes predicted by each model by default
            unbiased: whether or not to use Bessel's correction in computing
                the standard deviation, True by default
        """
        species, magnitudes = self.force_magnitudes(
            species_coordinates, cell, pbc, average=False
        )

        max_magnitudes = magnitudes.max(dim=0).values
        min_magnitudes = magnitudes.min(dim=0).values

        if self.neural_networks.num_networks == 1:
            relative_stdev = torch.zeros_like(magnitudes).squeeze(0)
            relative_range = torch.ones_like(magnitudes).squeeze(0)
        else:
            mean_magnitudes = magnitudes.mean(0)
            relative_stdev = (magnitudes.std(0, unbiased=unbiased) + 1e-8) / (
                mean_magnitudes + 1e-8
            )
            relative_range = ((max_magnitudes - min_magnitudes) + 1e-8) / (
                mean_magnitudes + 1e-8
            )

        if average:
            magnitudes = mean_magnitudes

        return ForceStdev(species, magnitudes, relative_stdev, relative_range)

    def __len__(self):
        return self.neural_networks.num_networks


class PairPotentialsModel(BuiltinModel):
    def __init__(
        self, *args, pairwise_potentials: tp.Iterable[PairPotential] = tuple(), **kwargs
    ):
        super().__init__(*args, **kwargs)
        potentials: tp.List[Potential] = list(pairwise_potentials)
        aev_potential = AEVPotential(self.aev_computer, self.neural_networks)
        potentials.append(aev_potential)

        # We want to check the cutoffs of the potentials, and sort them
        # in order of decreasing cutoffs. this way the potential with the LARGEST
        # cutoff is computed first, then sequentially things that need smaller
        # cutoffs are computed.
        potentials = sorted(potentials, key=lambda x: x.cutoff, reverse=True)
        self.potentials = torch.nn.ModuleList(potentials)

    def forward(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> SpeciesEnergies:
        element_idxs, coordinates = self._maybe_convert_species(species_coordinates)
        previous_cutoff = self.potentials[0].cutoff
        neighbor_data = self.aev_computer.neighborlist(
            element_idxs, coordinates, previous_cutoff, cell, pbc
        )
        energies = torch.zeros(
            element_idxs.shape[0], device=element_idxs.device, dtype=coordinates.dtype
        )
        for pot in self.potentials:
            cutoff = pot.cutoff
            if cutoff < previous_cutoff:
                neighbor_data = rescreen(cutoff, neighbor_data)
                previous_cutoff = cutoff
            energies += pot(element_idxs, neighbor_data)
        return SpeciesEnergies(
            element_idxs, energies + self.energy_shifter(element_idxs)
        )

    @torch.jit.export
    def atomic_energies(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        average: bool = True,
        shift_energy: bool = True,
        include_non_aev_potentials: bool = True,
    ) -> SpeciesEnergies:
        element_idxs, coordinates = self._maybe_convert_species(species_coordinates)
        previous_cutoff = self.potentials[0].cutoff
        neighbor_data = self.aev_computer.neighborlist(
            element_idxs, coordinates, previous_cutoff, cell, pbc
        )

        # Here we add an extra axis to account for different models,
        # some potentials output atomic energies with shape (M, N, A), where
        # M is all models in the ensemble
        atomic_energies = torch.zeros(
            (
                self.neural_networks.num_networks,
                element_idxs.shape[0],
                element_idxs.shape[1],
            ),
            dtype=coordinates.dtype,
            device=coordinates.device,
        )
        if torch.jit.is_scripting():
            assert (
                include_non_aev_potentials
            ), "Scripted models must include non aev potentials in atomic energies"
            for pot in self.potentials:
                cutoff = pot.cutoff
                if cutoff < previous_cutoff:
                    neighbor_data = rescreen(cutoff, neighbor_data)
                    previous_cutoff = cutoff
                atomic_energies += pot.atomic_energies(element_idxs, neighbor_data)
        else:
            for pot in self.potentials:
                if not isinstance(pot, AEVPotential) and not include_non_aev_potentials:
                    continue
                cutoff = pot.cutoff
                if pot.cutoff < previous_cutoff:
                    cutoff
                    neighbor_data = rescreen(cutoff, neighbor_data)
                    previous_cutoff = cutoff
                atomic_energies += pot.atomic_energies(element_idxs, neighbor_data)

        if shift_energy:
            atomic_energies += self.energy_shifter.atomic_energies(element_idxs)

        if average:
            atomic_energies = atomic_energies.mean(dim=0)
        return SpeciesEnergies(species_coordinates[0], atomic_energies)

    # NOTE: members_energies does not need to be overriden, it works correctly as is
    def __getitem__(self, index: int) -> tpx.Self:
        non_aev_potentials = [
            p for p in self.potentials if not isinstance(p, AEVPotential)
        ]
        return type(self)(
            aev_computer=self.aev_computer,
            neural_networks=self.neural_networks.member(index),
            energy_shifter=self.energy_shifter,
            elements=self.get_chemical_symbols(),
            periodic_table_index=self.periodic_table_index,
            pairwise_potentials=non_aev_potentials,
        )


def ANI1x(**kwargs) -> BuiltinModel:
    from . import assembler  # noqa

    return assembler.ANI1x(**kwargs)


def ANI1ccx(**kwargs) -> BuiltinModel:
    from . import assembler  # noqa

    return assembler.ANI1ccx(**kwargs)


def ANI2x(**kwargs) -> BuiltinModel:
    from . import assembler  # noqa

    return assembler.ANI2x(**kwargs)


def ANIala(**kwargs) -> BuiltinModel:
    from . import assembler  # noqa

    return assembler.ANIala(**kwargs)


def ANIdr(**kwargs) -> BuiltinModel:
    from . import assembler  # noqa

    return assembler.ANIdr(**kwargs)
