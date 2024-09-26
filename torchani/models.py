"""The ANI model zoo that stores public ANI models.

The ANI "Model Zoo" stores all published ANI models as built-in models. Some of
the ANI models have been published in specific articles, and some have been
published in TorchANI 2.0.

If you use a built-in model in your work please cite the corresponding article.
If you discover any problem, be it a bug, a performance problem, or incorrect
behavior in some region of chemical space, please post an issue in GitHub. The
TorchANI developers will attempt to address issues and document problematic
behavior for of the models.

The parameters for the models are automatically downloaded the first time they
are used. If this is an issue for your application we recommend you
pre-download the parameters by instantiating the models before use.

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

For more detailed example of usage consult the examples documentation

.. code-block:: python
    import torchani

    model = torchani.models.ANI1x()
    # compute energy using a model ensemble
    _, energies = ani1x((species, coordinates))

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
    SpeciesEnergiesAtomicCharges,
    SpeciesEnergiesQBC,
    AtomicStdev,
    SpeciesForces,
    ForceStdev,
    ForceMagnitudes,
)
from torchani.electro import ChargeNormalizer
from torchani.atomics import AtomicContainer
from torchani.nn import SpeciesConverter
from torchani.constants import PERIODIC_TABLE, ATOMIC_NUMBER
from torchani.aev import AEVComputer
from torchani.potentials import (
    NNPotential,
    SeparateChargesNNPotential,
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
        symbols: tp.Sequence[str],
        aev_computer: AEVComputer,
        neural_networks: AtomicContainer,
        energy_shifter: EnergyAdder,
        periodic_table_index: bool = True,
    ):
        super().__init__()

        self.aev_computer = aev_computer
        self.neural_networks = neural_networks
        self.energy_shifter = energy_shifter
        device = self.energy_shifter.self_energies.device
        self.species_converter = SpeciesConverter(symbols).to(device)

        self.periodic_table_index = periodic_table_index
        numbers = torch.tensor([ATOMIC_NUMBER[e] for e in symbols], dtype=torch.long)
        self.register_buffer("atomic_numbers", numbers)

        # checks are performed to make sure all modules passed support the
        # correct number of species
        assert len(self.energy_shifter.self_energies) == len(self.atomic_numbers)
        assert self.aev_computer.num_species == len(self.atomic_numbers)
        assert self.neural_networks.num_species == len(self.atomic_numbers)

    def to_infer_model(self, use_mnp: bool = False) -> tpx.Self:
        """Convert the neural networks module of the model into a module
        optimized for inference.

        Assumes that the atomic networks consist of an MLP with
        torchani.utils.TightCELU activation functions.
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
        total_charge: float = 0.0,
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
        assert total_charge == 0.0, "Model only supports neutral molecules"
        species_coordinates = self._maybe_convert_species(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        species, energies = self.neural_networks(species_aevs)
        return SpeciesEnergies(species, energies + self.energy_shifter(species))

    @torch.jit.export
    def from_neighborlist(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        neighbor_idxs: Tensor,
        shift_values: Tensor,
        total_charge: float = 0.0,
        input_needs_screening: bool = True,
    ) -> SpeciesEnergies:
        # This entrypoint supports input from an external neighborlist
        species, coordinates = self._maybe_convert_species(species_coordinates)
        # Check shapes
        num_molecules, num_atoms = species.shape
        assert coordinates.shape == (num_molecules, num_atoms, 3)
        cutoff = self.aev_computer.radial_terms.cutoff
        neighbors = self.aev_computer.neighborlist.process_external_input(
            species,
            coordinates,
            neighbor_idxs,
            shift_values,
            cutoff,
            input_needs_screening,
        )
        aevs = self.aev_computer._compute_aev(species, neighbors=neighbors)
        species, energies = self.neural_networks((species, aevs))
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
        only_trainable_potentials: bool = False,
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

    def ase(
        self,
        overwrite: bool = False,
        stress_partial_fdotr: bool = False,
        stress_numerical: bool = False,
        jit: bool = False,
    ):
        """Get an ASE Calculator using this ANI model

        Arguments:
            kwargs: ase.Calculator kwargs

        Returns:
            calculator (:class:`ase.Calculator`): A calculator to be used with ASE
        """
        from torchani.ase import Calculator

        return Calculator(
            torch.jit.script(self) if jit else self,
            overwrite=overwrite,
            stress_partial_fdotr=stress_partial_fdotr,
            stress_numerical=stress_numerical,
        )

    def __getitem__(self, index: int) -> tpx.Self:
        return type(self)(
            symbols=self.get_chemical_symbols(),
            aev_computer=self.aev_computer,
            neural_networks=self.neural_networks.member(index),
            energy_shifter=self.energy_shifter,
            periodic_table_index=self.periodic_table_index,
        )

    @torch.jit.export
    def members_energies(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        only_trainable_potentials: bool = False,
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
            only_trainable_potentials=only_trainable_potentials,
        )
        return SpeciesEnergies(species, members_energies.sum(-1))

    @torch.jit.unused
    def members_forces(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        average: bool = False,
        only_trainable_potentials: bool = False,
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
            species_coordinates,
            cell,
            pbc,
            only_trainable_potentials=only_trainable_potentials,
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
        only_trainable_potentials: bool = False,
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
        species, energies = self.members_energies(
            species_coordinates,
            cell,
            pbc,
            only_trainable_potentials=only_trainable_potentials,
        )

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
        self,
        symbols: tp.Sequence[str],
        aev_computer: AEVComputer,
        neural_networks: AtomicContainer,
        energy_shifter: EnergyAdder,
        pairwise_potentials: tp.Iterable[PairPotential] = tuple(),
        periodic_table_index: bool = True,
    ):
        super().__init__(
            symbols=symbols,
            aev_computer=aev_computer,
            neural_networks=neural_networks,
            energy_shifter=energy_shifter,
            periodic_table_index=periodic_table_index,
        )
        potentials: tp.List[Potential] = list(pairwise_potentials)
        aev_potential = NNPotential(self.aev_computer, self.neural_networks)
        potentials.append(aev_potential)

        # We want to check the cutoffs of the potentials, and sort them
        # in order of decreasing cutoffs. this way the potential with the LARGEST
        # cutoff is computed first, then sequentially things that need smaller
        # cutoffs are computed.
        potentials = sorted(potentials, key=lambda x: x.cutoff, reverse=True)
        self.potentials = torch.nn.ModuleList(potentials)

    # unfortunately this is an UGLY workaround to a torchscript bug
    @torch.jit.export
    def _recast_long_buffers(self) -> None:
        self.species_converter.conv_tensor = self.species_converter.conv_tensor.to(
            dtype=torch.long
        )
        self.aev_computer.triu_index = self.aev_computer.triu_index.to(dtype=torch.long)
        self.aev_computer.neighborlist._recast_long_buffers()

    # TODO: Remove code repetition
    @torch.jit.export
    def from_neighborlist(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        neighbor_idxs: Tensor,
        shift_values: Tensor,
        total_charge: float = 0.0,
        input_needs_screening: bool = True,
    ) -> SpeciesEnergies:
        # This entrypoint supports input from an external neighborlist
        element_idxs, coordinates = self._maybe_convert_species(species_coordinates)
        # Check shapes
        num_molecules, num_atoms = element_idxs.shape
        assert coordinates.shape == (num_molecules, num_atoms, 3)

        previous_cutoff = self.potentials[0].cutoff
        neighbors = self.aev_computer.neighborlist.process_external_input(
            element_idxs,
            coordinates,
            neighbor_idxs,
            shift_values,
            previous_cutoff,
            input_needs_screening,
        )
        energies = torch.zeros(
            num_molecules, device=element_idxs.device, dtype=coordinates.dtype
        )
        for pot in self.potentials:
            cutoff = pot.cutoff
            if cutoff < previous_cutoff:
                neighbors = rescreen(cutoff, neighbors)
                previous_cutoff = cutoff
            energies += pot(element_idxs, neighbors)
        return SpeciesEnergies(
            element_idxs, energies + self.energy_shifter(element_idxs)
        )

    def forward(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        total_charge: float = 0.0,
    ) -> SpeciesEnergies:
        assert total_charge == 0.0, "Model only supports neutral molecules"
        element_idxs, coordinates = self._maybe_convert_species(species_coordinates)

        previous_cutoff = self.potentials[0].cutoff
        neighbors = self.aev_computer.neighborlist(
            element_idxs, coordinates, previous_cutoff, cell, pbc
        )
        energies = torch.zeros(
            element_idxs.shape[0], device=element_idxs.device, dtype=coordinates.dtype
        )
        for pot in self.potentials:
            cutoff = pot.cutoff
            if cutoff < previous_cutoff:
                neighbors = rescreen(cutoff, neighbors)
                previous_cutoff = cutoff
            energies += pot(element_idxs, neighbors)
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
        only_trainable_potentials: bool = False,
    ) -> SpeciesEnergies:
        element_idxs, coordinates = self._maybe_convert_species(species_coordinates)
        previous_cutoff = self.potentials[0].cutoff
        neighbors = self.aev_computer.neighborlist(
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
        for pot in self.potentials:
            if only_trainable_potentials and not pot.is_trainable:
                pass  # JIT does not support "continue"
            else:
                cutoff = pot.cutoff
                if pot.cutoff < previous_cutoff:
                    cutoff
                    neighbors = rescreen(cutoff, neighbors)
                    previous_cutoff = cutoff
                atomic_energies += pot.atomic_energies(element_idxs, neighbors)

        if shift_energy:
            atomic_energies += self.energy_shifter.atomic_energies(element_idxs)

        if average:
            atomic_energies = atomic_energies.mean(dim=0)
        return SpeciesEnergies(species_coordinates[0], atomic_energies)

    def __getitem__(self, index: int) -> tpx.Self:
        return type(self)(
            symbols=self.get_chemical_symbols(),
            aev_computer=self.aev_computer,
            neural_networks=self.neural_networks.member(index),
            energy_shifter=self.energy_shifter,
            periodic_table_index=self.periodic_table_index,
            pairwise_potentials=[p for p in self.potentials if not p.is_trainable],
        )


class PairPotentialsChargesModel(PairPotentialsModel):
    r"""
    Calculates energies and atomic charges. Charge networks share the input
    features with the energy networks, but are otherwise fully independent from them.

    WARNING: This model is an experimental feature and will probably be removed
    in the future.
    """

    def __init__(
        self,
        symbols: tp.Sequence[str],
        aev_computer: AEVComputer,
        neural_networks: AtomicContainer,
        energy_shifter: EnergyAdder,
        pairwise_potentials: tp.Iterable[PairPotential] = tuple(),
        periodic_table_index: bool = True,
        charge_networks: tp.Optional[AtomicContainer] = None,
        charge_normalizer: tp.Optional[ChargeNormalizer] = None,
    ):
        if charge_networks is None:
            raise NotImplementedError(
                "Model with fused charge-energy networks not yet implemented"
            )
        super().__init__(
            symbols=symbols,
            aev_computer=aev_computer,
            neural_networks=neural_networks,
            energy_shifter=energy_shifter,
            pairwise_potentials=pairwise_potentials,
            periodic_table_index=periodic_table_index,
        )
        self.charge_networks = charge_networks
        self.charge_normalizer = charge_normalizer
        charges_nnp = SeparateChargesNNPotential(
            self.aev_computer,
            self.neural_networks,
            self.charge_networks,
            self.charge_normalizer,
        )
        # Check which index has the NNPotential
        potentials = [pot for pot in self.potentials]
        for j, pot in enumerate(potentials):
            if pot.is_trainable:
                break
        # Replace with the ChargesNNPotential
        potentials[j] = charges_nnp
        # Re-register the ModuleList
        self.potentials = torch.nn.ModuleList(potentials)

    # TODO: Remove code duplication
    @torch.jit.export
    def energies_and_atomic_charges_from_neighborlist(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        neighbor_idxs: Tensor,
        shift_values: Tensor,
        total_charge: float = 0.0,
        input_needs_screening: bool = True,
    ) -> SpeciesEnergiesAtomicCharges:
        # This entrypoint supports input from an external neighborlist
        element_idxs, coordinates = self._maybe_convert_species(species_coordinates)
        # Check shapes
        num_molecules, num_atoms = element_idxs.shape
        assert coordinates.shape == (num_molecules, num_atoms, 3)
        assert total_charge == 0.0, "Model only supports neutral molecules"
        previous_cutoff = self.potentials[0].cutoff
        neighbors = self.aev_computer.neighborlist.process_external_input(
            element_idxs,
            coordinates,
            neighbor_idxs,
            shift_values,
            previous_cutoff,
            input_needs_screening,
        )
        energies = torch.zeros(
            num_molecules, device=element_idxs.device, dtype=coordinates.dtype
        )
        atomic_charges = torch.zeros(
            (num_molecules, num_atoms),
            device=element_idxs.device,
            dtype=coordinates.dtype,
        )
        for pot in self.potentials:
            cutoff = pot.cutoff
            if cutoff < previous_cutoff:
                neighbors = rescreen(cutoff, neighbors)
                previous_cutoff = cutoff
            if pot.is_trainable:
                output = pot.energies_and_atomic_charges(
                    element_idxs,
                    neighbors,
                    ghost_flags=None,
                    total_charge=total_charge,
                )
                energies += output.energies
                atomic_charges += output.atomic_charges
            else:
                energies += pot(element_idxs, neighbors)
        return SpeciesEnergiesAtomicCharges(
            element_idxs, energies + self.energy_shifter(element_idxs), atomic_charges
        )

    @torch.jit.export
    def energies_and_atomic_charges(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        total_charge: float = 0.0,
    ) -> SpeciesEnergiesAtomicCharges:
        assert total_charge == 0.0, "Model only supports neutral molecules"
        element_idxs, coordinates = self._maybe_convert_species(species_coordinates)
        previous_cutoff = self.potentials[0].cutoff
        neighbor_data = self.aev_computer.neighborlist(
            element_idxs, coordinates, previous_cutoff, cell, pbc
        )
        energies = torch.zeros(
            element_idxs.shape[0], device=element_idxs.device, dtype=coordinates.dtype
        )
        atomic_charges = torch.zeros(
            element_idxs.shape, device=element_idxs.device, dtype=coordinates.dtype
        )
        for pot in self.potentials:
            cutoff = pot.cutoff
            if cutoff < previous_cutoff:
                neighbor_data = rescreen(cutoff, neighbor_data)
                previous_cutoff = cutoff
            if pot.is_trainable:
                output = pot.energies_and_atomic_charges(
                    element_idxs,
                    neighbor_data,
                    ghost_flags=None,
                    total_charge=total_charge,
                )
                energies += output.energies
                atomic_charges += output.atomic_charges
            else:
                energies += pot(element_idxs, neighbor_data)
        return SpeciesEnergiesAtomicCharges(
            element_idxs, energies + self.energy_shifter(element_idxs), atomic_charges
        )

    def __getitem__(self, index: int) -> tpx.Self:
        return type(self)(
            symbols=self.get_chemical_symbols(),
            aev_computer=self.aev_computer,
            neural_networks=self.neural_networks.member(index),
            charge_networks=self.charge_networks,
            charge_normalizer=self.charge_normalizer,
            energy_shifter=self.energy_shifter,
            periodic_table_index=self.periodic_table_index,
            pairwise_potentials=[p for p in self.potentials if not p.is_trainable],
        )


def ANI1x(**kwargs) -> BuiltinModel:
    from torchani.assembler import ANI1x as build

    return build(**kwargs)


def ANI1ccx(**kwargs) -> BuiltinModel:
    from torchani.assembler import ANI1ccx as build

    return build(**kwargs)


def ANI2x(**kwargs) -> BuiltinModel:
    from torchani.assembler import ANI2x as build

    return build(**kwargs)


def ANIala(**kwargs) -> BuiltinModel:
    from torchani.assembler import ANIala as build

    return build(**kwargs)


def ANIdr(**kwargs) -> BuiltinModel:
    from torchani.assembler import ANIdr as build

    return build(**kwargs)


def ANImbis(**kwargs) -> BuiltinModel:
    from torchani.assembler import ANImbis as build

    return build(**kwargs)
