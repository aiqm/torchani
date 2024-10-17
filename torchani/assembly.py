r"""
Construction of ANI-style models and definition of their architecture

The assembler builds ANI-style models from the different necessary parts, in
such a way that all parts interact in the correct way and there are no
compatibility issues among them.

An ANI-style model consists of:

- Featurizer (typically a AEVComputer, which supports custom cuda ops, or subclass)
- Container for atomic networks (typically ANIModel or subclass)
- Atomic Networks Dict {"H": torch.nn.Module(), "C": torch.nn.Module, ...}
- Self Energies Dict (In Ha) {"H": -12.0, "C": -75.0, ...}
- Shifter (typically EnergyAdder)

An energy-predicting model may have PairPotentials (RepulsionXTB,
TwoBodyDispersion, VDW potential, Coulombic, etc.)

Each of the potentials has their own cutoff, and the Featurizer has two
cutoffs, an angular and a radial ona (the radial cutoff must be larger than
the angular cutoff, and it is recommended that the angular cutoff is kept
small, 3.5 Ang or less).

These pieces are assembled into a subclass of ANI
"""

import warnings
import functools
import math
from dataclasses import dataclass
from collections import OrderedDict
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
from torchani import atomics
from torchani.neighbors import parse_neighborlist, NeighborlistArg
from torchani.cutoffs import parse_cutoff_fn, Cutoff, CutoffArg
from torchani.aev import AEVComputer, StandardAngular, StandardRadial
from torchani.aev.terms import (
    RadialTermArg,
    AngularTermArg,
    parse_radial_term,
    parse_angular_term,
)
from torchani.neighbors import rescreen, NeighborData
from torchani.electro import ChargeNormalizer
from torchani.nn import SpeciesConverter, ANIModel, DummyANIModel, Ensemble
from torchani.atomics import AtomicContainer, AtomicNetwork, AtomicMakerArg, AtomicMaker
from torchani.constants import GSAES
from torchani.utils import sort_by_element
from torchani.paths import state_dicts_dir
from torchani.constants import PERIODIC_TABLE, ATOMIC_NUMBER
from torchani.potentials import (
    NNPotential,
    SeparateChargesNNPotential,
    MergedChargesNNPotential,
    Potential,
    PairPotential,
    RepulsionXTB,
    TwoBodyDispersionD3,
    EnergyAdder,
)


class ANI(torch.nn.Module):
    r"""ANI-style neural network interatomic potential"""

    atomic_numbers: Tensor
    periodic_table_index: Final[bool]
    _output_labels: tp.List[str]

    def __init__(
        self,
        symbols: tp.Sequence[str],
        aev_computer: AEVComputer,
        neural_networks: AtomicContainer,
        energy_shifter: EnergyAdder,
        pairwise_potentials: tp.Iterable[PairPotential] = (),
        periodic_table_index: bool = True,
        output_labels: tp.Sequence[str] = ("energies",),
    ):
        super().__init__()

        # NOTE: Keep these refs for later usage
        self.aev_computer = aev_computer
        self.neural_networks = neural_networks
        self.neighborlist = self.aev_computer.neighborlist
        self._output_labels = list(output_labels)

        device = energy_shifter.self_energies.device
        self.energy_shifter = energy_shifter
        self.species_converter = SpeciesConverter(symbols).to(device)

        potentials: tp.List[Potential] = list(pairwise_potentials)
        potentials.append(NNPotential(self.aev_computer, self.neural_networks))
        self.potentials_len = len(potentials)

        # Sort potentials in order of decresing cutoff. The potential with the
        # LARGEST cutoff is computed first, then sequentially things that need
        # SMALLER cutoffs are computed.
        potentials = sorted(potentials, key=lambda x: x.cutoff, reverse=True)
        self.potentials = torch.nn.ModuleList(potentials)

        self.periodic_table_index = periodic_table_index
        numbers = torch.tensor([ATOMIC_NUMBER[e] for e in symbols], dtype=torch.long)
        self.register_buffer("atomic_numbers", numbers)

        # Make sure all modules passed support the correct num species
        assert len(self.energy_shifter.self_energies) == len(self.atomic_numbers)
        assert self.aev_computer.num_species == len(self.atomic_numbers)
        assert self.neural_networks.num_species == len(self.atomic_numbers)

    @torch.jit.export
    def sp(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        total_charge: float = 0.0,
        ensemble_average: bool = True,
        shift_energy: bool = True,
    ) -> tp.Dict[str, Tensor]:
        _, energies = self(
            species_coordinates, cell, pbc, total_charge, ensemble_average, shift_energy
        )
        return {self._output_labels[0]: energies}

    def forward(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        total_charge: float = 0.0,
        ensemble_average: bool = True,
        shift_energy: bool = True,
    ) -> SpeciesEnergies:
        """Calculate energies for a minibatch of molecules

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set
                to None if PBC is not enabled
            total_charge: (float): The total charge of the molecules. Only
                the scalar 0.0 is currently supported.
            ensemble_average (bool): If True (default), return the average
                over all models in the ensemble (output shape ``(C, A)``), otherwise
                return one atomic energy per model (output shape ``(M, C, A)``).
            shift_energy (bool): Add the a constant energy shift to the returned
                energies. ``True`` by default.

        Returns:
            species_energies: tuple of tensors, species and energies for the
                given configurations
        """
        assert total_charge == 0.0, "Model only supports neutral molecules"

        # Unoptimized path to obtain member energies, and eventually QBC
        if not ensemble_average:
            elem_idxs, energies = self.atomic_energies(
                species_coordinates,
                cell=cell,
                pbc=pbc,
                total_charge=total_charge,
                ensemble_average=False,
                shift_energy=shift_energy,
            )
            return SpeciesEnergies(elem_idxs, energies.sum(-1))

        elem_idxs, coords = self._maybe_convert_species(species_coordinates)
        assert coords.shape[:-1] == elem_idxs.shape
        assert coords.shape[-1] == 3

        # Optimized path, use merged Neighborlist-AEVomputer
        if self.potentials_len == 1:
            species, energies = self.neural_networks(
                self.aev_computer((elem_idxs, coords), cell=cell, pbc=pbc)
            )
            return SpeciesEnergies(species, energies + self.energy_shifter(species))

        # Unoptimized path
        largest_cutoff = self.potentials[0].cutoff
        neighbors = self.neighborlist(elem_idxs, coords, largest_cutoff, cell, pbc)
        energies = self._energy_of_pots(elem_idxs, coords, largest_cutoff, neighbors)

        if shift_energy:
            energies += self.energy_shifter(elem_idxs)
        return SpeciesEnergies(elem_idxs, energies)

    @torch.jit.export
    def from_neighborlist(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        neighbor_idxs: Tensor,
        shift_values: Tensor,
        total_charge: float = 0.0,
        ensemble_average: bool = True,
        shift_energy: bool = True,
        input_needs_screening: bool = True,
    ) -> SpeciesEnergies:
        r"""
        This entrypoint supports input from an external neighborlist
        """
        assert total_charge == 0.0, "Model only supports neutral molecules"
        elem_idxs, coords = self._maybe_convert_species(species_coordinates)
        assert coords.shape[:-1] == elem_idxs.shape
        assert coords.shape[-1] == 3
        largest_cutoff = self.potentials[0].cutoff
        neighbors = self.neighborlist.process_external_input(
            elem_idxs,
            coords,
            neighbor_idxs,
            shift_values,
            largest_cutoff,
            input_needs_screening,
        )
        if not ensemble_average:
            energies = self._atomic_energy_of_pots(
                elem_idxs, coords, largest_cutoff, neighbors
            ).mean(dim=1)

        energies = self._energy_of_pots(elem_idxs, coords, largest_cutoff, neighbors)
        if shift_energy:
            dummy = NeighborData(torch.empty(0), torch.empty(0), torch.empty(0))
            energies += self.energy_shifter.atomic_energies(
                elem_idxs, dummy, None, ensemble_average=ensemble_average
            ).sum(dim=-1)
        return SpeciesEnergies(elem_idxs, energies)

    @torch.jit.export
    def atomic_energies(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        total_charge: float = 0.0,
        ensemble_average: bool = True,
        shift_energy: bool = True,
    ) -> SpeciesEnergies:
        r"""Calculate predicted atomic energies of all atoms in a molecule

        Arguments and return value are the same as that of forward(), but
        the returned energies have shape (molecules, atoms)
        """
        assert total_charge == 0.0, "Model only supports neutral molecules"
        elem_idxs, coords = self._maybe_convert_species(species_coordinates)

        # Optimized path, go through the merged Neighborlist-AEVomputer only
        if self.potentials_len == 1:
            atomic_energies = self.neural_networks.members_atomic_energies(
                self.aev_computer((elem_idxs, coords), cell=cell, pbc=pbc)
            )
        # Iterate over all potentials
        else:
            largest_cutoff = self.potentials[0].cutoff
            neighbors = self.neighborlist(elem_idxs, coords, largest_cutoff, cell, pbc)
            atomic_energies = self._atomic_energy_of_pots(
                elem_idxs, coords, largest_cutoff, neighbors
            )

        if shift_energy:
            atomic_energies += self.energy_shifter.atomic_energies(
                elem_idxs, ensemble_average=False
            )

        if ensemble_average:
            atomic_energies = atomic_energies.mean(dim=0)
        return SpeciesEnergies(species_coordinates[0], atomic_energies)

    def to_infer_model(self, use_mnp: bool = False) -> tpx.Self:
        r"""Convert the neural networks module of the model into a module
        optimized for inference.

        Assumes that the atomic networks are multi layer perceptrons (MLPs)
        with torchani.utils.TightCELU activation functions.
        """
        self.neural_networks = self.neural_networks.to_infer_model(use_mnp=use_mnp)
        return self

    def ase(
        self,
        overwrite: bool = False,
        stress_partial_fdotr: bool = False,
        stress_numerical: bool = False,
        jit: bool = False,
    ):
        r"""Get an ASE Calculator using this ANI model

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

    @torch.jit.unused
    def get_chemical_symbols(self) -> tp.Tuple[str, ...]:
        return tuple(PERIODIC_TABLE[z] for z in self.atomic_numbers)

    @torch.jit.unused
    def strip_non_trainable_potentials(self):
        r"""
        Remove all potentials in the network that are non-trainable
        """
        self.potentials = torch.nn.ModuleList(
            [p for p in self.potentials if p.is_trainable]
        )

    def __len__(self):
        return self.neural_networks.num_networks

    def __getitem__(self, index: int) -> tpx.Self:
        return type(self)(
            symbols=self.get_chemical_symbols(),
            aev_computer=self.aev_computer,
            neural_networks=self.neural_networks.member(index),
            energy_shifter=self.energy_shifter,
            pairwise_potentials=[
                p for p in self.potentials if not isinstance(p, NNPotential)
            ],
            periodic_table_index=self.periodic_table_index,
            output_labels=self._output_labels,
        )

    def _atomic_energy_of_pots(
        self,
        elem_idxs: Tensor,
        coords: Tensor,
        previous_cutoff: float,
        neighbors: NeighborData,
    ) -> Tensor:
        # Add extra axis, since potentials return atomic E of shape (memb, N, A)
        shape = (
            self.neural_networks.num_networks,
            elem_idxs.shape[0],
            elem_idxs.shape[1],
        )
        energies = torch.zeros(shape, dtype=coords.dtype, device=coords.device)
        for pot in self.potentials:
            cutoff = pot.cutoff
            if cutoff < previous_cutoff:
                neighbors = rescreen(cutoff, neighbors)
                previous_cutoff = cutoff
            energies += pot.atomic_energies(elem_idxs, neighbors, _coordinates=coords)
        return energies

    def _energy_of_pots(
        self,
        elem_idxs: Tensor,
        coords: Tensor,
        previous_cutoff: float,
        neighbors: NeighborData,
    ) -> Tensor:
        energies = torch.zeros(
            elem_idxs.shape[0], dtype=coords.dtype, device=coords.device
        )
        for pot in self.potentials:
            cutoff = pot.cutoff
            if cutoff < previous_cutoff:
                neighbors = rescreen(cutoff, neighbors)
                previous_cutoff = cutoff
            energies += pot(elem_idxs, neighbors, _coordinates=coords)
        return energies

    # Needed for bw compatibility
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        old_keys = list(state_dict.keys())
        if not any(k.startswith("potentials") for k in old_keys):
            for oldk in old_keys:
                if oldk.startswith("aev_computer"):
                    k = f"potentials.0.{oldk}"
                    state_dict[k] = state_dict[oldk]
                if oldk.startswith("neural_networks"):
                    k = f"potentials.0.{oldk}"
                    state_dict[k] = state_dict[oldk]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.jit.export
    def _maybe_convert_species(
        self, species_coordinates: tp.Tuple[Tensor, Tensor]
    ) -> tp.Tuple[Tensor, Tensor]:
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        if (species_coordinates[0] >= self.aev_computer.num_species).any():
            raise ValueError(f"Unknown species found in {species_coordinates[0]}")
        return species_coordinates

    # Unfortunately this is an UGLY workaround for a torchscript bug
    @torch.jit.export
    def _recast_long_buffers(self) -> None:
        self.species_converter.conv_tensor = self.species_converter.conv_tensor.to(
            dtype=torch.long
        )
        self.aev_computer.triu_index = self.aev_computer.triu_index.to(dtype=torch.long)
        self.neighborlist._recast_long_buffers()

    @torch.jit.unused
    def members_forces(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> SpeciesForces:
        """Calculates predicted forces from ensemble members

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not
                enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set
                to none if PBC is not enabled

        Returns:
            SpeciesForces: species, molecular energies, and atomic forces
                predicted by an ensemble of neural network models
        """
        coordinates = species_coordinates[1].requires_grad_()
        members_energies = self(
            species_coordinates,
            cell,
            pbc,
            total_charge=0.0,
            ensemble_average=False,
            shift_energy=True,
        ).energies
        _forces = []
        for energy in members_energies:
            _forces.append(
                -torch.autograd.grad(energy.sum(), coordinates, retain_graph=True)[0]
            )
        forces = torch.stack(_forces, dim=0)
        return SpeciesForces(species_coordinates[0], members_energies, forces)

    @torch.jit.export
    def energies_qbcs(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        unbiased: bool = True,
        total_charge: float = 0.0,
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
        species, energies = self(
            species_coordinates,
            cell,
            pbc,
            total_charge=0.0,
            ensemble_average=False,
            shift_energy=True,
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
        ensemble_average: bool = False,
        shift_energy: bool = False,
        unbiased: bool = True,
    ) -> AtomicStdev:
        r"""Returns standard deviation of atomic energies across an ensemble

        shift_energy returns the shifted atomic energies according to the model used

        If the model has only 1 network, a value of 0.0 is output for the stdev
        """
        species_coordinates = self._maybe_convert_species(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        atomic_energies = self.neural_networks.members_atomic_energies(species_aevs)

        if shift_energy:
            atomic_energies += self.energy_shifter.atomic_energies(
                species_coordinates[0],
                ensemble_average=ensemble_average,
            )

        if self.neural_networks.num_networks == 1:
            stdev_atomic_energies = torch.zeros_like(atomic_energies).squeeze(0)
        else:
            stdev_atomic_energies = atomic_energies.std(0, unbiased=unbiased)

        if ensemble_average:
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
        ensemble_average: bool = True,
    ) -> ForceMagnitudes:
        """
        Computes the L2 norm of predicted atomic force vectors, returning magnitudes,
        averaged by default.

        Args:
            species_coordinates: minibatch of configurations
            ensemble_average: by default, returns the ensemble average magnitude for
                each atomic force vector
        """
        species, _, members_forces = self.members_forces(species_coordinates, cell, pbc)
        magnitudes = members_forces.norm(dim=-1)
        if ensemble_average:
            magnitudes = magnitudes.mean(0)
        return ForceMagnitudes(species, magnitudes)

    @torch.jit.unused
    def force_qbc(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        ensemble_average: bool = False,
        unbiased: bool = True,
    ) -> ForceStdev:
        """
        Returns the mean force magnitudes and relative range and standard deviation
        of predicted forces across an ensemble of networks.

        Args:
            species_coordinates: minibatch of configurations
            ensemble_average: returns magnitudes predicted by each model by default
            unbiased: whether or not to use Bessel's correction in computing
                the standard deviation, True by default
        """
        species, magnitudes = self.force_magnitudes(
            species_coordinates, cell, pbc, ensemble_average=False
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

        if ensemble_average:
            magnitudes = mean_magnitudes

        return ForceStdev(species, magnitudes, relative_stdev, relative_range)


class ANIq(ANI):
    r"""
    ANI-style model that can calculate both atomic charges and energies

    Charge networks share the input features with the energy networks, and may either
    be fully independent of them, or share weights to some extent.

    The output energies of these models don't necessarily include a coulombic
    term, but they may.
    """

    def __init__(
        self,
        symbols: tp.Sequence[str],
        aev_computer: AEVComputer,
        neural_networks: AtomicContainer,
        energy_shifter: EnergyAdder,
        pairwise_potentials: tp.Iterable[PairPotential] = (),
        periodic_table_index: bool = True,
        charge_networks: tp.Optional[AtomicContainer] = None,
        charge_normalizer: tp.Optional[ChargeNormalizer] = None,
        output_labels: tp.Sequence[str] = ("energies", "atomic_charges"),
    ):
        super().__init__(
            symbols=symbols,
            aev_computer=aev_computer,
            neural_networks=neural_networks,
            energy_shifter=energy_shifter,
            pairwise_potentials=pairwise_potentials,
            periodic_table_index=periodic_table_index,
            output_labels=output_labels,
        )
        nnp: NNPotential
        if charge_networks is None:
            warnings.warn("Merged charges potential is experimental")
            nnp = MergedChargesNNPotential(
                self.aev_computer,
                self.neural_networks,
                charge_normalizer,
            )
        else:
            nnp = SeparateChargesNNPotential(
                self.aev_computer,
                self.neural_networks,
                charge_networks,
                charge_normalizer,
            )

        # Check which index has the NNPotential and replace with ChargesNNPotential
        potentials = [pot for pot in self.potentials]
        for j, pot in enumerate(potentials):
            if isinstance(pot, NNPotential):
                potentials[j] = nnp
                break
        # Re-register the ModuleList
        self.potentials = torch.nn.ModuleList(potentials)

    @torch.jit.export
    def sp(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        total_charge: float = 0.0,
        ensemble_average: bool = True,
        shift_energy: bool = True,
    ) -> tp.Dict[str, Tensor]:
        _, energies, atomic_charges = self.energies_and_atomic_charges(
            species_coordinates, cell, pbc, total_charge
        )
        return {
            self._output_labels[0]: energies,
            self._output_labels[1]: atomic_charges,
        }

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
        element_idxs, coords = self._maybe_convert_species(species_coordinates)
        # Check shapes
        num_molecules, num_atoms = element_idxs.shape
        assert coords.shape == (num_molecules, num_atoms, 3)
        assert total_charge == 0.0, "Model only supports neutral molecules"
        previous_cutoff = self.potentials[0].cutoff
        neighbors = self.neighborlist.process_external_input(
            element_idxs,
            coords,
            neighbor_idxs,
            shift_values,
            previous_cutoff,
            input_needs_screening,
        )
        energies = torch.zeros(
            num_molecules, device=element_idxs.device, dtype=coords.dtype
        )
        atomic_charges = torch.zeros(
            (num_molecules, num_atoms),
            device=element_idxs.device,
            dtype=coords.dtype,
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
                    _coordinates=coords,
                    ghost_flags=None,
                    total_charge=total_charge,
                )
                energies += output.energies
                atomic_charges += output.atomic_charges
            else:
                energies += pot(element_idxs, neighbors, _coordinates=coords)
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
        element_idxs, coords = self._maybe_convert_species(species_coordinates)
        previous_cutoff = self.potentials[0].cutoff
        neighbor_data = self.neighborlist(
            element_idxs, coords, previous_cutoff, cell, pbc
        )
        energies = torch.zeros(
            element_idxs.shape[0], device=element_idxs.device, dtype=coords.dtype
        )
        atomic_charges = torch.zeros(
            element_idxs.shape, device=element_idxs.device, dtype=coords.dtype
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
                    _coordinates=coords,
                    ghost_flags=None,
                    total_charge=total_charge,
                )
                energies += output.energies
                atomic_charges += output.atomic_charges
            else:
                energies += pot(element_idxs, neighbor_data, _coordinates=coords)
        return SpeciesEnergiesAtomicCharges(
            element_idxs, energies + self.energy_shifter(element_idxs), atomic_charges
        )

    def __getitem__(self, index: int) -> tpx.Self:
        for p in self.potentials:
            if isinstance(p, NNPotential):
                charge_normalizer = getattr(p, "charge_normalizer", None)
                charge_networks = getattr(p, "charge_networks", None)
                break

        return type(self)(
            symbols=self.get_chemical_symbols(),
            aev_computer=self.aev_computer,
            neural_networks=self.neural_networks.member(index),
            charge_networks=charge_networks,
            charge_normalizer=charge_normalizer,
            energy_shifter=self.energy_shifter,
            pairwise_potentials=[
                p for p in self.potentials if not isinstance(p, NNPotential)
            ],
            periodic_table_index=self.periodic_table_index,
            output_labels=self._output_labels,
        )


FeaturizerType = tp.Type[AEVComputer]
PairPotentialType = tp.Type[PairPotential]
ContainerType = tp.Type[AtomicContainer]


# "global" cutoff means the global cutoff_fn will be used
# Otherwise, a specific cutoff fn can be specified
class FeaturizerWrapper:
    def __init__(
        self,
        cls: FeaturizerType,
        radial_terms: RadialTermArg,
        angular_terms: AngularTermArg,
        cutoff_fn: CutoffArg = "global",
        extra: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> None:
        self.cls = cls
        self.cutoff_fn = cutoff_fn
        self.radial_terms = parse_radial_term(radial_terms)
        self.angular_terms = parse_angular_term(angular_terms)
        if self.angular_terms.cutoff > self.radial_terms.cutoff:
            raise ValueError("Angular cutoff must be smaller or equal to radial cutoff")
        if self.angular_terms.cutoff <= 0 or self.radial_terms.cutoff <= 0:
            raise ValueError("Cutoffs must be strictly positive")
        self.extra = extra


@dataclass
class PairPotentialWrapper:
    cls: PairPotentialType
    cutoff_fn: CutoffArg = "global"
    cutoff: float = math.inf
    extra: tp.Optional[tp.Dict[str, tp.Any]] = None


class Assembler:
    def __init__(
        self,
        ensemble_size: int = 1,
        symbols: tp.Sequence[str] = (),
        container_type: ContainerType = ANIModel,
        model_type: tp.Type[ANI] = ANI,
        featurizer: tp.Optional[FeaturizerWrapper] = None,
        neighborlist: NeighborlistArg = "full_pairwise",
        periodic_table_index: bool = True,
        output_labels: tp.Sequence[str] = ("energies",),
    ) -> None:
        self._global_cutoff_fn: tp.Optional[Cutoff] = None

        self._neighborlist = parse_neighborlist(neighborlist)
        self._featurizer = featurizer
        self._pairwise_potentials: tp.List[PairPotentialWrapper] = []
        self._output_labels = output_labels

        # This part of the assembler organizes the self-energies, the
        # symbols and the atomic networks
        self._self_energies: tp.Dict[str, float] = {}
        self._fn_for_atomics: tp.Optional[AtomicMaker] = None
        self._fn_for_charges: tp.Optional[AtomicMaker] = None
        self._container_type: ContainerType = container_type
        self._charge_container_type: tp.Optional[ContainerType] = None
        self._charge_normalizer: tp.Optional[ChargeNormalizer] = None
        self._symbols: tp.Tuple[str, ...] = tuple(symbols)
        self._ensemble_size: int = ensemble_size

        # This is the general container for all the parts of the model
        self._model_type: tp.Type[ANI] = model_type

        # This is a deprecated feature, it should probably not be used
        self.periodic_table_index = periodic_table_index

    def _check_symbols(self, symbols: tp.Optional[tp.Iterable[str]] = None) -> None:
        if not self.symbols:
            raise ValueError(
                "Please set symbols before setting the gsaes as self energies"
            )
        if symbols is not None:
            if set(self.symbols) != set(symbols):
                raise ValueError(
                    f"Passed symbols don't match supported elements {self._symbols}"
                )

    @property
    def ensemble_size(self) -> int:
        return self._ensemble_size

    @ensemble_size.setter
    def ensemble_size(self, value: int) -> None:
        if value < 0:
            raise ValueError("Ensemble size must be positive")
        self._ensemble_size = value

    @property
    def elements_num(self) -> int:
        return len(self._symbols)

    @property
    def symbols(self) -> tp.Tuple[str, ...]:
        return self._symbols

    def set_symbols(self, symbols: tp.Sequence[str], auto_sort: bool = True) -> None:
        if auto_sort:
            self._symbols = sort_by_element(symbols)
        else:
            self._symbols = tuple(symbols)

    @property
    def fn_for_atomics(self) -> AtomicMaker:
        if self._fn_for_atomics is None:
            raise RuntimeError(
                "fn for atomics is not set, please call 'set_atomic_networks'"
            )
        return self._fn_for_atomics

    @property
    def fn_for_charges(self) -> AtomicMaker:
        if self._fn_for_charges is None:
            raise RuntimeError(
                "fn for charges is not set, please call 'set_charge_networks'"
            )
        return self._fn_for_charges

    @property
    def self_energies(self) -> tp.Dict[str, float]:
        if not self._self_energies:
            raise RuntimeError("Self energies have not been set")
        return self._self_energies

    @self_energies.setter
    def self_energies(self, value: tp.Mapping[str, float]) -> None:
        self._check_symbols(value.keys())
        self._self_energies = {k: v for k, v in value.items()}

    def set_zeros_as_self_energies(self) -> None:
        self._check_symbols()
        self.self_energies = {s: 0.0 for s in self.symbols}

    def set_gsaes_as_self_energies(
        self,
        lot: str = "",
        functional: str = "",
        basis_set: str = "",
    ) -> None:
        self._check_symbols()
        if (functional and basis_set) and not lot:
            lot = f"{functional}-{basis_set}"
        elif not (functional or basis_set) and lot:
            pass
        else:
            raise ValueError(
                "Incorrect specification."
                " Either specify *only* lot (preferred)"
                " or *both* functional *and* basis_set"
            )
        gsaes = GSAES[lot.lower()]
        self.self_energies = {s: gsaes[s] for s in self.symbols}

    def set_atomic_networks(
        self,
        container_type: ContainerType,
        fn: AtomicMaker,
    ) -> None:
        self._container_type = container_type
        self._fn_for_atomics = fn

    def set_charge_networks(
        self,
        container_type: ContainerType,
        fn: AtomicMaker,
        normalizer: tp.Optional[ChargeNormalizer] = None,
    ) -> None:
        if not issubclass(self._model_type, ANIq):
            raise ValueError("Model must be a subclass of ANIq to use charge networks")
        self._charge_container_type = container_type
        self._charge_normalizer = normalizer
        self._fn_for_charges = fn

    def set_featurizer(
        self,
        featurizer_type: FeaturizerType,
        angular_terms: AngularTermArg,
        radial_terms: RadialTermArg,
        cutoff_fn: CutoffArg = "global",
        extra: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> None:
        self._featurizer = FeaturizerWrapper(
            featurizer_type,
            cutoff_fn=cutoff_fn,
            angular_terms=angular_terms,
            radial_terms=radial_terms,
            extra=extra,
        )

    def set_neighborlist(
        self,
        neighborlist: NeighborlistArg,
    ) -> None:
        self._neighborlist = parse_neighborlist(neighborlist)

    def set_global_cutoff_fn(
        self,
        cutoff_fn: CutoffArg,
    ) -> None:
        self._global_cutoff_fn = parse_cutoff_fn(cutoff_fn)

    def add_pairwise_potential(
        self,
        pair_type: PairPotentialType,
        cutoff: float = math.inf,
        cutoff_fn: CutoffArg = "global",
        extra: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> None:
        self._pairwise_potentials.append(
            PairPotentialWrapper(
                pair_type,
                cutoff=cutoff,
                cutoff_fn=cutoff_fn,
                extra=extra,
            )
        )

    def build_atomic_networks(
        self,
        fn_for_networks: AtomicMaker,
        in_dim: int,
    ) -> tp.OrderedDict[str, AtomicNetwork]:
        return OrderedDict([(s, fn_for_networks(s, in_dim)) for s in self.symbols])

    def assemble(self) -> ANI:
        if not self.symbols:
            raise RuntimeError("Symbols not set. Call 'set_symbols()' before assembly")
        if self._featurizer is None:
            raise RuntimeError(
                "Featurizer not set. Call 'set_featurizer' before assembly"
            )

        feat_cutoff_fn = parse_cutoff_fn(
            self._featurizer.cutoff_fn, self._global_cutoff_fn
        )

        self._featurizer.angular_terms.cutoff_fn = feat_cutoff_fn
        self._featurizer.radial_terms.cutoff_fn = feat_cutoff_fn
        feat_kwargs = {}
        if self._featurizer.extra is not None:
            feat_kwargs.update(self._featurizer.extra)

        featurizer = self._featurizer.cls(
            neighborlist=self._neighborlist,
            cutoff_fn=feat_cutoff_fn,
            angular_terms=self._featurizer.angular_terms,
            radial_terms=self._featurizer.radial_terms,
            num_species=self.elements_num,
            **feat_kwargs,  # type: ignore
        )
        neural_networks: AtomicContainer
        if self.ensemble_size > 1:
            containers = []
            for j in range(self.ensemble_size):
                containers.append(
                    self._container_type(
                        self.build_atomic_networks(
                            self.fn_for_atomics, featurizer.aev_length
                        )
                    )
                )
            neural_networks = Ensemble(containers)
        else:
            neural_networks = self._container_type(
                self.build_atomic_networks(self.fn_for_atomics, featurizer.aev_length)
            )

        charge_networks: tp.Optional[AtomicContainer] = None
        if self._charge_container_type is not None:
            charge_networks = self._charge_container_type(
                self.build_atomic_networks(self.fn_for_charges, featurizer.aev_length)
            )

        self_energies = self.self_energies
        shifter = EnergyAdder(
            symbols=self.symbols,
            self_energies=tuple(self_energies[k] for k in self.symbols),
        )
        kwargs: tp.Dict[str, tp.Any] = {}
        if self._pairwise_potentials:
            potentials = []
            for pot in self._pairwise_potentials:
                if pot.extra is not None:
                    pot_kwargs = pot.extra
                else:
                    pot_kwargs = {}
                if hasattr(pot.cls, "from_functional") and "functional" in pot_kwargs:
                    builder = pot.cls.from_functional
                else:
                    builder = pot.cls
                potentials.append(
                    builder(
                        symbols=self.symbols,
                        cutoff=pot.cutoff,
                        cutoff_fn=parse_cutoff_fn(
                            pot.cutoff_fn, self._global_cutoff_fn
                        ),
                        **pot_kwargs,
                    )
                )
            kwargs.update({"pairwise_potentials": potentials})

        if charge_networks is not None:
            kwargs["charge_networks"] = charge_networks
        if self._charge_normalizer is not None:
            kwargs["charge_normalizer"] = self._charge_normalizer

        return self._model_type(
            symbols=self.symbols,
            aev_computer=featurizer,
            energy_shifter=shifter,
            neural_networks=neural_networks,
            periodic_table_index=self.periodic_table_index,
            output_labels=self._output_labels,
            **kwargs,
        )


def simple_ani(
    lot: str,  # method-basis
    symbols: tp.Sequence[str],
    ensemble_size: int = 1,
    radial_start: float = 0.9,
    angular_start: float = 0.9,
    radial_cutoff: float = 5.2,
    angular_cutoff: float = 3.5,
    radial_shifts: int = 16,
    angular_shifts: int = 8,
    angle_sections: int = 4,
    radial_precision: float = 19.7,
    angular_precision: float = 12.5,
    angular_zeta: float = 14.1,
    cutoff_fn: CutoffArg = "smooth2",
    dispersion: bool = False,
    repulsion: bool = True,
    atomic_maker: AtomicMakerArg = "ani2x",
    activation: tp.Union[str, torch.nn.Module] = "gelu",
    bias: bool = False,
    use_cuda_ops: bool = False,
    output_label: str = "energies",
) -> ANI:
    r"""
    Flexible builder to create ANI-style models. Defaults are similar to ANI-2x.

    To reproduce the ANI-2x AEV exactly use the following defaults:
        - cutoff_fn='cosine'
        - radial_start=0.8
        - angular_start=0.8
        - radial_cutoff=5.1
    """
    asm = Assembler(
        ensemble_size=ensemble_size,
        periodic_table_index=True,
        output_labels=(output_label,),
    )
    asm.set_symbols(symbols)
    asm.set_global_cutoff_fn(cutoff_fn)
    asm.set_featurizer(
        AEVComputer,
        radial_terms=StandardRadial.cover_linearly(
            start=radial_start,
            cutoff=radial_cutoff,
            eta=radial_precision,
            num_shifts=radial_shifts,
        ),
        angular_terms=StandardAngular.cover_linearly(
            start=angular_start,
            eta=angular_precision,
            zeta=angular_zeta,
            num_shifts=angular_shifts,
            num_angle_sections=angle_sections,
            cutoff=angular_cutoff,
        ),
        extra={"use_cuda_extension": use_cuda_ops, "use_cuaev_interface": use_cuda_ops},
    )
    atomic_maker = functools.partial(
        atomics.parse_atomics(atomic_maker),
        activation=atomics.parse_activation(activation),
        bias=bias,
    )
    asm.set_atomic_networks(ANIModel, atomic_maker)
    asm.set_neighborlist("full_pairwise")
    asm.set_gsaes_as_self_energies(lot)
    if repulsion:
        asm.add_pairwise_potential(
            RepulsionXTB,
            cutoff=radial_cutoff,
        )
    if dispersion:
        asm.add_pairwise_potential(
            TwoBodyDispersionD3,
            cutoff=8.0,
            extra={"functional": lot.split("-")[0]},
        )
    return asm.assemble()


def simple_aniq(
    lot: str,  # method-basis
    symbols: tp.Sequence[str],
    ensemble_size: int = 1,
    radial_start: float = 0.9,
    angular_start: float = 0.9,
    radial_cutoff: float = 5.2,
    angular_cutoff: float = 3.5,
    radial_shifts: int = 16,
    angular_shifts: int = 8,
    angle_sections: int = 4,
    radial_precision: float = 19.7,
    angular_precision: float = 12.5,
    angular_zeta: float = 14.1,
    cutoff_fn: CutoffArg = "smooth2",
    dispersion: bool = False,
    repulsion: bool = True,
    atomic_maker: AtomicMakerArg = "ani2x",
    activation: tp.Union[str, torch.nn.Module] = "gelu",
    bias: bool = False,
    use_cuda_ops: bool = False,
    merge_charge_networks: bool = False,
    scale_charge_normalizer_weights: bool = True,
    dummy_energies: bool = False,
    output_label: str = "energies",
    second_output_label: str = "atomic_charges",
) -> ANI:
    r"""
    Flexible builder to create ANI-style models with separated or merged charge
    networks. Defaults are similar to ANI-2x.

    To reproduce the ANI-2x AEV exactly use the following defaults:
        - cutoff_fn='cosine'
        - radial_start=0.8
        - angular_start=0.8
        - radial_cutoff=5.1
    """
    asm = Assembler(
        ensemble_size=ensemble_size,
        periodic_table_index=True,
        model_type=ANIq,
        output_labels=(output_label, second_output_label),
    )
    asm.set_symbols(symbols)
    asm.set_global_cutoff_fn(cutoff_fn)
    asm.set_featurizer(
        AEVComputer,
        radial_terms=StandardRadial.cover_linearly(
            start=radial_start,
            cutoff=radial_cutoff,
            eta=radial_precision,
            num_shifts=radial_shifts,
        ),
        angular_terms=StandardAngular.cover_linearly(
            start=angular_start,
            eta=angular_precision,
            zeta=angular_zeta,
            num_shifts=angular_shifts,
            num_angle_sections=angle_sections,
            cutoff=angular_cutoff,
        ),
        extra={"use_cuda_extension": use_cuda_ops, "use_cuaev_interface": use_cuda_ops},
    )
    normalizer = ChargeNormalizer.from_electronegativity_and_hardness(
        asm.symbols,
        scale_weights_by_charges_squared=scale_charge_normalizer_weights,
    )
    if merge_charge_networks:
        if dummy_energies:
            raise ValueError("Can't output dummy energies with merged charge network")
        atomic_maker = functools.partial(
            atomics.parse_atomics(atomic_maker),
            out_dim=2,
            activation=atomics.parse_activation(activation),
            bias=bias,
        )
    else:
        atomic_maker = functools.partial(
            atomics.parse_atomics(atomic_maker),
            out_dim=1,
            activation=atomics.parse_activation(activation),
            bias=bias,
        )
        asm.set_charge_networks(
            ANIModel,
            atomic_maker,
            normalizer=normalizer,
        )
    asm.set_atomic_networks(
        ANIModel if not dummy_energies else DummyANIModel, atomic_maker
    )
    asm.set_neighborlist("full_pairwise")
    if not dummy_energies:
        asm.set_gsaes_as_self_energies(lot)
    else:
        asm.set_zeros_as_self_energies()
    if repulsion and not dummy_energies:
        asm.add_pairwise_potential(
            RepulsionXTB,
            cutoff=radial_cutoff,
        )
    if dispersion and not dummy_energies:
        asm.add_pairwise_potential(
            TwoBodyDispersionD3,
            cutoff=8.0,
            extra={"functional": lot.split("-")[0]},
        )
    return asm.assemble()


def fetch_state_dict(
    state_dict_file: str,
    local: bool = False,
    private: bool = False,
) -> tp.OrderedDict[str, Tensor]:
    # If pretrained=True then load state dict from a remote url or a local path
    # NOTE: torch.hub caches remote state_dicts after first download
    if local:
        dict_ = torch.load(state_dict_file, map_location=torch.device("cpu"))
        return OrderedDict(dict_)
    PUBLIC_ZOO_URL = (
        "https://github.com/roitberg-group/torchani_model_zoo/releases/download/v0.1/"
    )
    if private:
        url = "http://moria.chem.ufl.edu/animodel/private/"
    else:
        url = PUBLIC_ZOO_URL
    dict_ = torch.hub.load_state_dict_from_url(
        f"{url}/{state_dict_file}",
        model_dir=str(state_dicts_dir()),
        map_location=torch.device("cpu"),
    )
    return OrderedDict(dict_)
