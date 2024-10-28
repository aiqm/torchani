r"""
Construction of ANI-style models and definition of their architecture

ANI-style models are all subclasses of the ``ANI`` base class. They can be either
constructed directly, or their construction can be managed by a different class, the
``Assembler``. Think of the ``Assembler`` as your own personal employee, whose job is to
create the class from all the necessary components, in such a way that all parts
interact in the correct way and there are no compatibility issues among them.

An ANI-style model consists of:

- ``AEVComputer`` (or subclass)
- A container for atomic networks (typically ``ANINetworks`` or subclass)
- An ``AtomicNetwork`` dict ``{"H": AtomicNetwork(...), "C": AtomicNetwork(...), ...}``
- A self energies dict (in Hartree) ``{"H": -12.0, "C": -75.0, ...}``

An energy-predicting model may also have one or more ``PairPotential``
(``RepulsionXTB``, ``TwoBodyDispersion``, etc.).

Each ``PariPotential`` has their own cutoff, and the ``AEVComputer`` has two cutoffs, an
angular and a radial one (the radial cutoff must be larger than the angular cutoff, and
it is recommended that the angular cutoff is kept small, 3.5 Ang or less).

These pieces are combined by ``Assembler`` into a subclass of ``ANI``, when the
``Assembler.assemble(size=<num-networks-in-ensemble>)`` method is called.
"""

from copy import deepcopy
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
from torchani.annotations import StressKind
from torchani.neighbors import parse_neighborlist, NeighborlistArg
from torchani.cutoffs import parse_cutoff_fn, Cutoff, CutoffArg
from torchani.aev import (
    AEVComputer,
    StandardAngular,
    StandardRadial,
    RadialTermArg,
    AngularTermArg,
    parse_radial_term,
    parse_angular_term,
)
from torchani.nn import (
    SpeciesConverter,
    AtomicContainer,
    ANINetworks,
    ANIEnsemble,
    AtomicNetwork,
    AtomicMakerArg,
    AtomicMaker,
    parse_network_maker,
    parse_activation,
)
from torchani.neighbors import rescreen, NeighborData
from torchani.electro import ChargeNormalizer
from torchani.nn._internal import _ZeroANINetworks
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

    def __init__(
        self,
        symbols: tp.Sequence[str],
        aev_computer: AEVComputer,
        neural_networks: AtomicContainer,
        energy_shifter: EnergyAdder,
        pair_potentials: tp.Iterable[PairPotential] = (),
        periodic_table_index: bool = True,
    ):
        super().__init__()
        if not periodic_table_index:
            warnings.warn(
                "Setting 'periodic_table_index=False' is deprecated "
                " and will be removed in the future"
            )

        # NOTE: Keep these refs for later usage
        self.aev_computer = aev_computer
        self.neural_networks = neural_networks
        self.neighborlist = self.aev_computer.neighborlist

        device = energy_shifter.self_energies.device
        self.energy_shifter = energy_shifter
        self.species_converter = SpeciesConverter(symbols).to(device)

        self._has_pair_pots = bool(pair_potentials)
        potentials: tp.List[Potential] = list(pair_potentials)
        potentials.append(NNPotential(self.aev_computer, self.neural_networks))

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
    def set_active_members(self, idxs: tp.List[int]) -> None:
        self.neural_networks.set_active_members(idxs)
        for p in self.potentials:
            if hasattr(p, "neural_networks"):
                p.neural_networks.set_active_members(idxs)

    @torch.jit.export
    def set_strategy(self, strategy: str = "pyaev") -> None:
        self.aev_computer.set_strategy(strategy)
        for p in self.potentials:
            if hasattr(p, "aev_computer"):
                p.aev_computer.set_strategy(strategy)

    @torch.jit.export
    def sp(
        self,
        species: Tensor,
        coordinates: Tensor,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        total_charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
    ) -> tp.Dict[str, Tensor]:
        """Calculate properties for a batch of molecules

        Args:
            species: Int tensor with the atomic numbers of molecules in the batch, shape
                (molecules, atoms).
            coordinates: Float tensor with coords of molecules
                in the batch, shape (molecules, atoms, 3)
            cell: Float tensorwith the cell used for PBC computations. Set to None if
                PBC is not enabled, shape (3, 3)
            pbc: Boolean tensor that indicates enabled PBC directions. Set
                to None if PBC is not enabled. shape (3,)
            total_charge: The total charge of the molecules. Only
                the scalar 0 is currently supported.
            atomic: Perform atomic decoposition of the energies
        Returns:
            result (dict[str, Tensor]): Output of the single point calculation
        """
        _, energies = self(
            species_coordinates=(species, coordinates),
            cell=cell,
            pbc=pbc,
            total_charge=total_charge,
            atomic=atomic,
            ensemble_values=ensemble_values,
        )
        out: tp.Dict[str, Tensor] = {}
        if ensemble_values:
            if atomic:
                out["atomic_energies"] = energies.mean(dim=0)
                _values = energies.sum(dim=-1)
            else:
                _values = energies
            out["energies"] = _values.mean(dim=0)
            if _values.shape[0] == 1:
                out["ensemble_std"] = _values.new_zeros(energies.shape)
            else:
                out["ensemble_std"] = _values.std(dim=0, unbiased=True)
            out["ensemble_values"] = _values
            return out

        if atomic:
            return {"energies": energies.sum(dim=-1), "atomic_energies": energies}
        return {"energies": energies}

    def forward(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        total_charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
    ) -> SpeciesEnergies:
        species, coords = species_coordinates
        self._check_inputs(species, coords, total_charge)
        elem_idxs = self.species_converter(species, nop=not self.periodic_table_index)

        # Optimized branch that uses the cuAEV-fused strategy
        if not self._has_pair_pots and self.aev_computer._strategy == "cuaev-fused":
            aevs = self.aev_computer(elem_idxs, coords, cell=cell, pbc=pbc)
            energies = self.neural_networks(elem_idxs, aevs, atomic=atomic)
            energies += self.energy_shifter(elem_idxs, atomic=atomic)
            return SpeciesEnergies(elem_idxs, energies)

        # Less optimized branch that goes through the internal neighborlist
        max_cutoff = self.potentials[0].cutoff
        neighbors = self.neighborlist(elem_idxs, coords, max_cutoff, cell, pbc)
        energies, _ensemble_values = self._potentials_energies(
            elem_idxs,
            coords,
            max_cutoff,
            neighbors,
            atomic=atomic,
            ensemble_values=ensemble_values,
        )
        if ensemble_values:
            assert _ensemble_values is not None
            energies = _ensemble_values + energies.unsqueeze(0)
        return SpeciesEnergies(elem_idxs, energies)

    # Branch that gets its input from an external neighborlist
    @torch.jit.export
    def from_neighborlist(
        self,
        species: Tensor,
        coords: Tensor,
        neighbor_idxs: Tensor,
        shift_values: Tensor,
        total_charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
        input_needs_screening: bool = True,
    ) -> SpeciesEnergies:
        r"""This entrypoint supports input from an external neighborlist"""
        self._check_inputs(species, coords, total_charge)
        elem_idxs = self.species_converter(species, nop=not self.periodic_table_index)

        max_cutoff = self.potentials[0].cutoff
        neighbors = self.neighborlist.process_external_input(
            elem_idxs,
            coords,
            neighbor_idxs,
            shift_values,
            max_cutoff,
            input_needs_screening,
        )
        energies, _ensemble_values = self._potentials_energies(
            elem_idxs,
            coords,
            max_cutoff,
            neighbors,
            atomic=atomic,
            ensemble_values=ensemble_values,
        )
        if ensemble_values:
            assert _ensemble_values is not None
            energies = _ensemble_values + energies.unsqueeze(0)
        return SpeciesEnergies(elem_idxs, energies)

    @torch.jit.export
    def atomic_energies(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        total_charge: int = 0,
        ensemble_values: bool = False,
    ) -> SpeciesEnergies:
        r"""Calculate predicted atomic energies of all atoms in a molecule

        Arguments and return value are the same as that of forward(), but
        the returned energies have shape (molecules, atoms)
        """
        return self(
            species_coordinates,
            cell,
            pbc,
            total_charge=total_charge,
            atomic=True,
            ensemble_values=ensemble_values,
        )

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
        stress_kind: StressKind = "scaling",
        jit: bool = False,
    ):
        r"""Get an ASE Calculator using this ANI model

        Arguments:
            jit (bool): Whether to JIT-compile the model before wrapping in a
                Calculator
            overwrite (bool): After wrapping atoms into central box, whether
                to replace the original positions stored in :class:`ase.Atoms`
                object with the wrapped positions.
            stress_kind (str): Strategy to calculate stress, valid options are
                'fdotr', 'scaling', and 'numerical'. The fdotr approach does not need
                the cell's box information and can be used for multiple domians when
                running parallel on multi-GPUs. Default is 'scaling'. kwargs:
                torchani.ase.Calculator kwargs

        Returns:
            calculator (:class:`ase.Calculator`): A calculator to be used with ASE
        """
        from torchani.ase import Calculator

        model = torch.jit.script(self) if jit else self
        return Calculator(model, overwrite=overwrite, stress_kind=stress_kind)

    @property
    @torch.jit.unused
    def symbols(self) -> tp.Tuple[str, ...]:
        return tuple(PERIODIC_TABLE[z] for z in self.atomic_numbers)

    # TODO make this be useful
    @torch.jit.unused
    def disable_non_nnp(self):
        r"""
        Disable all potentials that don't have neural_networks
        """
        for p in self.potentials:
            if hasattr(p, "neural_networks"):
                p.is_enabled = False

    # TODO This is confusing, it may be a good idea to deprecate it, or at least warn
    def __len__(self):
        return self.neural_networks.get_active_members_num()

    # TODO This is confusing, it may be a good idea to deprecate it, or at least warn
    # Can be optimized but its not worth it
    def __getitem__(self, idx: int) -> tpx.Self:
        model = deepcopy(self)
        model.neural_networks = self.neural_networks.member(idx)
        for p in model.potentials:
            if isinstance(p, NNPotential):
                p.neural_networks = model.neural_networks
        return model

    # For now this assumes that there is only one potential with ensemble values
    def _potentials_energies(
        self,
        elem_idxs: Tensor,
        coords: Tensor,
        previous_cutoff: float,
        neighbors: NeighborData,
        atomic: bool,
        ensemble_values: bool,
    ) -> tp.Tuple[Tensor, tp.Optional[Tensor]]:
        # Output shape depends on the atomic flag
        if atomic:
            energies = coords.new_zeros(elem_idxs.shape)
        else:
            energies = coords.new_zeros(elem_idxs.shape[0])
        _values: tp.Optional[Tensor] = None
        for pot in self.potentials:
            cutoff = pot.cutoff
            if cutoff < previous_cutoff:
                neighbors = rescreen(cutoff, neighbors)
                previous_cutoff = cutoff
            # Separate the values of the potential that has ensemble values if requested
            if ensemble_values and hasattr(pot, "ensemble_values"):
                _values = pot.ensemble_values(
                    elem_idxs,
                    neighbors,
                    _coordinates=coords,
                    atomic=atomic,
                )
            else:
                energies += pot(
                    elem_idxs,
                    neighbors,
                    _coordinates=coords,
                    atomic=atomic,
                )
        energies += self.energy_shifter(elem_idxs, atomic=atomic)
        return energies, _values

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

    def _check_inputs(
        self, elem_idxs: Tensor, coords: Tensor, total_charge: int = 0
    ) -> None:
        # Check inputs
        assert elem_idxs.dim() == 2
        assert coords.shape == (elem_idxs.shape[0], elem_idxs.shape[1], 3)
        assert total_charge == 0, "Model only supports neutral molecules"

    @torch.jit.unused
    def members_forces(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        total_charge: int = 0,
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
        species, coordinates = species_coordinates
        coordinates.requires_grad_(True)
        elem_idxs, energies = self(
            (species, coordinates),
            cell,
            pbc,
            total_charge=total_charge,
            atomic=False,
            ensemble_values=True,
        )
        _forces = []
        for energy in energies:
            _forces.append(
                -torch.autograd.grad(energy.sum(), coordinates, retain_graph=True)[0]
            )
        forces = torch.stack(_forces, dim=0)
        return SpeciesForces(elem_idxs, energies, forces)

    @torch.jit.export
    def energies_qbcs(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        unbiased: bool = True,
        total_charge: int = 0,
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
        elem_idxs, energies = self(
            species_coordinates,
            cell,
            pbc,
            total_charge=total_charge,
            atomic=False,
            ensemble_values=True,
        )

        if energies.shape[0] == 1:
            qbc_factors = torch.zeros_like(energies).squeeze(0)
        else:
            # standard deviation is taken across ensemble members
            qbc_factors = energies.std(0, unbiased=unbiased)

        # rho's (qbc factors) are weighted by dividing by the square root of
        # the number of atoms in each molecule
        num_atoms = (elem_idxs >= 0).sum(dim=1, dtype=energies.dtype)
        qbc_factors = qbc_factors / num_atoms.sqrt()
        energies = energies.mean(dim=0)
        assert qbc_factors.shape == energies.shape
        return SpeciesEnergiesQBC(elem_idxs, energies, qbc_factors)

    @torch.jit.export
    def atomic_stdev(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        total_charge: int = 0,
        ensemble_values: bool = False,
        unbiased: bool = True,
    ) -> AtomicStdev:
        r"""Returns standard deviation of atomic energies across an ensemble

        If the model has only 1 network, a value of 0.0 is output for the stdev
        """
        elem_idxs, energies = self(
            species_coordinates,
            cell=cell,
            pbc=pbc,
            total_charge=total_charge,
            atomic=True,
            ensemble_values=True,
        )

        if energies.shape[0] == 1:
            stdev = torch.zeros_like(energies).squeeze(0)
        else:
            stdev = energies.std(0, unbiased=unbiased)

        if not ensemble_values:
            energies = energies.mean(0)

        return AtomicStdev(elem_idxs, energies, stdev)

    @torch.jit.unused
    def force_magnitudes(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        ensemble_values: bool = False,
    ) -> ForceMagnitudes:
        """
        Computes the L2 norm of predicted atomic force vectors, returning magnitudes,
        averaged by default.

        Args:
            species_coordinates: minibatch of configurations
            ensemble_values: Return force magnitudes of members of the ensemble
        """
        species, _, members_forces = self.members_forces(species_coordinates, cell, pbc)
        magnitudes = members_forces.norm(dim=-1)
        if not ensemble_values:
            magnitudes = magnitudes.mean(0)
        return ForceMagnitudes(species, magnitudes)

    @torch.jit.unused
    def force_qbc(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        ensemble_values: bool = False,
        unbiased: bool = True,
    ) -> ForceStdev:
        """
        Returns the mean force magnitudes and relative range and standard deviation
        of predicted forces across an ensemble of networks.

        Args:
            species_coordinates: minibatch of configurations
            ensemble_values: Return magnitudes of each model of the ensemble
            unbiased: whether or not to use Bessel's correction in computing
                the standard deviation, True by default
        """
        species, mags = self.force_magnitudes(
            species_coordinates, cell, pbc, ensemble_values=True
        )

        eps = 1e-8
        mean_mags = mags.mean(0)
        if mags.shape[0] == 1:
            relative_std = torch.zeros_like(mags).squeeze(0)
            relative_range = torch.ones_like(mags).squeeze(0)
        else:
            relative_std = (mags.std(0, unbiased=unbiased) + eps) / (mean_mags + eps)
            relative_range = (
                (mags.max(dim=0).values - mags.min(dim=0).values) + eps
            ) / (mean_mags + eps)

        if not ensemble_values:
            mags = mean_mags
        return ForceStdev(species, mags, relative_std, relative_range)


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
        pair_potentials: tp.Iterable[PairPotential] = (),
        periodic_table_index: bool = True,
        charge_networks: tp.Optional[AtomicContainer] = None,
        charge_normalizer: tp.Optional[ChargeNormalizer] = None,
    ):
        super().__init__(
            symbols=symbols,
            aev_computer=aev_computer,
            neural_networks=neural_networks,
            energy_shifter=energy_shifter,
            pair_potentials=pair_potentials,
            periodic_table_index=periodic_table_index,
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
        species: Tensor,
        coordinates: Tensor,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        total_charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
    ) -> tp.Dict[str, Tensor]:
        _, energies, atomic_charges = self.energies_and_atomic_charges(
            (species, coordinates),
            cell=cell,
            pbc=pbc,
            total_charge=total_charge,
            atomic=atomic,
            ensemble_values=ensemble_values,
        )
        if atomic:
            return {
                "energies": energies.sum(dim=-1),
                "atomic_energies": energies,
                "atomic_charges": atomic_charges,
            }
        return {"energies": energies, "atomic_charges": atomic_charges}

    # TODO: Remove code duplication, the next two functions should be reformulated so
    # the code is not repeated
    @torch.jit.export
    def energies_and_atomic_charges_from_neighborlist(
        self,
        species: Tensor,
        coords: Tensor,
        neighbor_idxs: Tensor,
        shift_values: Tensor,
        total_charge: int = 0,
        input_needs_screening: bool = True,
        atomic: bool = False,
        ensemble_values: bool = False,
    ) -> SpeciesEnergiesAtomicCharges:
        if ensemble_values:
            raise ValueError("ensemble_values not supported for ANIq")
        # This entrypoint supports input from an external neighborlist
        self._check_inputs(species, coords, total_charge)
        elem_idxs = self.species_converter(species, nop=not self.periodic_table_index)

        previous_cutoff = self.potentials[0].cutoff
        neighbors = self.neighborlist.process_external_input(
            elem_idxs,
            coords,
            neighbor_idxs,
            shift_values,
            previous_cutoff,
            input_needs_screening,
        )
        if atomic:
            energies = coords.new_zeros(elem_idxs.shape)
        else:
            energies = coords.new_zeros(elem_idxs.shape[0])
        atomic_charges = coords.new_zeros(elem_idxs.shape)
        for pot in self.potentials:
            cutoff = pot.cutoff
            if cutoff < previous_cutoff:
                neighbors = rescreen(cutoff, neighbors)
                previous_cutoff = cutoff
            if hasattr(pot, "energies_and_atomic_charges"):
                output = pot.energies_and_atomic_charges(
                    elem_idxs,
                    neighbors,
                    _coordinates=coords,
                    ghost_flags=None,
                    total_charge=total_charge,
                    atomic=atomic,
                )
                energies += output.energies
                atomic_charges += output.atomic_charges
            else:
                energies += pot(elem_idxs, neighbors, _coordinates=coords)
        energies += self.energy_shifter(elem_idxs, atomic=atomic)
        return SpeciesEnergiesAtomicCharges(elem_idxs, energies, atomic_charges)

    @torch.jit.export
    def energies_and_atomic_charges(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        total_charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
    ) -> SpeciesEnergiesAtomicCharges:
        if ensemble_values:
            raise ValueError("atomic E and ensemble values not supported")
        species, coords = species_coordinates
        self._check_inputs(species, coords, total_charge)
        elem_idxs = self.species_converter(species, nop=not self.periodic_table_index)

        prev_cutoff = self.potentials[0].cutoff
        neighbor_data = self.neighborlist(elem_idxs, coords, prev_cutoff, cell, pbc)
        energies = coords.new_zeros(elem_idxs.shape[0])
        atomic_charges = coords.new_zeros(elem_idxs.shape)
        if atomic:
            energies = coords.new_zeros(elem_idxs.shape)
        else:
            energies = coords.new_zeros(elem_idxs.shape[0])
        for pot in self.potentials:
            cutoff = pot.cutoff
            if cutoff < prev_cutoff:
                neighbor_data = rescreen(cutoff, neighbor_data)
                prev_cutoff = cutoff
            if hasattr(pot, "energies_and_atomic_charges"):
                output = pot.energies_and_atomic_charges(
                    elem_idxs,
                    neighbor_data,
                    _coordinates=coords,
                    ghost_flags=None,
                    total_charge=total_charge,
                    atomic=atomic,
                )
                energies += output.energies
                atomic_charges += output.atomic_charges
            else:
                energies += pot(elem_idxs, neighbor_data, _coordinates=coords)
        energies += self.energy_shifter(elem_idxs)
        return SpeciesEnergiesAtomicCharges(elem_idxs, energies, atomic_charges)


AEVComputerCls = tp.Type[AEVComputer]
PairPotentialCls = tp.Type[PairPotential]
ContainerCls = tp.Type[AtomicContainer]
ModelCls = tp.Type[ANI]


# "global" cutoff means the global cutoff_fn will be used
# Otherwise, a specific cutoff fn can be specified
class AEVComputerWrapper:
    def __init__(
        self,
        cls: AEVComputerCls,
        radial_terms: RadialTermArg,
        angular_terms: AngularTermArg,
        cutoff_fn: CutoffArg = "global",
        strategy: str = "pyaev",
    ) -> None:
        self.cls = cls
        self.cutoff_fn = cutoff_fn
        self.radial_terms = parse_radial_term(radial_terms)
        self.angular_terms = parse_angular_term(angular_terms)
        if self.angular_terms.cutoff > self.radial_terms.cutoff:
            raise ValueError("Angular cutoff must be smaller or equal to radial cutoff")
        if self.angular_terms.cutoff <= 0 or self.radial_terms.cutoff <= 0:
            raise ValueError("Cutoffs must be strictly positive")
        self.strategy = strategy


@dataclass
class PairPotentialWrapper:
    cls: PairPotentialCls
    cutoff_fn: CutoffArg = "global"
    cutoff: float = math.inf
    extra: tp.Optional[tp.Dict[str, tp.Any]] = None


class Assembler:
    def __init__(
        self,
        symbols: tp.Sequence[str] = (),
        model_cls: ModelCls = ANI,
        neighborlist: NeighborlistArg = "full_pairwise",
        periodic_table_index: bool = True,
    ) -> None:
        self._global_cutoff_fn: tp.Optional[Cutoff] = None

        self._neighborlist = parse_neighborlist(neighborlist)
        self._aevcomp: tp.Optional[AEVComputerWrapper] = None
        self._pair_potentials: tp.List[PairPotentialWrapper] = []

        # This part of the assembler organizes the self-energies, the
        # symbols and the atomic networks
        self._self_energies: tp.Dict[str, float] = {}
        self._fn_for_atomics: tp.Optional[AtomicMaker] = None
        self._fn_for_charges: tp.Optional[AtomicMaker] = None
        self._container_cls: ContainerCls = ANINetworks
        self._charge_container_cls: tp.Optional[ContainerCls] = None
        self._charge_normalizer: tp.Optional[ChargeNormalizer] = None
        self._symbols: tp.Tuple[str, ...] = tuple(symbols)

        # The general container for all the parts of the model
        self._model_cls: ModelCls = model_cls

        # This is a deprecated feature, it should probably not be used
        self.periodic_table_index = periodic_table_index

    @property
    def symbols(self) -> tp.Tuple[str, ...]:
        return self._symbols

    def set_symbols(self, symbols: tp.Sequence[str], auto_sort: bool = True) -> None:
        self._symbols = sort_by_element(symbols) if auto_sort else tuple(symbols)

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

    def set_self_energies(self, value: tp.Mapping[str, float]) -> None:
        self._check_symbols(value.keys())
        self._self_energies = {k: v for k, v in value.items()}

    def set_zeros_as_self_energies(self) -> None:
        self._check_symbols()
        self.set_self_energies({s: 0.0 for s in self.symbols})

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
        self.set_self_energies({s: gsaes[s] for s in self.symbols})

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

    def set_atomic_networks(
        self,
        fn: AtomicMaker,
        container_cls: ContainerCls = ANINetworks,
    ) -> None:
        self._container_cls = container_cls
        self._fn_for_atomics = fn

    def set_charge_networks(
        self,
        fn: AtomicMaker,
        normalizer: tp.Optional[ChargeNormalizer] = None,
        container_cls: ContainerCls = ANINetworks,
    ) -> None:
        if not issubclass(self._model_cls, ANIq):
            raise ValueError("Model must be a subclass of ANIq to use charge networks")
        self._charge_container_cls = container_cls
        self._charge_normalizer = normalizer
        self._fn_for_charges = fn

    def set_aev_computer(
        self,
        angular_terms: AngularTermArg,
        radial_terms: RadialTermArg,
        cutoff_fn: CutoffArg = "global",
        strategy: str = "pyaev",
        aev_computer_cls: AEVComputerCls = AEVComputer,
    ) -> None:
        self._aevcomp = AEVComputerWrapper(
            aev_computer_cls,
            cutoff_fn=cutoff_fn,
            angular_terms=angular_terms,
            radial_terms=radial_terms,
            strategy=strategy,
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

    def add_pair_potential(
        self,
        pair_cls: PairPotentialCls,
        cutoff: float = math.inf,
        cutoff_fn: CutoffArg = "global",
        extra: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> None:
        self._pair_potentials.append(
            PairPotentialWrapper(
                pair_cls,
                cutoff=cutoff,
                cutoff_fn=cutoff_fn,
                extra=extra,
            )
        )

    def _build_atomic_networks(
        self,
        fn_for_networks: AtomicMaker,
        in_dim: int,
    ) -> tp.OrderedDict[str, AtomicNetwork]:
        return OrderedDict([(s, fn_for_networks(s, in_dim)) for s in self.symbols])

    def assemble(self, ensemble_size: int = 1) -> ANI:
        if ensemble_size < 0:
            raise ValueError("Ensemble size must be positive")
        if not self.symbols:
            raise RuntimeError("Symbols not set. Call 'set_symbols()' before assembly")
        if self._aevcomp is None:
            raise RuntimeError(
                "AEVComputer not set. Call 'set_aev_computer' before assembly"
            )

        feat_cutoff_fn = parse_cutoff_fn(
            self._aevcomp.cutoff_fn, self._global_cutoff_fn
        )

        self._aevcomp.angular_terms.cutoff_fn = feat_cutoff_fn
        self._aevcomp.radial_terms.cutoff_fn = feat_cutoff_fn
        aevcomp = self._aevcomp.cls(
            neighborlist=self._neighborlist,
            cutoff_fn=feat_cutoff_fn,
            angular_terms=self._aevcomp.angular_terms,
            radial_terms=self._aevcomp.radial_terms,
            num_species=len(self.symbols),
            strategy=self._aevcomp.strategy,
        )
        neural_networks: AtomicContainer
        if ensemble_size > 1:
            containers = []
            for j in range(ensemble_size):
                containers.append(
                    self._container_cls(
                        self._build_atomic_networks(
                            self.fn_for_atomics, aevcomp.aev_length
                        )
                    )
                )
            neural_networks = ANIEnsemble(containers)
        else:
            neural_networks = self._container_cls(
                self._build_atomic_networks(self.fn_for_atomics, aevcomp.aev_length)
            )

        charge_networks: tp.Optional[AtomicContainer] = None
        if self._charge_container_cls is not None:
            charge_networks = self._charge_container_cls(
                self._build_atomic_networks(self.fn_for_charges, aevcomp.aev_length)
            )

        self_energies = self.self_energies
        shifter = EnergyAdder(
            symbols=self.symbols,
            self_energies=tuple(self_energies[k] for k in self.symbols),
        )
        kwargs: tp.Dict[str, tp.Any] = {}
        if self._pair_potentials:
            potentials = []
            for pot in self._pair_potentials:
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
            kwargs.update({"pair_potentials": potentials})

        if charge_networks is not None:
            kwargs["charge_networks"] = charge_networks
        if self._charge_normalizer is not None:
            kwargs["charge_normalizer"] = self._charge_normalizer

        return self._model_cls(
            symbols=self.symbols,
            aev_computer=aevcomp,
            energy_shifter=shifter,
            neural_networks=neural_networks,
            periodic_table_index=self.periodic_table_index,
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
    network_factory: AtomicMakerArg = "ani2x",
    activation: tp.Union[str, torch.nn.Module] = "gelu",
    bias: bool = False,
    strategy: str = "pyaev",
    use_cuda_ops: bool = False,
) -> ANI:
    r"""
    Flexible builder to create ANI-style models. Defaults are similar to ANI-2x.

    To reproduce the ANI-2x AEV exactly use the following defaults:
        - cutoff_fn='cosine'
        - radial_start=0.8
        - angular_start=0.8
        - radial_cutoff=5.1
    """
    if strategy not in ["pyaev", "cuaev"]:
        raise ValueError(f"Unavailable strategy: {strategy}")
    if use_cuda_ops:
        warnings.warn("use_cuda_ops is deprecated, please use strategy = 'cuaev'")
        strategy = "cuaev"
    asm = Assembler()
    asm.set_symbols(symbols)
    asm.set_global_cutoff_fn(cutoff_fn)
    asm.set_aev_computer(
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
        strategy=strategy,
    )
    network_factory = functools.partial(
        parse_network_maker(network_factory),
        activation=parse_activation(activation),
        bias=bias,
    )
    asm.set_atomic_networks(network_factory)
    asm.set_neighborlist("full_pairwise")
    asm.set_gsaes_as_self_energies(lot)
    if repulsion:
        asm.add_pair_potential(
            RepulsionXTB,
            cutoff=radial_cutoff,
        )
    if dispersion:
        asm.add_pair_potential(
            TwoBodyDispersionD3,
            cutoff=8.0,
            extra={"functional": lot.split("-")[0]},
        )
    return asm.assemble(ensemble_size)


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
    network_factory: AtomicMakerArg = "ani2x",
    activation: tp.Union[str, torch.nn.Module] = "gelu",
    bias: bool = False,
    strategy: str = "pyaev",
    merge_charge_networks: bool = False,
    scale_charge_normalizer_weights: bool = True,
    dummy_energies: bool = False,
    use_cuda_ops: bool = False,
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
    if strategy not in ["pyaev", "cuaev"]:
        raise ValueError(f"Unavailable strategy: {strategy}")
    if use_cuda_ops:
        warnings.warn("use_cuda_ops is deprecated, please use strategy = 'cuaev'")
        strategy = "cuaev"
    asm = Assembler(model_cls=ANIq)
    asm.set_symbols(symbols)
    asm.set_global_cutoff_fn(cutoff_fn)
    asm.set_aev_computer(
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
        strategy=strategy,
    )
    normalizer = ChargeNormalizer.from_electronegativity_and_hardness(
        asm.symbols,
        scale_weights_by_charges_squared=scale_charge_normalizer_weights,
    )
    if merge_charge_networks:
        if dummy_energies:
            raise ValueError("Can't output dummy energies with merged charge network")
        network_factory = functools.partial(
            parse_network_maker(network_factory),
            out_dim=2,
            activation=parse_activation(activation),
            bias=bias,
        )
    else:
        network_factory = functools.partial(
            parse_network_maker(network_factory),
            out_dim=1,
            activation=parse_activation(activation),
            bias=bias,
        )
        asm.set_charge_networks(
            network_factory,
            normalizer=normalizer,
        )

    asm.set_atomic_networks(
        network_factory,
        container_cls=_ZeroANINetworks if dummy_energies else ANINetworks,
    )
    asm.set_neighborlist("full_pairwise")
    if not dummy_energies:
        asm.set_gsaes_as_self_energies(lot)
    else:
        asm.set_zeros_as_self_energies()
    if repulsion and not dummy_energies:
        asm.add_pair_potential(
            RepulsionXTB,
            cutoff=radial_cutoff,
        )
    if dispersion and not dummy_energies:
        asm.add_pair_potential(
            TwoBodyDispersionD3,
            cutoff=8.0,
            extra={"functional": lot.split("-")[0]},
        )
    return asm.assemble(ensemble_size)


def fetch_state_dict(
    state_dict_file: str,
    local: bool = False,
    private: bool = False,
) -> tp.OrderedDict[str, Tensor]:
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
