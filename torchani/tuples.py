r"""
:class:`typing.NamedTuple` used as outputs of some TorchANI functions.
"""

import typing as tp

from torch import Tensor


class EnergiesForces(tp.NamedTuple):
    energies: Tensor
    forces: Tensor


class EnergiesForcesHessians(tp.NamedTuple):
    energies: Tensor
    forces: Tensor
    hessians: Tensor


class ForcesHessians(tp.NamedTuple):
    forces: Tensor
    hessians: Tensor


class SpeciesAEV(tp.NamedTuple):
    r"""
    Chemical elements and AEV feature tensor
    """

    species: Tensor
    aevs: Tensor


class VibAnalysis(tp.NamedTuple):
    r"""
    Frequencies, modes, force constants and reduced masses in vibrational analysis
    """

    freqs: Tensor
    modes: Tensor
    fconstants: Tensor
    rmasses: Tensor


class NeighborData(tp.NamedTuple):
    r"""
    Output data of the neighborlist module
    """

    indices: Tensor
    distances: Tensor
    diff_vectors: Tensor
    shift_values: tp.Optional[Tensor] = None


class SpeciesCoordinates(tp.NamedTuple):
    r"""
    Defines the input for built-in ANI models
    """

    species: Tensor
    coordinates: Tensor


class SpeciesEnergies(tp.NamedTuple):
    r"""
    Tuple used in output from NNP models, used for total energy and
     atomic energies functions.
    """

    species: Tensor
    energies: Tensor


class SpeciesEnergiesAtomicCharges(tp.NamedTuple):
    species: Tensor
    energies: Tensor
    atomic_charges: Tensor


class EnergiesAtomicCharges(tp.NamedTuple):
    energies: Tensor
    atomic_charges: Tensor


class SpeciesAtomicCharges(tp.NamedTuple):
    energies: Tensor
    atomic_charges: Tensor


class SpeciesEnergiesQBC(tp.NamedTuple):
    """
    Tuple used in output from energies_qbcs function.
    """

    species: Tensor
    energies: Tensor
    qbcs: Tensor


class AtomicStdev(tp.NamedTuple):
    """
    Tuple used in output from atomic_stdev function.
    """

    species: Tensor
    energies: Tensor
    stdev_atomic_energies: Tensor


class SpeciesForces(tp.NamedTuple):
    """
    Tuple used in output from members_forces function.
    """

    species: Tensor
    energies: Tensor
    forces: Tensor


class ForceStdev(tp.NamedTuple):
    """
    Tuple used in output from force_qbc function.
    """

    species: Tensor
    magnitudes: Tensor
    relative_stdev: Tensor
    relative_range: Tensor


class ForceMagnitudes(tp.NamedTuple):
    """
    Tuple used in output from force_magnitudes function.
    """

    species: Tensor
    magnitudes: Tensor
