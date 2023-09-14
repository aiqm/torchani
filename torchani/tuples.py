"""
Location of data classes for use in ANI built-in models.
"""

from typing import NamedTuple
from torch import Tensor


class SpeciesCoordinates(NamedTuple):
    '''
    Defines the input for built-in ANI models
    '''
    species: Tensor
    coordinates: Tensor


class SpeciesEnergies(NamedTuple):
    '''
    Tuple used in output from NNP models, used for total energy and
     atomic energies functions.
    '''
    species: Tensor
    energies: Tensor


class SpeciesEnergiesQBC(NamedTuple):
    '''
    Tuple used in output from energies_qbcs function.
    '''
    species: Tensor
    energies: Tensor
    qbcs: Tensor


class AtomicStdev(NamedTuple):
    '''
    Tuple used in output from atomic_stdev function.
    '''
    species: Tensor
    energies: Tensor
    stdev_atomic_energies: Tensor


class SpeciesForces(NamedTuple):
    '''
    Tuple used in output from members_forces function.
    '''
    species: Tensor
    energies: Tensor
    forces: Tensor


class ForceStdev(NamedTuple):
    '''
    Tuple used in output from force_qbc function.
    '''
    species: Tensor
    magnitudes: Tensor
    relative_stdev: Tensor
    relative_range: Tensor


class ForceMagnitudes(NamedTuple):
    '''
    Tuple used in output from force_magnitudes function.
    '''
    species: Tensor
    magnitudes: Tensor
