# -*- coding: utf-8 -*-
"""Tools for interfacing with `ASE`_.

.. _ASE:
    https://wiki.fysik.dtu.dk/ase
"""

import ase
from . import utils


class NeighborList:
    """ASE neighborlist computer"""

    def __init__(self, cell=None, pbc=None):
        self.pbc = pbc
        self.cell = cell

    def __call__(self, species, coordinates, cutoff):
        conformations = species.shape[0]
        for i in range(conformations):
            s = species[i]
            c = coordinates[i]
            s, c = utils.strip_redundant_padding(s, c)
            idx1, idx2, d, D = ase.neighborlist.primitive_neighbor_list(
                'ijdD', self.pbc, self.cell, c.squeeze(), cutoff)
            