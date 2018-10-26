# -*- coding: utf-8 -*-
"""Tools for interfacing with `ASE`_.

.. _ASE:
    https://wiki.fysik.dtu.dk/ase
"""

import math
import torch
import ase.neighborlist
from . import utils


class NeighborList:
    """ASE neighborlist computer

    Arguments:
        cell: same as in :class:`ase.Atoms`
        pbc: same as in :class:`ase.Atoms`
    """

    def __init__(self, cell=None, pbc=None):
        self.pbc = pbc
        self.cell = cell

    def __call__(self, species, coordinates, cutoff):
        conformations = species.shape[0]
        max_atoms = species.shape[1]
        neighbor_species = []
        neighbor_distances = []
        neighbor_vecs = []
        for i in range(conformations):
            s = species[i].unsqueeze(0)
            c = coordinates[i].unsqueeze(0)
            s, c = utils.strip_redundant_padding(s, c)
            s = s.squeeze()
            c = c.squeeze()
            atoms = s.shape[0]
            atoms_object = ase.Atoms(
                'C'*atoms,  # chemical symbols are not important here
                positions=c.numpy(),
                pbc=self.pbc,
                cell=self.cell)
            idx1, idx2, d, D = ase.neighborlist.neighbor_list(
                'ijdD', atoms_object, cutoff)
            idx1 = torch.from_numpy(idx1).to(coordinates.device)
            idx2 = torch.from_numpy(idx2).to(coordinates.device)
            d = torch.from_numpy(d).to(coordinates.device) \
                .to(coordinates.dtype)
            D = torch.from_numpy(D).to(coordinates.device) \
                .to(coordinates.dtype)
            neighbor_species1 = []
            neighbor_distances1 = []
            neighbor_vecs1 = []
            for i in range(atoms):
                this_atom_indices = (idx1 == i).nonzero().flatten()
                neighbor_indices = idx2[this_atom_indices]
                neighbor_species1.append(s[neighbor_indices])
                neighbor_distances1.append(d[this_atom_indices])
                neighbor_vecs1.append(D.index_select(0, this_atom_indices))
            for i in range(max_atoms - atoms):
                neighbor_species1.append(torch.full((1,), -1))
                neighbor_distances1.append(torch.full((1,), math.inf))
                neighbor_vecs1.append(torch.full((1, 3), 0))
            neighbor_species1 = torch.nn.utils.rnn.pad_sequence(
                neighbor_species1, padding_value=-1)
            neighbor_distances1 = torch.nn.utils.rnn.pad_sequence(
                neighbor_distances1, padding_value=math.inf)
            neighbor_vecs1 = torch.nn.utils.rnn.pad_sequence(
                neighbor_vecs1, padding_value=0)
            neighbor_species.append(neighbor_species1)
            neighbor_distances.append(neighbor_distances1)
            neighbor_vecs.append(neighbor_vecs1)
        neighbor_species = torch.nn.utils.rnn.pad_sequence(
            neighbor_species, batch_first=True, padding_value=-1)
        neighbor_distances = torch.nn.utils.rnn.pad_sequence(
            neighbor_distances, batch_first=True, padding_value=math.inf)
        neighbor_vecs = torch.nn.utils.rnn.pad_sequence(
            neighbor_vecs, batch_first=True, padding_value=0)
        return neighbor_species.permute(0, 2, 1), \
            neighbor_distances.permute(0, 2, 1), \
            neighbor_vecs.permute(0, 2, 1, 3)
