import torch


def pad(species):
    """Put different species together into single tensor.

    If the species are from molecules of different number of total atoms, then
    ghost atoms with atom type -1 will be added to make it fit into the same
    shape.

    Arguments:
        species (:class:`collections.abc.Sequence`): sequence of species.
            Species must be of shape ``(N, A)``, where ``N`` is the number of
            3D structures, ``A`` is the number of atoms.

    Returns:
        :class:`torch.Tensor`: species batched together.
    """
    max_atoms = max([s.shape[1] for s in species])
    padded_species = []
    for s in species:
        natoms = s.shape[1]
        if natoms < max_atoms:
            padding = torch.full((s.shape[0], max_atoms - natoms), -1,
                                 dtype=torch.long, device=s.device)
            s = torch.cat([s, padding], dim=1)
        padded_species.append(s)
    return torch.cat(padded_species)


def pad_coordinates(species_coordinates):
    """Put different species and coordinates together into single tensor.

    If the species and coordinates are from molecules of different number of
    total atoms, then ghost atoms with atom type -1 and coordinate (0, 0, 0)
    will be added to make it fit into the same shape.

    Arguments:
        species_coordinates (:class:`collections.abc.Sequence`): sequence of
            pairs of species and coordinates. Species must be of shape
            ``(N, A)`` and coordinates must be of shape ``(N, A, 3)``, where
            ``N`` is the number of 3D structures, ``A`` is the number of atoms.

    Returns:
        (:class:`torch.Tensor`, :class:`torch.Tensor`): Species, and
        coordinates batched together.
    """
    max_atoms = max([c.shape[1] for _, c in species_coordinates])
    species = []
    coordinates = []
    for s, c in species_coordinates:
        natoms = c.shape[1]
        if len(s.shape) == 1:
            s = s.unsqueeze(0)
        if natoms < max_atoms:
            padding = torch.full((s.shape[0], max_atoms - natoms), -1,
                                 dtype=torch.long, device=s.device)
            s = torch.cat([s, padding], dim=1)
            padding = torch.full((c.shape[0], max_atoms - natoms, 3), 0,
                                 dtype=c.dtype, device=c.device)
            c = torch.cat([c, padding], dim=1)
        s = s.expand(c.shape[0], max_atoms)
        species.append(s)
        coordinates.append(c)
    return torch.cat(species), torch.cat(coordinates)


def present_species(species):
    """Given a vector of species of atoms, compute the unique species present.

    Arguments:
        species (:class:`torch.Tensor`): 1D vector of shape ``(atoms,)``

    Returns:
        :class:`torch.Tensor`: 1D vector storing present atom types sorted.
    """
    present_species = species.flatten().unique(sorted=True)
    if present_species[0].item() == -1:
        present_species = present_species[1:]
    return present_species


def strip_redundant_padding(species, coordinates):
    """Strip trailing padding atoms.

    Arguments:
        species (:class:`torch.Tensor`): Long tensor of shape
            ``(conformations, atoms)``.
        coordinates (:class:`torch.Tensor`): Tensor of shape
            ``(conformations, atoms, 3)``.

    Returns:
        (:class:`torch.Tensor`, :class:`torch.Tensor`): species and coordinates
        with redundant padding atoms stripped.
    """
    non_padding = (species >= 0).any(dim=0).nonzero().squeeze()
    species = species.index_select(1, non_padding)
    coordinates = coordinates.index_select(1, non_padding)
    return species, coordinates


class EnergyShifter(torch.nn.Module):
    """Helper class for adding and subtracting self atomic energies

    This is a subclass of :class:`torch.nn.Module`, so it can be used directly
    in a pipeline as ``[input->AEVComputer->ANIModel->EnergyShifter->output]``.

    Arguments:
        self_energies (:class:`collections.abc.Sequence`): Sequence of floating
            numbers for the self energy of each atom type. The numbers should
            be in order, i.e. ``self_energies[i]`` should be atom type ``i``.
    """

    def __init__(self, self_energies):
        super(EnergyShifter, self).__init__()
        self_energies = torch.tensor(self_energies, dtype=torch.double)
        self.register_buffer('self_energies', self_energies)

    def sae(self, species):
        """Compute self energies for molecules.

        Padding atoms will be automatically excluded.

        Arguments:
            species (:class:`torch.Tensor`): Long tensor in shape
                ``(conformations, atoms)``.

        Returns:
            :class:`torch.Tensor`: 1D vector in shape ``(conformations,)``
            for molecular self energies.
        """
        self_energies = self.self_energies[species]
        self_energies[species == -1] = 0
        return self_energies.sum(dim=1)

    def subtract_from_dataset(self, species, coordinates, properties):
        """Transformer for :class:`torchani.data.BatchedANIDataset` that
        subtract self energies.
        """
        energies = properties['energies']
        device = energies.device
        energies = energies.to(torch.double) - self.sae(species).to(device)
        properties['energies'] = energies
        return species, coordinates, properties

    def forward(self, species_energies):
        """(species, molecular energies)->(species, molecular energies + sae)
        """
        species, energies = species_energies
        sae = self.sae(species).to(energies.dtype).to(energies.device)
        return species, energies + sae


class ChemicalSymbolsToInts:
    """Helper that can be called to convert chemical symbol string to integers

    Arguments:
        all_species (:class:`collections.abc.Sequence` of :class:`str`):
            sequence of all supported species, in order.
    """

    def __init__(self, all_species):
        self.rev_species = {}
        for i, s in enumerate(all_species):
            self.rev_species[s] = i

    def __call__(self, species):
        """Convert species from squence of strings to 1D tensor"""
        rev = [self.rev_species[s] for s in species]
        return torch.tensor(rev, dtype=torch.long)


__all__ = ['pad', 'pad_coordinates', 'present_species',
           'strip_redundant_padding', 'ChemicalSymbolsToInts']
