import torch
import itertools
import math
from .env import buildin_const_file
from .benchmarked import BenchmarkedModule


class AEVComputer(BenchmarkedModule):
    __constants__ = ['Rcr', 'Rca', 'radial_sublength', 'radial_length',
                     'angular_sublength', 'angular_length', 'aev_length']

    """Base class of various implementations of AEV computer

    Attributes
    ----------
    benchmark : boolean
        Whether to enable benchmark
    const_file : str
        The name of the original file that stores constant.
    Rcr, Rca : float
        Cutoff radius
    EtaR, ShfR, Zeta, ShfZ, EtaA, ShfA : torch.Tensor
        Tensor storing constants.
    radial_sublength : int
        The length of radial subaev of a single species
    radial_length : int
        The length of full radial aev
    angular_sublength : int
        The length of angular subaev of a single species
    angular_length : int
        The length of full angular aev
    aev_length : int
        The length of full aev
    """

    def __init__(self, benchmark=False, const_file=buildin_const_file):
        super(AEVComputer, self).__init__(benchmark)
        self.const_file = const_file

        # load constants from const file
        const = {}
        with open(const_file) as f:
            for i in f:
                try:
                    line = [x.strip() for x in i.split('=')]
                    name = line[0]
                    value = line[1]
                    if name == 'Rcr' or name == 'Rca':
                        setattr(self, name, float(value))
                    elif name in ['EtaR', 'ShfR', 'Zeta',
                                  'ShfZ', 'EtaA', 'ShfA']:
                        value = [float(x.strip()) for x in value.replace(
                            '[', '').replace(']', '').split(',')]
                        value = torch.tensor(value)
                        const[name] = value
                    elif name == 'Atyp':
                        value = [x.strip() for x in value.replace(
                            '[', '').replace(']', '').split(',')]
                        self.species = value
                except Exception:
                    raise ValueError('unable to parse const file')

        # Compute lengths
        self.radial_sublength = const['EtaR'].shape[0] * const['ShfR'].shape[0]
        self.radial_length = len(self.species) * self.radial_sublength
        self.angular_sublength = const['EtaA'].shape[0] * \
            const['Zeta'].shape[0] * const['ShfA'].shape[0] * \
            const['ShfZ'].shape[0]
        species = len(self.species)
        self.angular_length = int(
            (species * (species + 1)) / 2) * self.angular_sublength
        self.aev_length = self.radial_length + self.angular_length

        # convert constant tensors to a ready-to-broadcast shape
        # shape convension (..., EtaR, ShfR)
        const['EtaR'] = const['EtaR'].view(-1, 1)
        const['ShfR'] = const['ShfR'].view(1, -1)
        # shape convension (..., EtaA, Zeta, ShfA, ShfZ)
        const['EtaA'] = const['EtaA'].view(-1, 1, 1, 1)
        const['Zeta'] = const['Zeta'].view(1, -1, 1, 1)
        const['ShfA'] = const['ShfA'].view(1, 1, -1, 1)
        const['ShfZ'] = const['ShfZ'].view(1, 1, 1, -1)

        # register buffers
        for i in const:
            self.register_buffer(i, const[i])

    def forward(self, coordinates_species):
        """Compute AEV from coordinates and species

        Parameters
        ----------
        (species, coordinates)
        species : torch.LongTensor
            Long tensor for the species, where a value k means the species is
            the same as self.species[k]
        coordinates : torch.Tensor
            The tensor that specifies the xyz coordinates of atoms in the
            molecule. The tensor must have shape (conformations, atoms, 3)

        Returns
        -------
        (torch.Tensor, torch.LongTensor)
            Returns full AEV and species
        """
        raise NotImplementedError('subclass must override this method')


class PrepareInput(torch.nn.Module):

    def __init__(self, species):
        super(PrepareInput, self).__init__()
        self.species = species

    def species_to_tensor(self, species, device):
        """Convert species list into a long tensor.

        Parameters
        ----------
        species : list
            List of string for the species of each atoms.
        device : torch.device
            The device to store tensor

        Returns
        -------
        torch.Tensor
            Long tensor for the species, where a value k means the species is
            the same as self.species[k].
        """
        indices = {self.species[i]: i for i in range(len(self.species))}
        values = [indices[i] for i in species]
        return torch.tensor(values, dtype=torch.long, device=device)

    def sort_by_species(self, species, *tensors):
        """Sort the data by its species according to the order in `self.species`

        Parameters
        ----------
        species : torch.Tensor
            Tensor storing species of each atom.
        *tensors : tuple
            Tensors of shape (conformations, atoms, ...) for data.

        Returns
        -------
        (species, ...)
            Tensors sorted by species.
        """
        species, reverse = torch.sort(species)
        new_tensors = []
        for t in tensors:
            new_tensors.append(t.index_select(1, reverse))
        return (species, *new_tensors)

    def forward(self, species_coordinates):
        species, coordinates = species_coordinates
        species = self.species_to_tensor(species, coordinates.device)
        return self.sort_by_species(species, coordinates)


def _cutoff_cosine(distances, cutoff):
    """Compute the elementwise cutoff cosine function

    The cutoff cosine function is define in
    https://arxiv.org/pdf/1610.08935.pdf equation 2

    Parameters
    ----------
    distances : torch.Tensor
        The pytorch tensor that stores Rij values. This tensor can
        have any shape since the cutoff cosine function is computed
        elementwise.
    cutoff : float
        The cutoff radius, i.e. the Rc in the equation. For any Rij > Rc,
        the function value is defined to be zero.

    Returns
    -------
    torch.Tensor
        The tensor of the same shape as `distances` that stores the
        computed function values.
    """
    return torch.where(
        distances <= cutoff,
        0.5 * torch.cos(math.pi * distances / cutoff) + 0.5,
        torch.zeros_like(distances)
    )


class SortedAEV(AEVComputer):
    """The AEV computer assuming input coordinates sorted by species

    Attributes
    ----------
    timers : dict
        Dictionary storing the the benchmark result. It has the following keys:
            radial_subaev : time spent on computing radial subaev
            angular_subaev : time spent on computing angular subaev
            total : total time for computing everything.
    """

    def __init__(self, benchmark=False, const_file=buildin_const_file):
        super(SortedAEV, self).__init__(benchmark, const_file)
        if benchmark:
            self.radial_subaev_terms = self._enable_benchmark(
                self.radial_subaev_terms, 'radial terms')
            self.angular_subaev_terms = self._enable_benchmark(
                self.angular_subaev_terms, 'angular terms')
            self.terms_and_indices = self._enable_benchmark(
                self.terms_and_indices, 'terms and indices')
            self.combinations = self._enable_benchmark(
                self.combinations, 'combinations')
            self.compute_mask_r = self._enable_benchmark(
                self.compute_mask_r, 'mask_r')
            self.compute_mask_a = self._enable_benchmark(
                self.compute_mask_a, 'mask_a')
            self.assemble = self._enable_benchmark(self.assemble, 'assemble')
            self.forward = self._enable_benchmark(self.forward, 'total')

    def radial_subaev_terms(self, distances):
        """Compute the radial subAEV terms of the center atom given neighbors

        The radial AEV is define in
        https://arxiv.org/pdf/1610.08935.pdf equation 3.
        The sum computed by this method is over all given neighbors,
        so the caller of this method need to select neighbors if the
        caller want a per species subAEV.

        Parameters
        ----------
        distances : torch.Tensor
            Pytorch tensor of shape (..., neighbors) storing the |Rij|
            length where i are the center atoms, and j are their neighbors.

        Returns
        -------
        torch.Tensor
            A tensor of shape (..., neighbors, `radial_sublength`) storing
            the subAEVs.
        """
        distances = distances.unsqueeze(-1).unsqueeze(-1)
        fc = _cutoff_cosine(distances, self.Rcr)
        # Note that in the equation in the paper there is no 0.25
        # coefficient, but in NeuroChem there is such a coefficient.
        # We choose to be consistent with NeuroChem instead of the paper here.
        ret = 0.25 * torch.exp(-self.EtaR * (distances - self.ShfR)**2) * fc
        return ret.flatten(start_dim=-2)

    def angular_subaev_terms(self, vectors1, vectors2):
        """Compute the angular subAEV terms of the center atom given neighbor pairs.

        The angular AEV is define in
        https://arxiv.org/pdf/1610.08935.pdf equation 4.
        The sum computed by this method is over all given neighbor pairs,
        so the caller of this method need to select neighbors if the caller
        want a per species subAEV.

        Parameters
        ----------
        vectors1, vectors2: torch.Tensor
            Tensor of shape (..., pairs, 3) storing the Rij vectors of pairs
            of neighbors. The vectors1(..., j, :) and vectors2(..., j, :) are
            the Rij vectors of the two atoms of pair j.

        Returns
        -------
        torch.Tensor
            Tensor of shape (..., pairs, `angular_sublength`) storing the
            subAEVs.
        """
        vectors1 = vectors1.unsqueeze(
            -1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        vectors2 = vectors2.unsqueeze(
            -1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        distances1 = vectors1.norm(2, dim=-5)
        distances2 = vectors2.norm(2, dim=-5)

        # 0.95 is multiplied to the cos values to prevent acos from
        # returning NaN.
        cos_angles = 0.95 * \
            torch.nn.functional.cosine_similarity(
                vectors1, vectors2, dim=-5)
        angles = torch.acos(cos_angles)

        fcj1 = _cutoff_cosine(distances1, self.Rca)
        fcj2 = _cutoff_cosine(distances2, self.Rca)
        factor1 = ((1 + torch.cos(angles - self.ShfZ)) / 2) ** self.Zeta
        factor2 = torch.exp(-self.EtaA *
                            ((distances1 + distances2) / 2 - self.ShfA) ** 2)
        ret = 2 * factor1 * factor2 * fcj1 * fcj2
        # ret now have shape (..., pairs, ?, ?, ?, ?) where ? depend on
        # constants

        # flat the last 4 dimensions to view the subAEV as one dimension vector
        return ret.flatten(start_dim=-4)

    def terms_and_indices(self, coordinates):
        """Compute radial and angular subAEV terms, and original indices.

        Terms will be sorted according to their distances to central atoms,
        and only these within cutoff radius are valid. The returned indices
        contains what would their original indices be if they were unsorted.

        Parameters
        ----------
        coordinates : torch.Tensor
            The tensor that specifies the xyz coordinates of atoms in the
            molecule. The tensor must have shape (conformations, atoms, 3)

        Returns
        -------
        (radial_terms, angular_terms, indices_r, indices_a)
        radial_terms : torch.Tensor
            Tensor shaped (conformations, atoms, neighbors, `radial_sublength`)
            for the (unsummed) radial subAEV terms.
        angular_terms : torch.Tensor
            Tensor of shape (conformations, atoms, pairs, `angular_sublength`)
            for the (unsummed) angular subAEV terms.
        indices_r : torch.Tensor
            Tensor of shape (conformations, atoms, neighbors).
            Let l = indices_r(i,j,k), then this means that
            radial_terms(i,j,k,:) is in the subAEV term of conformation i
            between atom j and atom l.
        indices_a : torch.Tensor
            Same as indices_r, except that the cutoff radius is Rca instead of
            Rcr.
        """

        vec = coordinates.unsqueeze(2) - coordinates.unsqueeze(1)
        """Shape (conformations, atoms, atoms, 3) storing Rij vectors"""

        distances = vec.norm(2, -1)
        """Shape (conformations, atoms, atoms) storing Rij distances"""

        distances, indices = distances.sort(-1)

        min_distances, _ = distances.flatten(end_dim=1).min(0)
        inRcr = (min_distances <= self.Rcr).nonzero().flatten()[
            1:]  # TODO: can we use something like find_first?
        inRca = (min_distances <= self.Rca).nonzero().flatten()[1:]

        distances = distances.index_select(-1, inRcr)
        indices_r = indices.index_select(-1, inRcr)
        radial_terms = self.radial_subaev_terms(distances)

        indices_a = indices.index_select(-1, inRca)
        new_shape = list(indices_a.shape) + [3]
        # TODO: can we add something like expand_dim(dim=0, repeat=3)
        _indices_a = indices_a.unsqueeze(-1).expand(*new_shape)
        # TODO: can we make gather broadcast??
        vec = vec.gather(-2, _indices_a)
        # TODO: can we move combinations to ATen?
        vec = self.combinations(vec, -2)
        angular_terms = self.angular_subaev_terms(*vec)

        return radial_terms, angular_terms, indices_r, indices_a

    def combinations(self, tensor, dim=0):
        n = tensor.shape[dim]
        r = torch.arange(n).type(torch.long).to(tensor.device)
        grid_x, grid_y = torch.meshgrid([r, r])
        index1 = grid_y.masked_select(
            torch.triu(torch.ones(n, n, device=self.EtaR.device),
                       diagonal=1) == 1)
        index2 = grid_x.masked_select(
            torch.triu(torch.ones(n, n, device=self.EtaR.device),
                       diagonal=1) == 1)
        return tensor.index_select(dim, index1), \
            tensor.index_select(dim, index2)

    def compute_mask_r(self, species_r):
        """Partition indices according to their species, radial part

        Parameters
        ----------
        species_r : torch.Tensor
            Tensor of shape (conformations, atoms, neighbors) storing
            species of neighbors.

        Returns
        -------
        torch.Tensor
            Tensor of shape (conformations, atoms, neighbors, all species)
            storing the mask for each species.
        """
        mask_r = (species_r.unsqueeze(-1) ==
                  torch.arange(len(self.species), device=self.EtaR.device))
        return mask_r

    def compute_mask_a(self, species_a, present_species):
        """Partition indices according to their species, angular part

        Parameters
        ----------
        species_a : torch.Tensor
            Tensor of shape (conformations, atoms, neighbors) storing the
            species of neighbors.
        present_species : torch.Tensor
            Long tensor for the species, already uniqued.

        Returns
        -------
        torch.Tensor
            Tensor of shape (conformations, atoms, pairs, present species,
            present species) storing the mask for each pair.
        """
        species_a1, species_a2 = self.combinations(species_a, -1)
        mask_a1 = (species_a1.unsqueeze(-1) == present_species).unsqueeze(-1)
        mask_a2 = (species_a2.unsqueeze(-1).unsqueeze(-1) == present_species)
        mask = mask_a1 * mask_a2
        mask_rev = mask.permute(0, 1, 2, 4, 3)
        mask_a = (mask + mask_rev) > 0
        return mask_a

    def assemble(self, radial_terms, angular_terms, present_species,
                 mask_r, mask_a):
        """Assemble radial and angular AEV from computed terms according
        to the given partition information.

        Parameters
        ----------
        radial_terms : torch.Tensor
            Tensor shaped (conformations, atoms, neighbors, `radial_sublength`)
            for the (unsummed) radial subAEV terms.
        angular_terms : torch.Tensor
            Tensor of shape (conformations, atoms, pairs, `angular_sublength`)
            for the (unsummed) angular subAEV terms.
        present_species : torch.Tensor
            Long tensor for species of atoms present in the molecules.
        mask_r : torch.Tensor
            Tensor of shape (conformations, atoms, neighbors, present species)
            storing the mask for each species.
        mask_a : torch.Tensor
            Tensor of shape (conformations, atoms, pairs, present species,
            present species) storing the mask for each pair.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Returns (radial AEV, angular AEV), both are pytorch tensor of
            `dtype`. The radial AEV must be of shape (conformations, atoms,
            radial_length) The angular AEV must be of shape (conformations,
            atoms, angular_length)
        """
        conformations = radial_terms.shape[0]
        atoms = radial_terms.shape[1]

        # assemble radial subaev
        present_radial_aevs = (
            radial_terms.unsqueeze(-2) *
            mask_r.unsqueeze(-1).type(radial_terms.dtype)
        ).sum(-3)
        """shape (conformations, atoms, present species, radial_length)"""
        radial_aevs = present_radial_aevs.flatten(start_dim=2)

        # assemble angular subaev
        # TODO: can we use find_first?
        rev_indices = {present_species[i].item(): i
                       for i in range(len(present_species))}
        """shape (conformations, atoms, present species,
                  present species, angular_length)"""
        angular_aevs = []
        zero_angular_subaev = torch.zeros(
            # TODO: can we make stack and cat broadcast?
            conformations, atoms, self.angular_sublength,
            dtype=self.EtaR.dtype, device=self.EtaR.device)
        for s1, s2 in itertools.combinations_with_replacement(
                                        range(len(self.species)), 2):
            if s1 in rev_indices and s2 in rev_indices:
                i1 = rev_indices[s1]
                i2 = rev_indices[s2]
                mask = mask_a[..., i1, i2].unsqueeze(-1).type(self.EtaR.dtype)
                subaev = (angular_terms * mask).sum(-2)
            else:
                subaev = zero_angular_subaev
            angular_aevs.append(subaev)

        return radial_aevs, torch.cat(angular_aevs, dim=2)

    def forward(self, species_coordinates):
        species, coordinates = species_coordinates
        present_species = species.unique(sorted=True)

        radial_terms, angular_terms, indices_r, indices_a = \
            self.terms_and_indices(coordinates)

        species_r = species.take(indices_r)
        mask_r = self.compute_mask_r(species_r)
        species_a = species.take(indices_a)
        mask_a = self.compute_mask_a(species_a, present_species)

        radial, angular = self.assemble(radial_terms, angular_terms,
                                        present_species, mask_r, mask_a)
        fullaev = torch.cat([radial, angular], dim=2)
        return species, fullaev
