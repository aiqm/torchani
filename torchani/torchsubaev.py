import torch
import numpy
import itertools
from .aev_base import AEVComputer
from . import buildin_const_file
from .torchaev import AEV


class SubAEV(AEVComputer):
    """The subAEV computer fully implemented using pytorch

    Note that this AEV computer only computes the subAEV of a given species
    of a single atom.  The `__call__` method is intentionally left unimplemented.
    """

    def __init__(self, dtype=torch.cuda.float32, const_file=buildin_const_file):
        super(SubAEV, self).__init__(dtype, const_file)

    def radial_subaev(self, center, neighbors):
        """Compute the radial subAEV of the center atom given neighbors

        The radial AEV is define in https://arxiv.org/pdf/1610.08935.pdf equation 3.
        The sum computed by this method is over all given neighbors, so the caller
        of this method need to select neighbors if the caller want a per species subAEV.

        Parameters
        ----------
        center : pytorch tensor of `dtype`
            A tensor of shape (conformations, 3) that stores the xyz coordinate of the
            center atoms.
        neighbors : pytorch tensor of `dtype`
            A tensor of shape (conformations, N, 3) where N is the number of neighbors.
            The tensor stores the xyz coordinate of the neighbor atoms. Note that different
            conformations might have different neighbor atoms within the cutoff radius, if
            this is the case, the union of neighbors of all conformations should be given for
            this parameter.

        Returns
        -------
        pytorch tensor of `dtype`
            A tensor of shape (conformations, `per_species_radial_length()`) storing the subAEVs.
        """
        atoms = neighbors.shape[1]
        Rij_vec = neighbors - center.view(-1, 1, 3)
        """pytorch tensor of shape (conformations, N, 3) storing the Rij vectors where i is the
        center atom, and j is a neighbor. The Rij of conformation n is stored as (n,j,:)"""
        distances = torch.sqrt(torch.sum(Rij_vec ** 2, dim=-1))
        """pytorch tensor of shape (conformations, N) storing the |Rij| length where i is the
        center atom, and j is a neighbor. The |Rij| of conformation n is stored as (n,j)"""

        # use broadcasting semantics to do Cartesian product on constants
        # shape convension (conformations, atoms, EtaR, ShfR)
        distances = distances.view(-1, atoms, 1, 1)
        fc = AEV._cutoff_cosine(distances, self.constants['Rcr'])
        eta = torch.Tensor(self.constants['EtaR']).type(
            self.dtype).view(1, 1, -1, 1)
        radius_shift = torch.Tensor(self.constants['ShfR']).type(
            self.dtype).view(1, 1, 1, -1)
        # Note that in the equation in the paper there is no 0.25 coefficient, but in NeuroChem there is such a coefficient. We choose to be consistent with NeuroChem instead of the paper here.
        ret = 0.25 * torch.exp(-eta * (distances - radius_shift)**2) * fc
        # end of shape convension
        ret = torch.sum(ret, dim=1)
        # flat the last two dimensions to view the subAEV as one dimensional vector
        return ret.view(-1, self.per_species_radial_length())

    def angular_subaev(self, center, neighbors):
        """Compute the angular subAEV of the center atom given neighbor pairs.

        The angular AEV is define in https://arxiv.org/pdf/1610.08935.pdf equation 4.
        The sum computed by this method is over all given neighbor pairs, so the caller
        of this method need to select neighbors if the caller want a per species subAEV.

        Parameters
        ----------
        center : pytorch tensor of `dtype`
            A tensor of shape (conformations, 3) that stores the xyz coordinate of the
            center atoms.
        neighbors : pytorch tensor of `dtype`
            A tensor of shape (conformations, N, 2, 3) where N is the number of neighbor pairs.
            The tensor stores the xyz coordinate of the 2 atoms in neighbor pairs. Note that
            different conformations might have different neighbor pairs within the cutoff radius,
            if this is the case, the union of neighbors of all conformations should be given for
            this parameter.

        Returns
        -------
        pytorch tensor of `dtype`
            A tensor of shape (conformations, `per_species_angular_length()`) storing the subAEVs.
        """
        pairs = neighbors.shape[1]
        Rij_vec = neighbors - center.view(-1, 1, 1, 3)
        """pytorch tensor of shape (conformations, N, 2, 3) storing the Rij vectors where i is the
        center atom, and j is a neighbor. The vector (n,k,l,:) is the Rij where j refer to the l-th
        atom of the k-th pair."""
        R_distances = torch.sqrt(torch.sum(Rij_vec ** 2, dim=-1))
        """pytorch tensor of shape (conformations, N, 2) storing the |Rij| length where i is the
        center atom, and j is a neighbor. The value at (n,k,l) is the |Rij| where j refer to the
        l-th atom of the k-th pair."""

        # Compute the product of two distances |Rij| * |Rik| where j and k are the two atoms in
        # a pair. The result tensor would have shape (conformations, pairs)
        Rijk_distance_prods = R_distances[:, :, 0] * R_distances[:, :, 1]

        # Compute the inner product Rij (dot) Rik where j and k are the two atoms in a pair.
        # The result tensor would have shape (conformations, pairs)
        Rijk_inner_prods = torch.sum(
            Rij_vec[:, :, 0, :] * Rij_vec[:, :, 1, :], dim=-1)

        # Compute the angles jik with i in the center and j and k are the two atoms in a pair.
        # The result tensor would have shape (conformations, pairs)
        # 0.95 is multiplied to the cos values to prevent acos from returning NaN.
        cos_angles = 0.95 * Rijk_inner_prods / Rijk_distance_prods
        angles = torch.acos(cos_angles)

        # use broadcasting semantics to combine constants
        # shape convension (conformations, pairs, EtaA, Zeta, ShfA, ShfZ)
        angles = angles.view(-1, pairs, 1, 1, 1, 1)
        Rij = R_distances.view(-1, pairs, 2, 1, 1, 1, 1)
        fcj = AEV._cutoff_cosine(Rij, self.constants['Rca'])
        eta = torch.Tensor(self.constants['EtaA']).type(
            self.dtype).view(1, 1, -1, 1, 1, 1)
        zeta = torch.Tensor(self.constants['Zeta']).type(
            self.dtype).view(1, 1, 1, -1, 1, 1)
        radius_shifts = torch.Tensor(self.constants['ShfA']).type(
            self.dtype).view(1, 1, 1, 1, -1, 1)
        angle_shifts = torch.Tensor(self.constants['ShfZ']).type(
            self.dtype).view(1, 1, 1, 1, 1, -1)
        ret = 2 * ((1 + torch.cos(angles - angle_shifts)) / 2) ** zeta * \
            torch.exp(-eta * ((Rij[:, :, 0, :, :, :, :] + Rij[:, :, 1, :, :, :, :]) / 2 - radius_shifts)
                      ** 2) * fcj[:, :, 0, :, :, :, :] * fcj[:, :, 1, :, :, :, :]
        # end of shape convension
        ret = torch.sum(ret, dim=1)
        # flat the last 4 dimensions to view the subAEV as one dimension vector
        return ret.view(-1, self.per_species_angular_length())


class AEVFromSubAEV(AEVComputer):
    pass
