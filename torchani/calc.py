import torch
from .geometry import displace_to_com_frame


def compute_dipole(species, coordinates, charges, center_of_mass=True):
    """
    Compute dipoles in eA

    Arguments:
        species (torch.Tensor): (M, N), species must be atomic numbers.
        coordinates (torch.Tensor): (M, N, 3), unit should be Angstrom.
        charges (torch.Tensor): (M, N), unit should be e.
        center_of_mass (Bool): When calculating dipole for charged molecule,
            it is necessary to displace the coordinates to the center-of-mass frame.
    Returns:
        dipoles (torch.Tensor): (M, 3)
    """
    assert species.shape == charges.shape == coordinates.shape[:-1]
    charges = charges.unsqueeze(-1)
    if center_of_mass:
        _, coordinates = displace_to_com_frame((species, coordinates))
    dipole = torch.sum(charges * coordinates, dim=1)
    return dipole
