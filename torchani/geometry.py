""" Utilities to generate some specific geometries"""
import torch
from torch import Tensor
from typing import Tuple
from .utils import get_atomic_masses


def displace_to_com_frame(species_coordinates: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
    r"""Displace coordinates to the center-of-mass frame, input species must be
    atomic numbers, padding atoms can be included with -1 as padding,
    returns the displaced coordinates and the center-of-mass coordinates"""
    species, coordinates = species_coordinates
    mask = (species == -1)
    masses = get_atomic_masses(species, dtype=coordinates.dtype)
    masses.masked_fill_(mask, 0.0)
    mass_sum = masses.unsqueeze(-1).sum(dim=1, keepdim=True)
    com_coordinates = coordinates * masses.unsqueeze(-1) / mass_sum
    com_coordinates = com_coordinates.sum(dim=1, keepdim=True)
    centered_coordinates = coordinates - com_coordinates
    centered_coordinates[mask, :] = 0.0
    return species, centered_coordinates


def tile_into_tight_cell(species_coordinates, repeats=(3, 3, 3), noise=None, delta=1.0,
                         density=None, fixed_displacement_size=None, make_coordinates_positive: bool = True):
    r""" Tile
    Arguments:
        repeats: Integer or tuple of integers (larger than zero), how many
            repeats in each direction, to expand the given species_coordinates.
            tiling can be into a square or rectangular cell.
        noise: uniform noise in the range -noise, +noise is added to the
            coordinates to prevent exact repetition if given.
        If density is given (units of molecule / A^3), the box length is scaled
        to produce the desired molecular density. For water, density = 0.0923
        at 300 K approximately.
    """
    species, coordinates = species_coordinates
    device = coordinates.device

    coordinates = coordinates.squeeze()
    if isinstance(repeats, int):
        repeats = torch.tensor([repeats, repeats, repeats],
                               dtype=torch.long,
                               device=device)
    else:
        assert len(repeats) == 3
        repeats = torch.tensor(repeats, dtype=torch.long, device=device)
        assert (repeats >= 1).all(), 'At least one molecule should be present'
    assert coordinates.dim() == 2

    # displace coordinates so that they are all positive
    eps = torch.tensor(1e-10, device=device, dtype=coordinates.dtype)
    neg_coords = torch.where(coordinates < 0, coordinates, -eps)
    min_x = neg_coords[:, 0].min()
    min_y = neg_coords[:, 1].min()
    min_z = neg_coords[:, 2].min()
    displace_r = torch.tensor([min_x, min_y, min_z], device=device, dtype=coordinates.dtype)
    assert (displace_r <= 0).all()
    coordinates_positive = coordinates - displace_r
    coordinates_positive = coordinates + eps
    assert (coordinates_positive > 0).all()

    # get the maximum position vector in the set of coordinates ("diameter" of molecule)
    coordinates_positive = coordinates_positive.unsqueeze(0)
    max_dist_to_origin = coordinates_positive.norm(2, -1).max()
    if make_coordinates_positive:
        coordinates = coordinates_positive
    else:
        coordinates = coordinates.unsqueeze(0)

    x_disp = torch.arange(0, repeats[0], device=device)
    y_disp = torch.arange(0, repeats[1], device=device)
    z_disp = torch.arange(0, repeats[2], device=device)

    displacements = torch.cartesian_prod(x_disp, y_disp, z_disp).to(coordinates.dtype)
    num_displacements = len(displacements)
    # Calculate what the displacement size should be to match a specific density
    # If delta is given instead, then the displacement size will be the molecule
    # diameter plus that fixed delta
    # If fixed_displacement_size is given instead, then that will be the displacement
    # size, without taking into account any other factors
    msg = "delta, density and fixed_displacement_size are mutually exclusive"
    if density is not None:
        assert delta == 1.0, msg
        assert fixed_displacement_size is None, msg
        delta = (1 / repeats) * ((num_displacements / density)**(1 / 3) - max_dist_to_origin)
        box_length = max_dist_to_origin + delta
    elif fixed_displacement_size is not None:
        assert delta == 1.0, msg
        assert density is None, msg
        box_length = fixed_displacement_size
    else:
        assert density is None, msg
        assert fixed_displacement_size is None, msg
        box_length = max_dist_to_origin + delta

    displacements *= box_length
    species = species.repeat(1, num_displacements)
    coordinates = torch.cat([coordinates + d for d in displacements], dim=1)
    if noise is not None:
        coordinates += torch.empty(coordinates.shape, device=device).uniform_(-noise, noise)
    cell_length = box_length * repeats
    cell = torch.diag(torch.tensor(cell_length.cpu().numpy().tolist(), device=device, dtype=coordinates.dtype))
    return species, coordinates, cell
