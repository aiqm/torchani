r""" Utilities to generate some specific geometries"""
import typing as tp

import torch
from torch import Tensor

from torchani.utils import AtomicNumbersToMasses

Reference = tp.Literal["center_of_mass", "center_of_geometry", "origin"]


class Displacer(torch.nn.Module):
    r"""
    Displace coordinates to the center-of-mass (center_of_mass=True) frame, or
    center_of_geometry frame (center_of_mass=False) input species must be
    atomic numbers, padding atoms can be included with -1 as padding. Returns
    the centered coordinates
    """

    def __init__(
        self,
        masses: tp.Iterable[float] = (),
        reference: Reference = "center_of_mass",
        device: tp.Union[torch.device, tp.Literal["cpu", "cuda"]] = "cpu",
        dtype: torch.dtype = torch.float,
    ) -> None:
        super().__init__()
        self._atomic_masses: Tensor = torch.tensor(masses, device=device, dtype=dtype)
        self._center_of_mass = (reference == "center_of_mass")
        self._skip = (reference == "origin")
        self._converter = AtomicNumbersToMasses(masses, dtype=dtype, device=device)

    def forward(self, species: Tensor, coordinates: Tensor) -> Tensor:
        # Do nothing if reference is origin
        if self._skip:
            return coordinates

        mask = species == -1
        if self._center_of_mass:
            masses = self._converter(species)
            mass_sum = masses.unsqueeze(-1).sum(dim=1, keepdim=True)
            weights = masses.unsqueeze(-1) / mass_sum
        else:
            is_not_dummy = ~mask
            not_dummy_sum = is_not_dummy.unsqueeze(-1).sum(dim=1, keepdim=True)
            weights = is_not_dummy.unsqueeze(-1) / not_dummy_sum
        com_coordinates = coordinates * weights
        com_coordinates = com_coordinates.sum(dim=1, keepdim=True)
        centered_coordinates = coordinates - com_coordinates
        centered_coordinates[mask, :] = 0.0
        return centered_coordinates


# Convenience fn around Displacer that is non-jittable
def displace(
    atomic_numbers: Tensor,
    coordinates: Tensor,
    reference: Reference = "center_of_mass",
) -> Tensor:
    if torch.jit.is_scripting():
        raise RuntimeError(
            "'torchani.geometry.displace' doesn't support JIT, "
            " consider using torchani.geometry.Displacer instead"
        )
    return Displacer(
        device=atomic_numbers.device,
        dtype=coordinates.dtype,
        reference=reference,
    )(atomic_numbers, coordinates)


# TODO: This function is not supported anymore probably
def tile_into_tight_cell(
    species_coordinates,
    repeats=(3, 3, 3),
    noise=None,
    delta=1.0,
    density=None,
    fixed_displacement_size=None,
    make_coordinates_positive: bool = True,
):
    r"""
    Tile to generate a tight cell

    If density is given (units of molecule / A^3), the box length is scaled to
    produce the desired molecular density. For water, density = 0.0923 at 300 K
    approximately.

    Arguments:
        repeats: Integer or tuple of integers (larger than zero), how many
            repeats in each direction, to expand the given species_coordinates.
            tiling can be into a square or rectangular cell.
        noise: Uniform noise in the range -noise, +noise is added to the
            coordinates to prevent exact repetition if given.

    Returns:
        Tensor: Tiled structure
    """
    species, coordinates = species_coordinates
    device = coordinates.device

    coordinates = coordinates.squeeze()
    if isinstance(repeats, int):
        repeats = torch.tensor(
            [repeats, repeats, repeats], dtype=torch.long, device=device
        )
    else:
        assert len(repeats) == 3
        repeats = torch.tensor(repeats, dtype=torch.long, device=device)
        assert (repeats >= 1).all(), "At least one molecule should be present"
    assert coordinates.dim() == 2

    # displace coordinates so that they are all positive
    eps = torch.tensor(1e-10, device=device, dtype=coordinates.dtype)
    neg_coords = torch.where(coordinates < 0, coordinates, -eps)
    min_x = neg_coords[:, 0].min()
    min_y = neg_coords[:, 1].min()
    min_z = neg_coords[:, 2].min()
    displace_r = torch.tensor(
        [min_x, min_y, min_z], device=device, dtype=coordinates.dtype
    )
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
        delta = (1 / repeats) * (
            (num_displacements / density) ** (1 / 3) - max_dist_to_origin
        )
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
        coordinates += torch.empty(coordinates.shape, device=device).uniform_(
            -noise, noise
        )
    cell_length = box_length * repeats
    cell = torch.diag(
        torch.tensor(
            cell_length.cpu().numpy().tolist(), device=device, dtype=coordinates.dtype
        )
    )
    return species, coordinates, cell
