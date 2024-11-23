import typing as tp

from torch import Tensor

from torchani.assembly import ANIq, ANI
from torchani._grad import (
    forces as _calc_forces,
    forces_and_hessians as _calc_forces_and_hessians,
)


def single_point(
    model: tp.Union[ANI, ANIq],
    species: Tensor,
    coordinates: Tensor,
    cell: tp.Optional[Tensor] = None,
    pbc: tp.Optional[Tensor] = None,
    charge: int = 0,
    forces: bool = False,
    hessians: bool = False,
    atomic_energies: bool = False,
    atomic_charges: bool = False,
    atomic_charges_grad: bool = False,
    ensemble_values: bool = False,
    keep_vars: bool = False,
) -> tp.Dict[str, Tensor]:
    r"""Calculate properties for a batch of molecules

    This is the main entrypoint of ANI-style models

    Args:
        species: |atomic_nums|
        coordinates: |coords|
        cell: |cell|
        pbc: |pbc|
        charge: The total charge of the molecules. Only
            the scalar 0 is currently supported.
        forces: Calculate the associated forces. Shape ``(molecules, atoms, 3)``
        hessians: Calculate the hessians. Shape is
            ``(molecules, atoms * 3, atoms * 3)``
        atomic_energies: Perform atomic decoposition of the energies
        atomic_charges: Only for models that support it, output atomic charges.
            Shape ``(molecules, atoms)``
        atomic_charges_grad: Only for models that support it, output atomic charge
            gradients. Shape ``(molecules, atoms, 3)``.
        ensemble_values: Differentiate values of different models of the ensemble
            Also output ensemble standard deviation and qbc factors
        keep_vars: The output scalars are detached from the graph unless
            ``keep_vars=True``.
    Returns:
        Result of the single point calculation. Dictionary that maps strings to
        various result tensors.
    """
    saved_requires_grad = coordinates.requires_grad
    if forces or hessians or atomic_charges_grad:
        coordinates.requires_grad_(True)
    result = model(
        species_coordinates=(species, coordinates),
        cell=cell,
        pbc=pbc,
        charge=charge,
        atomic=atomic_energies,
        ensemble_values=ensemble_values,
    )
    energies = result.energies
    out: tp.Dict[str, Tensor] = {}
    if atomic_charges:
        if not hasattr(result, "atomic_charges"):
            raise ValueError("Model doesn't support atomic charges")
        out["atomic_charges"] = result.atomic_charges
        if atomic_charges_grad:
            retain = forces or hessians
            out["atomic_charges_grad"] = -_calc_forces(
                result.atomic_charges, coordinates, retain_graph=retain
            )

    if ensemble_values:
        if atomic_energies:
            out["atomic_energies"] = energies.mean(dim=0)
            _values = energies.sum(dim=-1)
        else:
            _values = energies
        out["energies"] = _values.mean(dim=0)

        if _values.shape[0] == 1:
            out["ensemble_std"] = _values.new_zeros(energies.shape)
        else:
            out["ensemble_std"] = _values.std(dim=0, unbiased=True)
        out["ensemble_values"] = _values

        if _values.shape[0] == 1:
            qbc_factors = _values.new_zeros(_values.shape).squeeze(0)
        else:
            # std is taken across ensemble members
            qbc_factors = _values.std(0, unbiased=True)
        # rho's (qbc factors) are weighted by dividing by the square root of
        # the number of atoms in each molecule
        num_atoms = (species >= 0).sum(dim=1, dtype=energies.dtype)
        qbc_factors = qbc_factors / num_atoms.sqrt()
        assert qbc_factors.shape == out["energies"].shape
        out["qbcs"] = qbc_factors
    else:
        if atomic_energies:
            out["energies"] = energies.sum(dim=-1)
            out["atomic_energies"] = energies
        else:
            out["energies"] = energies
    if hessians:
        _forces, _hessians = _calc_forces_and_hessians(out["energies"], coordinates)
        out["forces"], out["hessians"] = _forces, _hessians
        if forces:
            out["forces"] = _forces
    elif forces:
        out["forces"] = _calc_forces(out["energies"], coordinates)
    coordinates.requires_grad_(saved_requires_grad)
    if not keep_vars:
        out = {k: v.detach() for k, v in out.items()}
    return out
