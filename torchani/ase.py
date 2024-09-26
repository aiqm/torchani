"""Tools for interfacing with the atomic simulation environment `ASE`_.


.. _ASE:
    https://wiki.fysik.dtu.dk/ase
"""

import warnings

import torch

try:
    import ase.units
    import ase.calculators.calculator
except ImportError:
    raise ImportError(
        "Error when trying to import 'torchani.ase':"
        " The ASE package could not be found. 'torchani.ase' and the '*.ase()' methods"
        " of models won't be available. Please install ase if you want to use them."
        " ('conda install ase' or 'pip install ase')"
    ) from None

from torchani.utils import map_to_central


class Calculator(ase.calculators.calculator.Calculator):
    """TorchANI calculator for ASE

    ANI models can be converted to their ASE Calculator form by calling the
    ``ANI.ase()`` method.

    .. code-block:: python

        import torchani
        model = torchani.models.ANI1x()
        calc = model.ase()  # Convert model into its ASE Calculator form

    Arguments:
        model (:class:`torch.nn.Module`): neural network potential model
            that convert coordinates into energies.
        overwrite (bool): After wrapping atoms into central box, whether
            to replace the original positions stored in :class:`ase.Atoms`
            object with the wrapped positions.
        stress_partial_fdotr (bool): Whether to use partial_fdotr approach to
            calculate stress. This approach does not need the cell's box
            information and can be used for multiple domians when running
            parallel on multi-GPUs. Default as False.
        stress_numerical (bool): Whether to calculate numerical stress
    """

    implemented_properties = ["energy", "forces", "stress", "free_energy"]

    def __init__(
        self,
        model,
        overwrite: bool = False,
        stress_partial_fdotr: bool = False,
        stress_numerical: bool = False,
    ):
        super().__init__()
        self.model = model
        # Since ANI is used in inference mode, no gradients on model parameters
        # are required here
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.overwrite = overwrite
        self.stress_partial_fdotr = stress_partial_fdotr
        self.stress_numerical = stress_numerical

        a_parameter = next(self.model.parameters())
        self.device = a_parameter.device
        self.dtype = a_parameter.dtype

        try:
            periodic_table_index = model.periodic_table_index
        except AttributeError:
            periodic_table_index = False

        if not periodic_table_index:
            raise ValueError("ASE models must have periodic_table_index=True")

    def calculate(
        self,
        atoms=None,
        properties=["energy"],
        system_changes=ase.calculators.calculator.all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        assert self.atoms is not None  # mypy
        cell = torch.tensor(
            self.atoms.get_cell(complete=True).array,
            dtype=self.dtype,
            device=self.device,
        )
        pbc = torch.tensor(self.atoms.get_pbc(), dtype=torch.bool, device=self.device)
        pbc_enabled = pbc.any().item()

        species = torch.tensor(
            self.atoms.get_atomic_numbers(), dtype=torch.long, device=self.device
        )
        species = species.unsqueeze(0)
        coordinates = torch.tensor(self.atoms.get_positions())
        coordinates = (
            coordinates.to(self.device)
            .to(self.dtype)
            .requires_grad_("forces" in properties)
        )

        if pbc_enabled and self.overwrite and atoms is not None:
            warnings.warn(
                "'overwrite' is set for PBC calculation."
                " Information about crossing PBC boundaries *will be lost*"
            )
            coordinates = map_to_central(coordinates, cell, pbc)
            atoms.set_positions(coordinates.detach().cpu().reshape(-1, 3).numpy())

        if "stress" in properties and not (
            self.stress_partial_fdotr or self.stress_numerical
        ):
            scaling = torch.eye(
                3, requires_grad=True, dtype=self.dtype, device=self.device
            )
            coordinates = coordinates @ scaling
        coordinates = coordinates.unsqueeze(0)

        if pbc_enabled:
            if "stress" in properties and not (
                self.stress_partial_fdotr or self.stress_numerical
            ):
                cell = cell @ scaling
            energy = self.model((species, coordinates), cell=cell, pbc=pbc).energies
        else:
            energy = self.model((species, coordinates)).energies

        energy = energy * ase.units.Hartree
        self.results["energy"] = energy.item()
        self.results["free_energy"] = energy.item()

        if "forces" in properties:
            forces = self._get_ani_forces(coordinates, energy, properties)
            self.results["forces"] = forces.squeeze(0).to("cpu").numpy()

        if "stress" in properties:
            volume = self.atoms.get_volume()
            if self.stress_partial_fdotr:
                diff_vectors = self.model.aev_computer.neighborlist.get_diff_vectors()
                stress = self._get_stress_partial_fdotr(diff_vectors, energy, volume)
                self.results["stress"] = stress.detach().cpu().numpy()
            elif self.stress_numerical:
                self.results["stress"] = self.calculate_numerical_stress(
                    atoms if atoms is not None else self.atoms
                )
            else:
                stress = torch.autograd.grad(energy.squeeze(), scaling)[0] / volume
                self.results["stress"] = stress.detach().cpu().numpy()

    def _get_ani_forces(self, coordinates, energy, properties):
        return -torch.autograd.grad(
            energy.squeeze(), coordinates, retain_graph="stress" in properties
        )[0]

    @staticmethod
    def _get_stress_partial_fdotr(diff_vectors, energy, volume):
        dEdR = torch.autograd.grad(energy.squeeze(), diff_vectors, retain_graph=True)[0]
        virial = dEdR.transpose(0, 1) @ diff_vectors
        stress = virial / volume
        return stress

    @staticmethod
    # TODO figure out a way to test
    def _get_stress_forces_partial_fdotr(coordinates, diff_vectors, energy, volume):
        forces, dEdR = torch.autograd.grad(
            energy.squeeze(), [coordinates, diff_vectors], retain_graph=True
        )
        forces = torch.neg(forces)

        virial = dEdR.transpose(0, 1) @ diff_vectors
        stress = virial / volume
        return forces, stress
