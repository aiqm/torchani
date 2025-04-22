r"""Provides access to all published ANI models.

Provided models are subclasses of `torchani.arch.ANI`. Some models have been
published in previous articles, and some in TorchANI 2. If you use any of these models
in your work please cite the corresponding article(s).

If for a given model you discover a bug, performance problem, or incorrect behavior in
some region of chemical space, please open an issue in GitHub. The TorchANI developers
will attempt to address and document issues.

Note that parameters of the ANI models are automatically downloaded and cached the first
time they are instantiated. If this is an issue for your application we recommend you
pre-download the parameters by instantiating the models once before use.

The models can be used directly once they are instantiated. Alternatively, they can be
converted to an ASE calculator by calling ``ANI.ase``.

Some models have an interanl set of neural networks (`torchani.nn.Ensemble`), and
they output their averaged values. Individual members of these ensembles can be accessed
by indexing, and ``len(ANI)`` can be used to query the number of networks in it.

The models also have three extra entry points for more specific use cases:
atomic_energies and energies_qbcs.

All entrypoints expect a tuple of tensors ``(species, coords)`` as input, together
with two optional tensors, ``cell`` and ``pbc``. ``coords`` and ``cell`` should be in
units of Angstroms, and the output energies are always in Hartrees

For more details consult the examples documentation

.. code-block:: python

    import torchani

    model = torchani.models.ANI2x()

    # Batch of molecules
    # shape is (molecules, atoms) for atomic_nums and (molecules, atoms, 3) for coords
    atomic_nums = torch.tensor([[8, 1, 1]])
    coords = torch.tensor([[...], [...], [...]])

    # Average energies over the ensemble, for all molecules
    # Output shape is (molecules,)
    energies = model((atomic_nums, coords)).energies

    # Average atomic energies over the ensemble for the batch
    # Output shape is (molecules, atoms)
    atomic_energies = model.atomic_energies((atomic_nums, coords)).energies

    # Individual energies of the members of the ensemble
    # Output shape is (ensemble-size, molecules)
    energies = model((atomic_nums, coords), ensemble_values=True).energies

    # QBC factors are used for active learning, shape is (molecules,)
    result = model.energies_qbcs((species, coords))
    energies = result.energies
    qbcs = result.qbcs

    # Individual submodels of the ensemble can be obtained by indexing, they are also
    # subclasses of ``ANI``, with the same functionality
    submodel = model[0]
"""

import typing as tp
import importlib

from torchani.cutoffs import CutoffSmooth
from torchani.utils import SYMBOLS_2X, SYMBOLS_1X, SYMBOLS_2X_ZNUM_ORDER
from torchani.electro import ChargeNormalizer
from torchani.arch import Assembler, ANI, ANIq, _fetch_state_dict, simple_ani
from torchani.neighbors import NeighborlistArg
from torchani.potentials import TwoBodyDispersionD3, RepulsionXTB
from torchani.annotations import Device, DType
from torchani.nn._internal import _ANINetworksDiscardFirstScalar
from torchani.paths import custom_models_dir


# Protocol used by factory functions that instantiate ani models, here for reference
class _ModelFactory(tp.Protocol):
    def __call__(
        self,
        model_index: tp.Optional[int] = None,
        neighborlist: NeighborlistArg = "all_pairs",
        strategy: str = "pyaev",
        periodic_table_index: bool = True,
        device: Device = None,
        dtype: DType = None,
    ) -> ANI:
        pass


def ANI1x(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "all_pairs",
    strategy: str = "pyaev",
    periodic_table_index: bool = True,
    device: Device = None,
    dtype: DType = None,
) -> ANI:
    """The ANI-1x model as in `ani-1x_8x on GitHub`_ and `Active Learning Paper`_.

    The ANI-1x model is an ensemble of 8 networks that was trained using
    active learning on the ANI-1x dataset, the target level of theory is
    wB97X/6-31G(d). It predicts energies on HCNO elements exclusively, it
    shouldn't be used with other atom types.

    .. _ani-1x_8x on GitHub:
        https://github.com/isayev/ASE_ANI/tree/master/ani_models/ani-1x_8x

    .. _Active Learning Paper:
        https://aip.scitation.org/doi/abs/10.1063/1.5023802
    """
    asm = Assembler(periodic_table_index=periodic_table_index)
    asm.set_symbols(SYMBOLS_1X)
    asm.set_atomic_networks(ctor="ani1x")
    asm.set_global_cutoff_fn("cosine")
    asm.set_aev_computer(angular="ani1x", radial="ani1x", strategy=strategy)
    asm.set_neighborlist(neighborlist)
    asm.set_gsaes_as_self_energies("wb97x-631gd")
    model = tp.cast(ANI, asm.assemble(8))
    model.load_state_dict(_fetch_state_dict("ani1x_state_dict.pt", private=False))
    model = model if model_index is None else model[model_index]
    model.requires_grad_(False)
    # TODO: Fix this
    model.to(device=device, dtype=dtype)
    return model


def ANI1ccx(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "all_pairs",
    strategy: str = "pyaev",
    periodic_table_index: bool = True,
    device: Device = None,
    dtype: DType = None,
) -> ANI:
    """The ANI-1ccx model as in `ani-1ccx_8x on GitHub`_ and `Transfer Learning Paper`_.

    The ANI-1ccx model is an ensemble of 8 networks that was trained
    on the ANI-1ccx dataset, using transfer learning. The target accuracy
    is CCSD(T)*/CBS (CCSD(T) using the DPLNO-CCSD(T) method). It predicts
    energies on HCNO elements exclusively, it shouldn't be used with other
    atom types.

    .. _ani-1ccx_8x on GitHub:
        https://github.com/isayev/ASE_ANI/tree/master/ani_models/ani-1ccx_8x

    .. _Transfer Learning Paper:
        https://doi.org/10.26434/chemrxiv.6744440.v1
    """
    asm = Assembler(periodic_table_index=periodic_table_index)
    asm.set_symbols(SYMBOLS_1X)
    asm.set_global_cutoff_fn("cosine")
    asm.set_aev_computer(radial="ani1x", angular="ani1x", strategy=strategy)
    asm.set_atomic_networks(ctor="ani1x")
    asm.set_neighborlist(neighborlist)
    asm.set_gsaes_as_self_energies("ccsd(t)star-cbs")
    model = tp.cast(ANI, asm.assemble(8))
    model.load_state_dict(_fetch_state_dict("ani1ccx_state_dict.pt", private=False))
    model = model if model_index is None else model[model_index]
    model.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    return model


def ANI2x(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "all_pairs",
    strategy: str = "pyaev",
    periodic_table_index: bool = True,
    device: Device = None,
    dtype: DType = None,
) -> ANI:
    """The ANI-2x model as in `ANI2x Paper`_ and `ANI2x Results on GitHub`_.

    The ANI-2x model is an ensemble of 8 networks that was trained on the ANI-2x
    dataset. The target level of theory is wB97X/6-31G(d). It predicts energies on
    HCNOFSCl elements exclusively it shouldn't be used with other atom types.

    .. _ANI2x Results on GitHub:
        https://github.com/cdever01/ani-2x_results

    .. _ANI2x Paper:
        https://doi.org/10.26434/chemrxiv.11819268.v1
    """
    asm = Assembler(periodic_table_index=periodic_table_index)
    asm.set_symbols(SYMBOLS_2X)
    asm.set_global_cutoff_fn("cosine")
    asm.set_aev_computer(radial="ani2x", angular="ani2x", strategy=strategy)
    asm.set_atomic_networks(ctor="ani2x")
    asm.set_neighborlist(neighborlist)
    # The self energies are overwritten by the state dict
    asm.set_gsaes_as_self_energies("wb97x-631gd")
    model = tp.cast(ANI, asm.assemble(8))
    model.load_state_dict(_fetch_state_dict("ani2x_state_dict.pt", private=False))
    model = model if model_index is None else model[model_index]
    model.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    return model


def ANImbis(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "all_pairs",
    strategy: str = "pyaev",
    periodic_table_index: bool = True,
    device: Device = None,
    dtype: DType = None,
) -> ANIq:
    r"""
    Experimental ANI-2x model with MBIS charges
    """
    asm = Assembler(cls=ANIq, periodic_table_index=periodic_table_index)
    asm.set_symbols(SYMBOLS_2X)
    asm.set_global_cutoff_fn("cosine")
    asm.set_aev_computer(radial="ani2x", angular="ani2x", strategy=strategy)
    asm.set_atomic_networks(ctor="ani2x")
    asm.set_charge_networks(
        cls=_ANINetworksDiscardFirstScalar,
        ctor="ani2x",
        kwargs={"out_dim": 2, "bias": False, "activation": "gelu"},
        normalizer=ChargeNormalizer.from_electronegativity_and_hardness(
            asm.symbols, scale_weights_by_charges_squared=True
        ),
    )
    asm.set_neighborlist(neighborlist)
    # The self energies are overwritten by the state dict
    asm.set_gsaes_as_self_energies("wb97x-631gd")
    model = tp.cast(ANIq, asm.assemble(8))

    ani2x_state_dict = _fetch_state_dict("ani2x_state_dict.pt")
    energy_nn_state_dict = {
        k.replace("neural_networks.", ""): v
        for k, v in ani2x_state_dict.items()
        if k.endswith("weight") or k.endswith("bias")
    }
    aev_state_dict = {
        k.replace("aev_computer.", ""): v
        for k, v in ani2x_state_dict.items()
        if k.startswith("aev_computer")
    }
    shifter_state_dict = {
        "self_energies": ani2x_state_dict["energy_shifter.self_energies"]
    }
    charge_nn_state_dict = _fetch_state_dict("charge_nn_state_dict.pt", private=True)
    model.energy_shifter.load_state_dict(shifter_state_dict)
    model.potentials["nnp"].aev_computer.load_state_dict(aev_state_dict)
    model.potentials["nnp"].neural_networks.load_state_dict(energy_nn_state_dict)
    model.potentials["nnp"].charge_networks.load_state_dict(charge_nn_state_dict)
    model = model if model_index is None else model[model_index]
    model.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    return model


def ANIala(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "all_pairs",
    strategy: str = "pyaev",
    periodic_table_index: bool = True,
    device: Device = None,
    dtype: DType = None,
) -> ANI:
    r"""Experimental Model fine tuned to solvated frames of ALA dipeptide"""
    if model_index is not None:
        raise ValueError("Model index is not supported for ANIala")
    asm = Assembler(periodic_table_index=periodic_table_index)
    asm.set_symbols(SYMBOLS_2X)
    asm.set_global_cutoff_fn("cosine")
    asm.set_aev_computer(radial="ani2x", angular="ani2x", strategy=strategy)
    asm.set_atomic_networks(ctor="aniala")
    asm.set_neighborlist(neighborlist)
    asm.set_gsaes_as_self_energies("wb97x-631gd")
    model = tp.cast(ANI, asm.assemble(1))
    model.load_state_dict(_fetch_state_dict("aniala_state_dict.pt", private=True))
    model.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    return model


def ANIdr(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "all_pairs",
    strategy: str = "pyaev",
    periodic_table_index: bool = True,
    device: Device = None,
    dtype: DType = None,
) -> ANI:
    r"""
    ANI model trained with both dispersion and repulsion

    The level of theory is B973c, it is an ensemble of 7 models. It predicts energies on
    HCNOFSCl elements
    """
    asm = Assembler(periodic_table_index=periodic_table_index)
    asm.set_symbols(SYMBOLS_2X)
    asm.set_global_cutoff_fn("smooth")
    asm.set_aev_computer(angular="ani2x", radial="ani2x", strategy=strategy)
    asm.set_atomic_networks(ctor="anidr")
    asm.add_potential(RepulsionXTB, name="repulsion_xtb", cutoff=5.3)
    asm.add_potential(
        TwoBodyDispersionD3,
        name="dispersion_d3",
        cutoff=8.5,
        cutoff_fn=CutoffSmooth(order=4),
        kwargs={"functional": "B973c"},
    )
    asm.set_neighborlist(neighborlist)
    asm.set_gsaes_as_self_energies("b973c-def2mtzvp")
    model = tp.cast(ANI, asm.assemble(7))
    model.load_state_dict(_fetch_state_dict("anidr_state_dict.pt", private=True))
    model = model if model_index is None else model[model_index]
    model.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    return model


def ANI2xr(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "all_pairs",
    strategy: str = "pyaev",
    periodic_table_index: bool = True,
    device: Device = None,
    dtype: DType = None,
) -> ANI:
    r"""
    Improved ANI model trained to the 2x dataset

    Trained to the wB97X level of theory with an added repulsion potential, and smoother
    PES.
    """
    model = simple_ani(
        lot="wb97x-631gd",
        symbols=SYMBOLS_2X_ZNUM_ORDER,
        ensemble_size=8,
        dispersion=False,
        repulsion=True,
        strategy=strategy,
        neighborlist=neighborlist,
        periodic_table_index=periodic_table_index,
    )
    model.load_state_dict(_fetch_state_dict("ani2xr-preview.pt", private=True))
    model = model if model_index is None else model[model_index]
    model.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    return model


def ANI2dr(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "all_pairs",
    strategy: str = "pyaev",
    periodic_table_index: bool = True,
    device: Device = None,
    dtype: DType = None,
) -> ANI:
    r"""
    Improved ANI model trained to the 2x dataset

    Trained to the B973c level of theory with added repulsion and dispersion potentials,
    and smoother PES.
    """
    model = simple_ani(
        lot="b973c-def2mtzvp",
        symbols=SYMBOLS_2X_ZNUM_ORDER,
        ensemble_size=8,
        dispersion=True,
        repulsion=True,
        strategy=strategy,
        neighborlist=neighborlist,
        periodic_table_index=periodic_table_index,
    )
    model.load_state_dict(_fetch_state_dict("ani2dr-preview.pt", private=True))
    model = model if model_index is None else model[model_index]
    model.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    return model


def ANIr2s(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "all_pairs",
    strategy: str = "pyaev",
    periodic_table_index: bool = True,
    device: Device = None,
    dtype: DType = None,
    solvent: tp.Optional[str] = None,
) -> ANI:
    r"""The ANI-r2s model, trained to the R2SCAN-3c level of theory

    R2SCAN models are trained with the def2-mTZVPP basis set, on the ANI-2x-R2SCAN
    dataset. There are different R2SCAN models trained using different SMD implicit
    solvents that can be accessed with ``solvent='water'``, ``solvent='chcl3'``,
    or ``solvent='ch3cn'``. Alternatively, the models ``ANIr2s_water``,
    ``ANIr2s_ch3cn`` and ``ANIr2s_chcl3`` can also be instantiated directly. By default
    the vacuum model is returned.
    """
    suffix = f"{'_' + solvent if solvent is not None else ''}"
    # These models were trained with _AltSmoothCutoff, but difference is negligible
    model = simple_ani(
        lot=f"r2scan3c{suffix}-def2mtzvpp",
        symbols=SYMBOLS_2X,
        ensemble_size=8,
        dispersion=False,
        repulsion=True,
        strategy=strategy,
        neighborlist=neighborlist,
        periodic_table_index=periodic_table_index,
        repulsion_cutoff=False,
        cutoff_fn="smooth",
        # Exact reproduction of 2x aev
        radial_start=0.8,
        angular_start=0.8,
        radial_cutoff=5.1,
    )
    model.load_state_dict(
        _fetch_state_dict(f"anir2s{suffix}_state_dict.pt", private=True)
    )
    model = model if model_index is None else model[model_index]
    model.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    return model


def ANIr2s_ch3cn(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "all_pairs",
    strategy: str = "pyaev",
    periodic_table_index: bool = True,
    device: Device = None,
    dtype: DType = None,
) -> ANI:
    return ANIr2s(
        model_index,
        neighborlist,
        strategy,
        periodic_table_index,
        device,
        dtype,
        solvent="ch3cn"
    )


def ANIr2s_chcl3(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "all_pairs",
    strategy: str = "pyaev",
    periodic_table_index: bool = True,
    device: Device = None,
    dtype: DType = None,
) -> ANI:
    return ANIr2s(
        model_index,
        neighborlist,
        strategy,
        periodic_table_index,
        device,
        dtype,
        solvent="chcl3"
    )


def ANIr2s_water(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "all_pairs",
    strategy: str = "pyaev",
    periodic_table_index: bool = True,
    device: Device = None,
    dtype: DType = None,
) -> ANI:
    return ANIr2s(
        model_index,
        neighborlist,
        strategy,
        periodic_table_index,
        device,
        dtype,
        solvent="water"
    )


def SnnANI2xr(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "all_pairs",
    strategy: str = "pyaev",
    periodic_table_index: bool = True,
    device: Device = None,
    dtype: DType = None,
) -> ANI:
    r"""Custom ANI model"""
    model = simple_ani(
        lot="wb97x-631gd",
        symbols=['H', 'C', 'N', 'O', 'F', 'S', 'Cl'],
        ensemble_size=8,
        neighborlist=neighborlist,
        periodic_table_index=periodic_table_index,
        strategy=strategy,
        container="SingleNN",
        container_ctor="large",
        repulsion=True,
        sections=6,
    )
    model.load_state_dict(_fetch_state_dict("snn-ani2xr-preview.pt", private=True))
    model = model if model_index is None else model[model_index]
    model.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    return model


# Custom models
def __getattr__(name: str):
    if name == "__path__":
        # This module is not a package
        raise AttributeError
    for p in sorted(custom_models_dir().iterdir()):
        if p.name.startswith(name):
            spec = importlib.util.spec_from_file_location("model", p / "model.py")
            if spec is None:
                raise ImportError(f"{p} / model.py could not be found")
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None  # mypy
            spec.loader.exec_module(module)
            return getattr(module, name)
    raise ImportError(f"Could not find custom model {name}")
