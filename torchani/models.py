r"""Provides access to all published ANI models.

Provided models are subclasses of `torchani.arch.ANI`. Some models have been published
in previous articles, and some in TorchANI 2.0. If you use any of these models in your
work please cite the corresponding article(s).

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

import warnings
import typing as tp
import importlib.util

from torchani.utils import SYMBOLS_2X, SYMBOLS_1X, SYMBOLS_2X_ZNUM_ORDER
from torchani.potentials import (
    SeparateChargesNNPotential,
    SeparateScalarsNNPotential,
)
from torchani.electro import ChargeNormalizer
from torchani.arch import (
    Assembler,
    ANI,
    ANIq,
    ANIscalars,
    _fetch_state_dict,
    simple_ani,
)
from torchani.neighbors import NeighborlistArg
from torchani.annotations import Device, DType
from torchani.nn._internal import _ANINetworksDiscardFirstScalar
from torchani.paths import custom_models_dir

__all__ = ["ANI1x", "ANI2x", "ANI1ccx", "ANI2xr", "ANI2dr", "ANImbis", "SnnANI2xr"]


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


def ANImbisv(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "all_pairs",
    strategy: str = "pyaev",
    periodic_table_index: bool = True,
    device: Device = None,
    dtype: DType = None,
) -> ANIscalars:
    r"""
    Experimental ANI-2x model with MBIS charges
    """
    asm = Assembler(cls=ANIscalars, periodic_table_index=periodic_table_index)
    asm.set_symbols(SYMBOLS_2X)
    asm.set_global_cutoff_fn("cosine")
    asm.set_aev_computer(radial="ani2x", angular="ani2x", strategy=strategy)
    asm.set_atomic_networks(ctor="ani2x")
    asm.set_charge_normalizer(
        normalizer=ChargeNormalizer.from_electronegativity_and_hardness(
            asm.symbols, scale_weights_by_charges_squared=True
        )
    )
    asm.add_scalar_networks(
        key="atomic_charges",
        cls=_ANINetworksDiscardFirstScalar,
        ctor="ani2x",
        kwargs={"out_dim": 2, "bias": False, "activation": "gelu"},
    )
    asm.add_scalar_networks("atomic_volumes")
    asm.set_neighborlist(neighborlist)
    # The self energies are overwritten by the state dict
    asm.set_gsaes_as_self_energies("wb97x-631gd")
    model = tp.cast(ANIscalars, asm.assemble(8))

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
    model.energy_shifter.load_state_dict(shifter_state_dict)

    # TODO: Here the volume_nn_state_dict and volume_shifter_state_dict should be loaded
    # volume_shifter_state_dict: tp.Dict[str, tp.Any] = {}
    # model.volume_shifter.load_state_dict(volume_shifter_state_dict)

    # volume_nn_state_dict: tp.Dict[str, tp.Any] = {}

    charge_nn_state_dict = _fetch_state_dict("charge_nn_state_dict.pt", private=False)
    nnp = tp.cast(SeparateScalarsNNPotential, model.nnp)
    nnp.aev_computer.load_state_dict(aev_state_dict)
    nnp.neural_networks.load_state_dict(energy_nn_state_dict)
    nnp.scalar_networks["atomic_charges"].load_state_dict(charge_nn_state_dict)
    # nnp.scalar_networks["atomic_volumes"].load_state_dict(volume_nn_state_dict)
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
    charge_nn_state_dict = _fetch_state_dict("charge_nn_state_dict.pt", private=False)
    model.energy_shifter.load_state_dict(shifter_state_dict)

    nnp = tp.cast(SeparateChargesNNPotential, model.nnp)
    nnp.aev_computer.load_state_dict(aev_state_dict)
    nnp.neural_networks.load_state_dict(energy_nn_state_dict)
    nnp.charge_networks.load_state_dict(charge_nn_state_dict)
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
    model.load_state_dict(_fetch_state_dict("ani2xr.pt", private=False))
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
    model.load_state_dict(_fetch_state_dict("ani2dr.pt", private=False))
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
    r""":meta private:"""
    # r"""The ANI-r2s model, trained to the R2SCAN-3c level of theory

    # R2SCAN models are trained with the def2-mTZVPP basis set, on the ANI-2x-R2SCAN
    # dataset. There are different R2SCAN models trained using different SMD implicit
    # solvents that can be accessed with ``solvent='water'``, ``solvent='chcl3'``, or
    # ``solvent='ch3cn'``. Alternatively, the models ``ANIr2s_water``, ``ANIr2s_ch3cn``
    # and ``ANIr2s_chcl3`` can also be instantiated directly. By default the vacuum
    # model is returned.
    # """
    warnings.warn("ANIr2s is experimental. Use at your own risk")
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
        _fetch_state_dict(f"anir2s{suffix}_state_dict.pt", private=False)
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
    r""":meta private:"""
    return ANIr2s(
        model_index,
        neighborlist,
        strategy,
        periodic_table_index,
        device,
        dtype,
        solvent="ch3cn",
    )


def ANIr2s_chcl3(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "all_pairs",
    strategy: str = "pyaev",
    periodic_table_index: bool = True,
    device: Device = None,
    dtype: DType = None,
) -> ANI:
    r""":meta private:"""
    return ANIr2s(
        model_index,
        neighborlist,
        strategy,
        periodic_table_index,
        device,
        dtype,
        solvent="chcl3",
    )


def ANIr2s_water(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "all_pairs",
    strategy: str = "pyaev",
    periodic_table_index: bool = True,
    device: Device = None,
    dtype: DType = None,
) -> ANI:
    r""":meta private:"""
    return ANIr2s(
        model_index,
        neighborlist,
        strategy,
        periodic_table_index,
        device,
        dtype,
        solvent="water",
    )


def SnnANI2xr(
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
        symbols=["H", "C", "N", "O", "F", "S", "Cl"],
        ensemble_size=8,
        neighborlist=neighborlist,
        periodic_table_index=periodic_table_index,
        strategy=strategy,
        container="SingleNN",
        container_ctor="large",
        repulsion=True,
        sections=6,
    )
    model.load_state_dict(_fetch_state_dict("snn-ani2xr.pt", private=False))
    model = model if model_index is None else model[model_index]
    model.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    return model


def ANI1xnr(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "all_pairs",
    strategy: str = "pyaev",
    periodic_table_index: bool = True,
    device: Device = None,
    dtype: DType = None,
) -> ANI:
    r"""The ANI-1nxr model as in `ani-1xnr Paper`_ and `ani-1xnr on GitHub`_

    The ANI-1nxr model as in `ani-1xnr Paper`_ and `ani-1xnr on GitHub`_. This model
    model is an ensemble of 8 networks that was trained on the ANI-1xnr dataset dataset.
    The target level of theory is BLYP/TZV2P. It is a reactive potential that predicts
    energies on HCNO elements exclusively. It shouldn't be used with other atom types.

    .. _ani-1xnr on GitHub:
        https://github.com/atomistic-ml/ani-1xnr/

    .. _ani-1xnr Paper:
        https://www.nature.com/articles/s41557-023-01427-3
    """
    model = simple_ani(
        lot="blyp-tzv2p",
        symbols=["H", "C", "N", "O"],
        radial_start=0.5,
        angular_start=0.5,
        radial_cutoff=5.2,
        angular_cutoff=3.5,
        radial_precision=65.7,
        angular_precision=10.1,
        angular_zeta=14.1,
        radial_shifts=32,
        angular_shifts=8,
        sections=4,
        cutoff_fn="cosine",
        container="ANINetworks",
        container_ctor="like_2x",
        dispersion=False,
        repulsion=False,
        ensemble_size=8,
        activation="celu",
        neighborlist=neighborlist,
        periodic_table_index=periodic_table_index,
        strategy=strategy,
        self_energies="zero",  # overwritten by the state dict
        bias=True,
    )
    model.load_state_dict(_fetch_state_dict("ani1xnr.pt", private=False))
    model = model if model_index is None else model[model_index]
    model.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    return model


# Custom models
def __getattr__(name: str):
    # __mro__ needed for sphinx
    if name in ["__path__", "__mro__"]:
        # This module is not a package
        raise AttributeError
    for p in sorted(custom_models_dir().iterdir()):
        if p.name.startswith(name):
            spec = importlib.util.spec_from_file_location("model", p / "model.py")
            if spec is None:
                raise AttributeError(f"{p} / model.py could not be found")
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None  # mypy
            spec.loader.exec_module(module)
            return getattr(module, name)
    raise AttributeError(f"Could not find custom model {name}")
