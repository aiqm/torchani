r"""
The ``models`` submodule provides access to all published ANI models, which are
subclasses of ``ANI``. Some models have been published in specific articles,
and some have been published in TorchANI 2.0. If you use any of these models in
your work please cite the corresponding article(s).

If for a given model you discover a bug, performance problem, or incorrect
behavior in some region of chemical space, please post an issue in GitHub. The
TorchANI developers will attempt to address and document issues.

Note that parameters of the ANI models are automatically downloaded and cached
the first time they are instantiated. If this is an issue for your application
we recommend you pre-download the parameters by instantiating the models before
use.

The ANI models can be used directly as callable functions once they are
instantiated. Alternatively, they can be cast to an ASE calculator by calling
``ANI.ase()``.

Some models have a "network container" that consists of an "ensemble" of neural
networks. Individual members of these ensembles can be accessed by indexing,
and ``len()`` can be used to query the number of networks in it.

The models also have three extra entry points for more specific use cases:
atomic_energies and energies_qbcs.

All entrypoints expect a tuple of tensors `(species, coordinates)` as input,
together with two optional tensors, `cell` and `pbc`.
`coordinates` and `cell` should be in units of Angstroms,
and the output energies are always in Hartrees

For more detailed examples of usage consult the examples documentation

.. code-block:: python

    import torchani

    model = torchani.models.ANI2x()

    # Batch of conformers
    # shape is (molecules, atoms) for znums and (molecules, atoms, 3) for coords
    znums = torch.tensor([[8, 1, 1]], dtype=torch.long)
    coords = torch.tensor([[...], [...], [...]], dtype=torch.float)

    # Average energies over the ensemble for the batch
    # Output shape is (molecules,)
    _, energies = model((species, coords))

    # Average atomic energies over the ensemble for the batch
    # Output shape is (molecules, atoms)
    _, atomic_ene = model.atomic_energies((species, coords))

    # Individual energies of the members of the ensemble
    # Output shape is (ensemble-size, molecules) or (ensemble-size, molecules, atoms)
    _, energies = model((species, coords), ensemble_mean=False)
    _, atomic_energies = model.atomic_energies((species, coords), ensemble_mean=False)


    # QBC factors are used for active learning, shape is (molecules,)
    _, energies, qbcs = model.energies_qbcs((species, coords))

    # Individual models of the ensemble can be obtained by indexing, they are also
    # subclasses of ``ANI``, with the same functionality
    member = model[0]
"""

from functools import partial
import typing as tp

from torchani.utils import SYMBOLS_2X, SYMBOLS_1X
from torchani.aev import AEVComputer, StandardRadial, StandardAngular
from torchani.electro import ChargeNormalizer
from torchani.assembly import Assembler, ANI, ANIq, fetch_state_dict
from torchani.neighbors import NeighborlistArg
from torchani.nn import ANIModel, _ANIModelDiscardFirstScalar
from torchani import atomics
from torchani.potentials import TwoBodyDispersionD3, RepulsionXTB


def ANI1x(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "full_pairwise",
    compute_strategy: str = "pyaev",
    periodic_table_index: bool = True,
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
    asm = Assembler(ensemble_size=8, periodic_table_index=periodic_table_index)
    asm.set_symbols(SYMBOLS_1X, auto_sort=False)
    asm.set_atomic_networks(ANIModel, atomics.like_1x)
    asm.set_global_cutoff_fn("cosine")
    asm.set_featurizer(
        AEVComputer,
        angular_terms=StandardAngular.like_1x(),
        radial_terms=StandardRadial.like_1x(),
        compute_strategy=compute_strategy,
    )
    asm.set_neighborlist(neighborlist)
    asm.set_gsaes_as_self_energies("wb97x-631gd")
    model = asm.assemble()
    model.load_state_dict(fetch_state_dict("ani1x_state_dict.pt", private=False))
    model.requires_grad_(False)
    return model if model_index is None else model[model_index]


def ANI1ccx(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "full_pairwise",
    compute_strategy: str = "pyaev",
    periodic_table_index: bool = True,
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
    asm = Assembler(ensemble_size=8, periodic_table_index=periodic_table_index)
    asm.set_symbols(SYMBOLS_1X, auto_sort=False)
    asm.set_global_cutoff_fn("cosine")
    asm.set_featurizer(
        AEVComputer,
        radial_terms=StandardRadial.like_1x(),
        angular_terms=StandardAngular.like_1x(),
        compute_strategy=compute_strategy,
    )
    asm.set_atomic_networks(ANIModel, atomics.like_1x)
    asm.set_neighborlist(neighborlist)
    asm.set_gsaes_as_self_energies("ccsd(t)star-cbs")
    model = asm.assemble()
    model.load_state_dict(fetch_state_dict("ani1ccx_state_dict.pt", private=False))
    model.requires_grad_(False)
    return model if model_index is None else model[model_index]


def ANI2x(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "full_pairwise",
    compute_strategy: str = "pyaev",
    periodic_table_index: bool = True,
) -> ANI:
    """The ANI-2x model as in `ANI2x Paper`_ and `ANI2x Results on GitHub`_.

    The ANI-2x model is an ensemble of 8 networks that was trained on the
    ANI-2x dataset. The target level of theory is wB97X/6-31G(d). It predicts
    energies on HCNOFSCl elements exclusively it shouldn't be used with other
    atom types.

    .. _ANI2x Results on GitHub:
        https://github.com/cdever01/ani-2x_results

    .. _ANI2x Paper:
        https://doi.org/10.26434/chemrxiv.11819268.v1
    """
    asm = Assembler(ensemble_size=8, periodic_table_index=periodic_table_index)
    asm.set_symbols(SYMBOLS_2X, auto_sort=False)
    asm.set_global_cutoff_fn("cosine")
    asm.set_featurizer(
        AEVComputer,
        radial_terms=StandardRadial.like_2x(),
        angular_terms=StandardAngular.like_2x(),
        compute_strategy=compute_strategy,
    )
    asm.set_atomic_networks(ANIModel, atomics.like_2x)
    asm.set_neighborlist(neighborlist)
    # The self energies are overwritten by the state dict
    asm.set_gsaes_as_self_energies("wb97x-631gd")
    model = asm.assemble()
    model.load_state_dict(fetch_state_dict("ani2x_state_dict.pt", private=False))
    model.requires_grad_(False)
    return model if model_index is None else model[model_index]


def ANImbis(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "full_pairwise",
    compute_strategy: str = "pyaev",
    periodic_table_index: bool = True,
) -> ANI:
    r"""
    ANI-2x model with MBIS experimental charges. Note: will be removed in the
    future.
    """
    if compute_strategy not in ["pyaev", "cuaev"]:
        raise ValueError(f"Unavailable strategy for ANImbis: {compute_strategy}")
    asm = Assembler(
        ensemble_size=8,
        periodic_table_index=periodic_table_index,
        model_type=ANIq,
        output_labels=("energies", "atomic_charges")
    )
    asm.set_symbols(SYMBOLS_2X, auto_sort=False)
    asm.set_global_cutoff_fn("cosine")
    asm.set_featurizer(
        AEVComputer,
        radial_terms=StandardRadial.like_2x(),
        angular_terms=StandardAngular.like_2x(),
        compute_strategy=compute_strategy,
    )
    asm.set_atomic_networks(ANIModel, atomics.like_2x)

    asm.set_charge_networks(
        _ANIModelDiscardFirstScalar,
        partial(atomics.like_2x, out_dim=2, bias=False, activation="gelu"),
        normalizer=ChargeNormalizer.from_electronegativity_and_hardness(
            asm.symbols, scale_weights_by_charges_squared=True
        ),
    )
    asm.set_neighborlist(neighborlist)
    # The self energies are overwritten by the state dict
    asm.set_gsaes_as_self_energies("wb97x-631gd")
    model = asm.assemble()

    ani2x_state_dict = fetch_state_dict("ani2x_state_dict.pt")
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
    charge_nn_state_dict = fetch_state_dict("charge_nn_state_dict.pt", private=True)
    model.energy_shifter.load_state_dict(shifter_state_dict)
    model.aev_computer.load_state_dict(aev_state_dict)
    model.neural_networks.load_state_dict(energy_nn_state_dict)
    model.potentials[0].charge_networks.load_state_dict(charge_nn_state_dict)
    model.requires_grad_(False)
    return model if model_index is None else model[model_index]


def ANIala(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "full_pairwise",
    compute_strategy: str = "pyaev",
    periodic_table_index: bool = True,
) -> ANI:
    r"""Experimental Model fine tuned to solvated frames of Ala dipeptide"""
    if model_index is not None:
        raise ValueError("Model index is not supported for ANIala")
    asm = Assembler(ensemble_size=1, periodic_table_index=periodic_table_index)
    asm.set_symbols(SYMBOLS_2X, auto_sort=False)
    asm.set_global_cutoff_fn("cosine")
    asm.set_featurizer(
        AEVComputer,
        radial_terms=StandardRadial.like_2x(),
        angular_terms=StandardAngular.like_2x(),
        compute_strategy=compute_strategy,
    )
    asm.set_atomic_networks(ANIModel, atomics.like_ala)
    asm.set_neighborlist(neighborlist)
    asm.set_gsaes_as_self_energies("wb97x-631gd")
    model = asm.assemble()
    model.load_state_dict(fetch_state_dict("aniala_state_dict.pt", private=True))
    model.requires_grad_(False)
    return model


def ANIdr(
    model_index: tp.Optional[int] = None,
    neighborlist: NeighborlistArg = "full_pairwise",
    compute_strategy: str = "pyaev",
    periodic_table_index: bool = True,
) -> ANI:
    """ANI model trained with both dispersion and repulsion

    The level of theory is B973c, it is an ensemble of 7 models.
    It predicts
    energies on HCNOFSCl elements
    """
    if compute_strategy not in ["pyaev", "cuaev"]:
        raise ValueError(f"Unavailable strategy for ANImbis: {compute_strategy}")
    asm = Assembler(ensemble_size=7, periodic_table_index=periodic_table_index)
    asm.set_symbols(SYMBOLS_2X, auto_sort=False)
    asm.set_global_cutoff_fn("smooth2")
    asm.set_featurizer(
        AEVComputer,
        angular_terms=StandardAngular.like_2x(),
        radial_terms=StandardRadial.like_2x(),
        compute_strategy=compute_strategy,
    )
    asm.set_atomic_networks(ANIModel, atomics.like_dr)
    asm.add_pairwise_potential(
        RepulsionXTB,
        cutoff=5.3,
    )
    asm.add_pairwise_potential(
        TwoBodyDispersionD3,
        cutoff=8.5,
        cutoff_fn="smooth4",
        extra={"functional": "B973c"},
    )
    asm.set_neighborlist(neighborlist)
    asm.set_gsaes_as_self_energies("b973c-def2mtzvp")
    model = asm.assemble()
    model.load_state_dict(fetch_state_dict("anidr_state_dict.pt", private=True))
    model.requires_grad_(False)
    return model if model_index is None else model[model_index]
