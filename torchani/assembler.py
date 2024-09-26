r""" WARNING: The assembler is currently experimental and it is not considered stable
API, modify under your own risk

The assembler responsibility is to build an ANI-style model from the different
necessary parts, in such a way that all the parts of the model interact in the
correct way and there are no compatibility issues among them.

An energy-predicting ANI-style model consists of:

- Featurizer (typically a AEVComputer, which supports custom cuda ops, or subclass)
- Container for atomic networks (typically ANIModel or subclass)
- Atomic Networks Dict {"H": torch.nn.Module(), "C": torch.nn.Module, ...}
- Self Energies Dict (In Ha) {"H": -12.0, "C": -75.0, ...}
- Shifter (typically EnergyAdder)

An energy-predicting model may have PairPotentials (RepulsionXTB,
TwoBodyDispersion, VDW potential, Coulombic, etc.)

Each of the potentials has their own cutoff, and the Featurizer has two
cutoffs, an angular and a radial ona (the radial cutoff must be larger than
the angular cutoff, and it is recommended that the angular cutoff is kept
small, 3.5 Ang or less).

These pieces are assembled into a subclass of ANI (or PairPotentialsModel if it
has PairPotentials).
"""
import functools
import math
from dataclasses import dataclass
from collections import OrderedDict
import typing as tp

import torch
from torch import Tensor

from torchani import atomics
from torchani.models import (
    ANI,
    PairPotentialsModel,
    PairPotentialsChargesModel,
)
from torchani.neighbors import parse_neighborlist, NeighborlistArg
from torchani.cutoffs import parse_cutoff_fn, Cutoff, CutoffArg
from torchani.potentials import (
    PairPotential,
    RepulsionXTB,
    TwoBodyDispersionD3,
    EnergyAdder,
)
from torchani.aev import AEVComputer, StandardAngular, StandardRadial
from torchani.aev.terms import (
    RadialTermArg,
    AngularTermArg,
    parse_radial_term,
    parse_angular_term,
)
from torchani.electro import ChargeNormalizer, _AdaptedChargesContainer
from torchani.nn import ANIModel, Ensemble
from torchani.atomics import AtomicContainer, AtomicNetwork, AtomicMakerArg, AtomicMaker
from torchani.utils import GSAES, sort_by_element, SYMBOLS_1X, SYMBOLS_2X
from torchani.paths import STATE_DICTS

FeaturizerType = tp.Type[AEVComputer]
PairPotentialType = tp.Type[PairPotential]
ContainerType = tp.Type[AtomicContainer]


# "global" cutoff means the global cutoff_fn will be used
# Otherwise, a specific cutoff fn can be specified
class FeaturizerWrapper:
    def __init__(
        self,
        cls: FeaturizerType,
        radial_terms: RadialTermArg,
        angular_terms: AngularTermArg,
        cutoff_fn: CutoffArg = "global",
        extra: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> None:
        self.cls = cls
        self.cutoff_fn = cutoff_fn
        self.radial_terms = parse_radial_term(radial_terms)
        self.angular_terms = parse_angular_term(angular_terms)
        if self.angular_terms.cutoff > self.radial_terms.cutoff:
            raise ValueError("Angular cutoff must be smaller or equal to radial cutoff")
        if self.angular_terms.cutoff <= 0 or self.radial_terms.cutoff <= 0:
            raise ValueError("Cutoffs must be strictly positive")
        self.extra = extra


@dataclass
class PairPotentialWrapper:
    cls: PairPotentialType
    cutoff_fn: CutoffArg = "global"
    cutoff: float = math.inf
    extra: tp.Optional[tp.Dict[str, tp.Any]] = None


class Assembler:
    def __init__(
        self,
        ensemble_size: int = 1,
        symbols: tp.Sequence[str] = (),
        container_type: ContainerType = ANIModel,
        model_type: tp.Type[ANI] = ANI,
        featurizer: tp.Optional[FeaturizerWrapper] = None,
        neighborlist: NeighborlistArg = "full_pairwise",
        periodic_table_index: bool = True,
    ) -> None:
        self._global_cutoff_fn: tp.Optional[Cutoff] = None

        self._neighborlist = parse_neighborlist(neighborlist)
        self._featurizer = featurizer
        self._pairwise_potentials: tp.List[PairPotentialWrapper] = []

        # This part of the assembler organizes the self-energies, the
        # symbols and the atomic networks
        self._self_energies: tp.Dict[str, float] = {}
        self._fn_for_atomics: tp.Optional[AtomicMaker] = None
        self._fn_for_charges: tp.Optional[AtomicMaker] = None
        self._container_type: ContainerType = container_type
        self._charge_container_type: tp.Optional[ContainerType] = None
        self._charge_normalizer: tp.Optional[ChargeNormalizer] = None
        self._symbols: tp.Tuple[str, ...] = tuple(symbols)
        self._ensemble_size: int = ensemble_size

        # This is the general container for all the parts of the model
        self._model_type: tp.Type[ANI] = model_type

        # This is a deprecated feature, it should probably not be used
        self.periodic_table_index = periodic_table_index

    def _check_symbols(self, symbols: tp.Optional[tp.Iterable[str]] = None) -> None:
        if not self.symbols:
            raise ValueError(
                "Please set symbols before setting the gsaes as self energies"
            )
        if symbols is not None:
            if set(self.symbols) != set(symbols):
                raise ValueError(
                    f"Passed symbols don't match supported elements {self._symbols}"
                )

    @property
    def ensemble_size(self) -> int:
        return self._ensemble_size

    @ensemble_size.setter
    def ensemble_size(self, value: int) -> None:
        if value < 0:
            raise ValueError("Ensemble size must be positive")
        self._ensemble_size = value

    @property
    def elements_num(self) -> int:
        return len(self._symbols)

    @property
    def symbols(self) -> tp.Tuple[str, ...]:
        return self._symbols

    def set_symbols(self, symbols: tp.Sequence[str], auto_sort: bool = True) -> None:
        if auto_sort:
            self._symbols = sort_by_element(symbols)
        else:
            self._symbols = tuple(symbols)

    @property
    def fn_for_atomics(self) -> AtomicMaker:
        if self._fn_for_atomics is None:
            raise RuntimeError(
                "fn for atomics is not set, please call 'set_atomic_networks'"
            )
        return self._fn_for_atomics

    @property
    def fn_for_charges(self) -> AtomicMaker:
        if self._fn_for_charges is None:
            raise RuntimeError(
                "fn for charges is not set, please call 'set_charge_networks'"
            )
        return self._fn_for_charges

    @property
    def self_energies(self) -> tp.Dict[str, float]:
        if not self._self_energies:
            raise RuntimeError("Self energies have not been set")
        return self._self_energies

    @self_energies.setter
    def self_energies(self, value: tp.Mapping[str, float]) -> None:
        self._check_symbols(value.keys())
        self._self_energies = {k: v for k, v in value.items()}

    def set_zeros_as_self_energies(self) -> None:
        self._check_symbols()
        self.self_energies = {s: 0.0 for s in self.symbols}

    def set_gsaes_as_self_energies(
        self,
        lot: str = "",
        functional: str = "",
        basis_set: str = "",
    ) -> None:
        self._check_symbols()
        if (functional and basis_set) and not lot:
            lot = f"{functional}-{basis_set}"
        elif not (functional or basis_set) and lot:
            pass
        else:
            raise ValueError(
                "Incorrect specification."
                " Either specify *only* lot (preferred)"
                " or *both* functional *and* basis_set"
            )
        gsaes = GSAES[lot.lower()]
        self.self_energies = {s: gsaes[s] for s in self.symbols}

    def set_atomic_networks(
        self,
        container_type: ContainerType,
        fn: AtomicMaker,
    ) -> None:
        self._container_type = container_type
        self._fn_for_atomics = fn

    def set_charge_networks(
        self,
        container_type: ContainerType,
        fn: AtomicMaker,
        normalizer: tp.Optional[ChargeNormalizer] = None,
    ) -> None:
        if self._model_type in (ANI, PairPotentialsModel):
            self._model_type = PairPotentialsChargesModel
        elif not issubclass(self._model_type, PairPotentialsChargesModel):
            raise ValueError(
                "The model class must support charges to add a charge maker"
            )
        self._charge_container_type = container_type
        self._charge_normalizer = normalizer
        self._fn_for_charges = fn

    def set_featurizer(
        self,
        featurizer_type: FeaturizerType,
        angular_terms: AngularTermArg,
        radial_terms: RadialTermArg,
        cutoff_fn: CutoffArg = "global",
        extra: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> None:
        self._featurizer = FeaturizerWrapper(
            featurizer_type,
            cutoff_fn=cutoff_fn,
            angular_terms=angular_terms,
            radial_terms=radial_terms,
            extra=extra,
        )

    def set_neighborlist(
        self,
        neighborlist: NeighborlistArg,
    ) -> None:
        self._neighborlist = parse_neighborlist(neighborlist)

    def set_global_cutoff_fn(
        self,
        cutoff_fn: CutoffArg,
    ) -> None:
        self._global_cutoff_fn = parse_cutoff_fn(cutoff_fn)

    def add_pairwise_potential(
        self,
        pair_type: PairPotentialType,
        cutoff: float = math.inf,
        cutoff_fn: CutoffArg = "global",
        extra: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> None:
        if not issubclass(self._model_type, PairPotentialsModel):
            # Override the model if it is exactly equal to this class
            if self._model_type == ANI:
                self._model_type = PairPotentialsModel
        elif not issubclass(self._model_type, PairPotentialsModel):
            raise ValueError(
                "The model class must support pair potentials to add potentials"
            )
        self._pairwise_potentials.append(
            PairPotentialWrapper(
                pair_type,
                cutoff=cutoff,
                cutoff_fn=cutoff_fn,
                extra=extra,
            )
        )

    def build_atomic_networks(
        self,
        fn_for_networks: AtomicMaker,
        in_dim: int,
    ) -> tp.OrderedDict[str, AtomicNetwork]:
        return OrderedDict([(s, fn_for_networks(s, in_dim)) for s in self.symbols])

    def assemble(self) -> ANI:
        if not self.symbols:
            raise RuntimeError("Symbols not set. Call 'set_symbols' before assembly")
        if self._featurizer is None:
            raise RuntimeError(
                "Featurizer not set. Call 'set_featurizer' before assembly"
            )

        feat_cutoff_fn = parse_cutoff_fn(
            self._featurizer.cutoff_fn, self._global_cutoff_fn
        )

        self._featurizer.angular_terms.cutoff_fn = feat_cutoff_fn
        self._featurizer.radial_terms.cutoff_fn = feat_cutoff_fn
        feat_kwargs = {}
        if self._featurizer.extra is not None:
            feat_kwargs.update(self._featurizer.extra)

        featurizer = self._featurizer.cls(
            neighborlist=self._neighborlist,
            cutoff_fn=feat_cutoff_fn,
            angular_terms=self._featurizer.angular_terms,
            radial_terms=self._featurizer.radial_terms,
            num_species=self.elements_num,
            **feat_kwargs,  # type: ignore
        )
        neural_networks: AtomicContainer
        if self.ensemble_size > 1:
            containers = []
            for j in range(self.ensemble_size):
                containers.append(
                    self._container_type(
                        self.build_atomic_networks(
                            self.fn_for_atomics, featurizer.aev_length
                        )
                    )
                )
            neural_networks = Ensemble(containers)
        else:
            neural_networks = self._container_type(
                self.build_atomic_networks(self.fn_for_atomics, featurizer.aev_length)
            )

        charge_networks: tp.Optional[AtomicContainer] = None
        if self._charge_container_type is not None:
            charge_networks = self._charge_container_type(
                self.build_atomic_networks(self.fn_for_charges, featurizer.aev_length)
            )

        self_energies = self.self_energies
        shifter = EnergyAdder(
            symbols=self.symbols,
            self_energies=tuple(self_energies[k] for k in self.symbols),
        )
        kwargs: tp.Dict[str, tp.Any] = {}
        if self._pairwise_potentials:
            potentials = []
            for pot in self._pairwise_potentials:
                if pot.extra is not None:
                    pot_kwargs = pot.extra
                else:
                    pot_kwargs = {}
                if hasattr(pot.cls, "from_functional") and "functional" in pot_kwargs:
                    builder = pot.cls.from_functional
                else:
                    builder = pot.cls
                potentials.append(
                    builder(
                        symbols=self.symbols,
                        cutoff=pot.cutoff,
                        cutoff_fn=parse_cutoff_fn(
                            pot.cutoff_fn, self._global_cutoff_fn
                        ),
                        **pot_kwargs,
                    )
                )
            kwargs.update({"pairwise_potentials": potentials})

        if charge_networks is not None:
            kwargs.update(
                {
                    "charge_networks": charge_networks,
                    "charge_normalizer": self._charge_normalizer,
                }
            )

        return self._model_type(
            symbols=self.symbols,
            aev_computer=featurizer,
            energy_shifter=shifter,
            neural_networks=neural_networks,
            periodic_table_index=self.periodic_table_index,
            **kwargs,
        )


def ANI1x(
    model_index: tp.Optional[int] = None,
    pretrained: bool = True,
    neighborlist: NeighborlistArg = "full_pairwise",
    use_cuda_extension: bool = False,
    use_cuaev_interface: bool = False,
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
        extra={
            "use_cuda_extension": use_cuda_extension,
            "use_cuaev_interface": use_cuaev_interface,
        },
    )
    asm.set_neighborlist(neighborlist)
    asm.set_gsaes_as_self_energies("wb97x-631gd")
    model = asm.assemble()
    if pretrained:
        model.load_state_dict(fetch_state_dict("ani1x_state_dict.pt", private=False))
    return model if model_index is None else model[model_index]


def ANI1ccx(
    model_index: tp.Optional[int] = None,
    pretrained: bool = True,
    neighborlist: NeighborlistArg = "full_pairwise",
    use_cuda_extension: bool = False,
    use_cuaev_interface: bool = False,
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
        extra={
            "use_cuda_extension": use_cuda_extension,
            "use_cuaev_interface": use_cuaev_interface,
        },
    )
    asm.set_atomic_networks(ANIModel, atomics.like_1x)
    asm.set_neighborlist(neighborlist)
    asm.set_gsaes_as_self_energies("ccsd(t)star-cbs")
    model = asm.assemble()
    if pretrained:
        model.load_state_dict(fetch_state_dict("ani1ccx_state_dict.pt", private=False))
    return model if model_index is None else model[model_index]


def ANI2x(
    model_index: tp.Optional[int] = None,
    pretrained: bool = True,
    neighborlist: NeighborlistArg = "full_pairwise",
    use_cuda_extension: bool = False,
    use_cuaev_interface: bool = False,
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
        extra={
            "use_cuda_extension": use_cuda_extension,
            "use_cuaev_interface": use_cuaev_interface,
        },
    )
    asm.set_atomic_networks(ANIModel, atomics.like_2x)
    asm.set_neighborlist(neighborlist)
    # The self energies are overwritten by the state dict
    asm.set_gsaes_as_self_energies("wb97x-631gd")
    model = asm.assemble()
    if pretrained:
        model.load_state_dict(fetch_state_dict("ani2x_state_dict.pt", private=False))
    return model if model_index is None else model[model_index]


def ANImbis(
    model_index: tp.Optional[int] = None,
    pretrained: bool = True,
    neighborlist: NeighborlistArg = "full_pairwise",
    use_cuda_ops: bool = False,
    periodic_table_index: bool = True,
) -> ANI:
    r"""
    ANI-2x model with MBIS experimental charges. Note: will be removed in the
    future.
    """
    asm = Assembler(
        ensemble_size=8,
        periodic_table_index=periodic_table_index,
        model_type=PairPotentialsChargesModel,
    )
    asm.set_symbols(SYMBOLS_2X, auto_sort=False)
    asm.set_global_cutoff_fn("cosine")
    asm.set_featurizer(
        AEVComputer,
        radial_terms=StandardRadial.like_2x(),
        angular_terms=StandardAngular.like_2x(),
        extra={
            "use_cuda_extension": use_cuda_ops,
            "use_cuaev_interface": use_cuda_ops,
        },
    )
    asm.set_atomic_networks(ANIModel, atomics.like_2x)
    asm.set_charge_networks(
        _AdaptedChargesContainer,
        atomics.like_mbis_charges,
        normalizer=ChargeNormalizer.from_electronegativity_and_hardness(
            asm.symbols, scale_weights_by_charges_squared=True
        ),
    )
    asm.set_neighborlist(neighborlist)
    # The self energies are overwritten by the state dict
    asm.set_gsaes_as_self_energies("wb97x-631gd")
    model = asm.assemble()
    if pretrained:
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
        model.charge_networks.load_state_dict(charge_nn_state_dict)
    return model if model_index is None else model[model_index]


def ANIala(
    model_index: tp.Optional[int] = None,
    pretrained: bool = True,
    neighborlist: NeighborlistArg = "full_pairwise",
    use_cuda_extension: bool = False,
    use_cuaev_interface: bool = False,
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
        extra={
            "use_cuda_extension": use_cuda_extension,
            "use_cuaev_interface": use_cuaev_interface,
        },
    )
    asm.set_atomic_networks(ANIModel, atomics.like_ala)
    asm.set_neighborlist(neighborlist)
    asm.set_gsaes_as_self_energies("wb97x-631gd")
    model = asm.assemble()
    if pretrained:
        model.load_state_dict(fetch_state_dict("aniala_state_dict.pt", private=True))
    return model


def ANIdr(
    model_index: tp.Optional[int] = None,
    pretrained: bool = True,
    neighborlist: NeighborlistArg = "full_pairwise",
    use_cuda_ops: bool = False,
    periodic_table_index: bool = True,
) -> ANI:
    """ANI model trained with both dispersion and repulsion

    The level of theory is B973c, it is an ensemble of 7 models.
    It predicts
    energies on HCNOFSCl elements
    """
    asm = Assembler(ensemble_size=7, periodic_table_index=periodic_table_index)
    asm.set_symbols(SYMBOLS_2X, auto_sort=False)
    asm.set_global_cutoff_fn("smooth2")
    asm.set_featurizer(
        AEVComputer,
        angular_terms=StandardAngular.like_2x(),
        radial_terms=StandardRadial.like_2x(),
        extra={"use_cuda_extension": use_cuda_ops, "use_cuaev_interface": use_cuda_ops},
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
    if pretrained:
        model.load_state_dict(fetch_state_dict("anidr_state_dict.pt", private=True))
    return model if model_index is None else model[model_index]


def FlexANI(
    lot: str,  # functional-basis
    symbols: tp.Sequence[str],
    ensemble_size: int,
    radial_cutoff: float,
    angular_cutoff: float,
    radial_shifts: int,
    angular_shifts: int,
    angle_sections: int,
    angular_precision: float,
    radial_precision: float,
    angular_zeta: float,
    cutoff_fn: CutoffArg,
    neighborlist: NeighborlistArg,
    dispersion: bool,
    repulsion: bool,
    atomic_maker: AtomicMakerArg,
    activation: tp.Union[str, torch.nn.Module],
    bias: bool,
    use_cuda_ops: bool,
    periodic_table_index: bool,
) -> ANI:
    r"""
    Flexible builder to create ANI-style models
    """
    asm = Assembler(
        ensemble_size=ensemble_size, periodic_table_index=periodic_table_index
    )
    asm.set_symbols(symbols)
    asm.set_global_cutoff_fn(cutoff_fn)
    asm.set_featurizer(
        AEVComputer,
        radial_terms=StandardRadial.cover_linearly(
            start=0.9,
            cutoff=radial_cutoff,
            eta=radial_precision,
            num_shifts=radial_shifts,
        ),
        angular_terms=StandardAngular.cover_linearly(
            start=0.9,
            eta=angular_precision,
            zeta=angular_zeta,
            num_shifts=angular_shifts,
            num_angle_sections=angle_sections,
            cutoff=angular_cutoff,
        ),
        extra={"use_cuda_extension": use_cuda_ops, "use_cuaev_interface": use_cuda_ops},
    )
    atomic_maker = functools.partial(
        atomics.parse_atomics(atomic_maker),
        atomics.parse_activation(activation),
        bias,
    )
    asm.set_atomic_networks(ANIModel, atomic_maker)
    asm.set_neighborlist(neighborlist)
    asm.set_gsaes_as_self_energies(lot)
    if repulsion:
        asm.add_pairwise_potential(
            RepulsionXTB,
            cutoff=radial_cutoff,
        )
    if dispersion:
        asm.add_pairwise_potential(
            TwoBodyDispersionD3,
            cutoff=8.0,
            extra={"functional": lot.split("-")[0]},
        )
    return asm.assemble()


def FlexANI1(
    lot: str,  # functional-basis
    symbols: tp.Sequence[str],
    ensemble_size: int = 1,
    radial_cutoff: float = 5.2,
    angular_cutoff: float = 3.5,
    radial_shifts: int = 16,
    angular_shifts: int = 4,
    angle_sections: int = 8,
    radial_precision: float = 16.0,
    angular_precision: float = 8.0,
    angular_zeta: float = 32.0,
    cutoff_fn: CutoffArg = "smooth2",
    neighborlist: NeighborlistArg = "full_pairwise",
    dispersion: bool = False,
    repulsion: bool = True,
    atomic_maker: AtomicMakerArg = "ani1x",
    activation: tp.Union[str, torch.nn.Module] = "gelu",
    bias: bool = False,
    use_cuda_ops: bool = False,
    periodic_table_index: bool = True,
) -> ANI:
    r"""
    Builder that uses defaults similar to ANI1x
    """
    return FlexANI(
        lot=lot,
        symbols=symbols,
        ensemble_size=ensemble_size,
        radial_cutoff=radial_cutoff,
        angular_cutoff=angular_cutoff,
        radial_shifts=radial_shifts,
        angular_shifts=angular_shifts,
        angle_sections=angle_sections,
        radial_precision=radial_precision,
        angular_precision=angular_precision,
        angular_zeta=angular_zeta,
        cutoff_fn=cutoff_fn,
        neighborlist=neighborlist,
        dispersion=dispersion,
        repulsion=repulsion,
        atomic_maker=atomic_maker,
        activation=activation,
        bias=bias,
        use_cuda_ops=use_cuda_ops,
        periodic_table_index=periodic_table_index,
    )


def FlexANI2(
    lot: str,  # functional-basis
    symbols: tp.Sequence[str],
    ensemble_size: int = 1,
    radial_cutoff: float = 5.2,
    angular_cutoff: float = 3.5,
    radial_shifts: int = 16,
    angular_shifts: int = 8,
    angle_sections: int = 4,
    radial_precision: float = 19.7,
    angular_precision: float = 12.5,
    angular_zeta: float = 14.1,
    cutoff_fn: CutoffArg = "smooth2",
    neighborlist: NeighborlistArg = "full_pairwise",
    dispersion: bool = False,
    repulsion: bool = True,
    atomic_maker: AtomicMakerArg = "ani2x",
    activation: tp.Union[str, torch.nn.Module] = "gelu",
    bias: bool = False,
    use_cuda_ops: bool = False,
    periodic_table_index: bool = True,
) -> ANI:
    r"""
    Builder that uses defaults similar to ANI-2x
    """
    return FlexANI(
        lot=lot,
        symbols=symbols,
        ensemble_size=ensemble_size,
        radial_cutoff=radial_cutoff,
        angular_cutoff=angular_cutoff,
        radial_shifts=radial_shifts,
        angular_shifts=angular_shifts,
        angle_sections=angle_sections,
        radial_precision=radial_precision,
        angular_precision=angular_precision,
        angular_zeta=angular_zeta,
        cutoff_fn=cutoff_fn,
        neighborlist=neighborlist,
        dispersion=dispersion,
        repulsion=repulsion,
        atomic_maker=atomic_maker,
        activation=activation,
        bias=bias,
        use_cuda_ops=use_cuda_ops,
        periodic_table_index=periodic_table_index,
    )


def fetch_state_dict(
    state_dict_file: str,
    local: bool = False,
    private: bool = False,
) -> tp.OrderedDict[str, Tensor]:
    # If we want a pretrained model then we load the state dict from a
    # remote url or a local path
    # NOTE: torch.hub caches remote state_dicts after they have been downloaded
    if local:
        dict_ = torch.load(state_dict_file, map_location=torch.device("cpu"))
        return OrderedDict(dict_)
    PUBLIC_ZOO_URL = (
        "https://github.com/roitberg-group/torchani_model_zoo/releases/download/v0.1/"
    )
    if private:
        url = "http://moria.chem.ufl.edu/animodel/private/"
    else:
        url = PUBLIC_ZOO_URL
    dict_ = torch.hub.load_state_dict_from_url(
        f"{url}/{state_dict_file}",
        model_dir=str(STATE_DICTS),
        map_location=torch.device("cpu"),
    )
    # if "energy_shifter.atomic_numbers" not in dict_:
    # dict_["energy_shifter.atomic_numbers"] = deepcopy(dict_["atomic_numbers"])
    return OrderedDict(dict_)
