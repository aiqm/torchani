r"""
The assembler's responsibility is to build an ANI-style model from the
different necessary parts, in such a way that all the parts of the model
interact in the correct way and there are no compatibility issues among them.

An energy-predicting ANI-style model consists of:

- Featurizer (typically an AEVComputer, or subclass)
- Container for atomic networks (typically ANIModel)
- Atomic Networks Dict {"H": torch.nn.Module(), "C": torch.nn.Module, ...}
- Self Energies Dict (In Ha) {"H": -12.0, "C": -75.0, ...}
- Shifter (typically EnergyAdder, or subclass)

One or more PairPotentials (Typically RepulsionXTB, TwoBodyDispersion)
TBA, VDW potential, Coulombic

Each of the potentials will have their own cutoff, and the Featurizer has two
cutoffs, an angular and a radial cutoff (the radial cutoff must be larger than
the angular cutoff, and it is recommended that the angular cutoff is kept
small, roughly 3.5 Ang or less).

These pieces are assembled into a Model, which is a subclass of BuiltinModel
(or PairPotentialsModel if it has PairPotentials).

Some of the Featurizers support custom made cuda operators that accelerate them
"""
import functools
from copy import deepcopy
import warnings
import math
from dataclasses import dataclass
from collections import OrderedDict
import typing as tp

import torch
from torch import Tensor

from torchani import atomics
from torchani.models import BuiltinModel, PairPotentialsModel
from torchani.neighbors import _parse_neighborlist, NeighborlistArg
from torchani.cutoffs import _parse_cutoff_fn, Cutoff, CutoffArg
from torchani.potentials import (
    PairPotential,
    RepulsionXTB,
    TwoBodyDispersionD3,
    EnergyAdder,
)
from torchani.aev import AEVComputer, StandardAngular, StandardRadial
from torchani.nn import ANIModel, Ensemble
from torchani.utils import GSAES, sort_by_element
from torchani.storage import STATE_DICTS_DIR

ModelType = tp.Type[BuiltinModel]
FeaturizerType = tp.Type[AEVComputer]
PairPotentialType = tp.Type[PairPotential]
ContainerType = tp.Type[ANIModel]
ShifterType = tp.Type[EnergyAdder]

SFCl: tp.Tuple[str, ...] = ("S", "F", "Cl")
ELEMENTS_1X: tp.Tuple[str, ...] = ("H", "C", "N", "O")
ELEMENTS_2X: tp.Tuple[str, ...] = ELEMENTS_1X + SFCl


# "global" cutoff means the global cutoff_fn will be used
# Otherwise, a specific cutoff fn can be specified
class FeaturizerWrapper:
    def __init__(
        self,
        cls: FeaturizerType,
        radial_terms: torch.nn.Module,
        angular_terms: torch.nn.Module,
        cutoff_fn: CutoffArg = "global",
        extra: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> None:
        self.cls = cls
        self.cutoff_fn = cutoff_fn
        self.radial_terms = radial_terms
        self.angular_terms = angular_terms
        if angular_terms.cutoff > radial_terms.cutoff:  # type: ignore
            raise ValueError("Angular cutoff must be smaller or equal to radial cutoff")
        if angular_terms.cutoff <= 0 or radial_terms.cutoff <= 0:  # type: ignore
            raise ValueError("Cutoffs must be strictly positive")
        self.extra = extra

    @property
    def angular_cutoff(self) -> float:
        return tp.cast(float, self.angular_terms.cutoff)

    @property
    def radial_cutoff(self) -> float:
        return tp.cast(float, self.radial_terms.cutoff)


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
        shifter_type: ShifterType = EnergyAdder,
        model_type: ModelType = BuiltinModel,
        featurizer: tp.Optional[FeaturizerWrapper] = None,
        neighborlist: NeighborlistArg = "full_pairwise",
        periodic_table_index: bool = True,
    ) -> None:
        self._global_cutoff_fn: tp.Optional[Cutoff] = None

        self._neighborlist = _parse_neighborlist(neighborlist)
        self._featurizer = featurizer
        self._pairwise_potentials: tp.List[PairPotentialWrapper] = []

        # This part of the assembler organizes the self-energies, the
        # symbols and the atomic networks
        self._self_energies: tp.Dict[str, float] = {}
        self._fn_for_networks: tp.Optional[
            tp.Callable[[str, int], torch.nn.Module]
        ] = None
        self._atomic_networks: tp.Dict[str, torch.nn.Module] = {}
        self._shifter_type: ShifterType = shifter_type
        self._container_type: ContainerType = container_type
        self._symbols: tp.Tuple[str, ...] = tuple(symbols)
        self._ensemble_size: int = ensemble_size

        # This is the general container for all the parts of the model
        self._model_type: ModelType = model_type

        # This is a deprecated feature, it should probably not be used
        self.periodic_table_index = periodic_table_index

    def _check_symbols(self, symbols: tp.Optional[tp.Iterable[str]] = None) -> None:
        if not self.symbols:
            raise ValueError("Please set symbols before setting the gsaes as self energies")
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
    def atomic_networks(self) -> tp.OrderedDict[str, torch.nn.Module]:
        odict = OrderedDict()
        for k in self.symbols:
            odict[k] = deepcopy(self._atomic_networks[k])
        return odict

    def set_atomic_maker(
        self,
        fn: tp.Callable[[str, int], torch.nn.Module],
    ) -> None:
        self._fn_for_networks = fn

    @property
    def self_energies(self) -> tp.Dict[str, float]:
        if not self._self_energies:
            raise RuntimeError("Self energies have not been set")
        return self._self_energies

    @self_energies.setter
    def self_energies(self, value: tp.Mapping[str, float]) -> None:
        self._check_symbols(value.keys())
        self._self_energies = {k: v for k, v in value.items()}

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
                "Incorrect specification, either specify only lot, or both functional and basis set"
            )
        gsaes = GSAES[lot.lower()]
        self.self_energies = {s: gsaes[s] for s in self.symbols}

    def set_shifter(self, shifter_type: ShifterType) -> None:
        self._shifter_type = shifter_type

    def set_container(
        self,
        container_type: ContainerType,
    ) -> None:
        self._container_type = container_type

    def set_featurizer(
        self,
        featurizer_type: FeaturizerType,
        angular_terms: torch.nn.Module,
        radial_terms: torch.nn.Module,
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
        self._neighborlist = _parse_neighborlist(neighborlist)

    def set_global_cutoff_fn(
        self,
        cutoff_fn: CutoffArg,
    ) -> None:
        self._global_cutoff_fn = _parse_cutoff_fn(cutoff_fn)

    def add_pairwise_potential(
        self,
        pair_type: PairPotentialType,
        cutoff: float = math.inf,
        cutoff_fn: CutoffArg = "global",
        extra: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> None:
        if not issubclass(self._model_type, PairPotentialsModel):
            # Override the model if it is exactly equal to this class
            if self._model_type == BuiltinModel:
                self._model_type = PairPotentialsModel
            else:
                raise ValueError(
                    "The model class must support pairwise potentials in order to add potentials"
                )
        self._pairwise_potentials.append(
            PairPotentialWrapper(
                pair_type,
                cutoff=cutoff,
                cutoff_fn=cutoff_fn,
                extra=extra,
            )
        )

    def assemble(self) -> BuiltinModel:
        if not self.symbols:
            raise RuntimeError(
                "At least one symbol is needed, please set `symbols` with a sequence of chemical symbols"
            )
        if self._featurizer is None:
            raise RuntimeError(
                "Can't assemble a model without a featurizer, please `set_featurizer` first"
            )

        if all(e == 0.0 for e in self.self_energies.values()):
            warnings.warn("Assembling model with ZERO self energies!")

        feat_cutoff_fn = _parse_cutoff_fn(
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
        # This fails because the attribute is marked as final, but it should not be
        neural_networks: tp.Union[ANIModel, Ensemble]
        if self._fn_for_networks is not None:
            self._atomic_networks = {
                s: self._fn_for_networks(s, featurizer.aev_length) for s in self.symbols
            }
        else:
            raise RuntimeError(
                "Can't assemble a model without a fn for the atomic networks, please call `set_atomic_maker` first"
            )
        if self.ensemble_size > 1:
            containers = []
            for j in range(self.ensemble_size):
                containers.append(self._container_type(self.atomic_networks))
            neural_networks = Ensemble(containers)
        else:
            neural_networks = self._container_type(self.atomic_networks)
        self_energies = self.self_energies
        shifter = self._shifter_type(symbols=self.symbols, self_energies=tuple(self_energies[k] for k in self.symbols))

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
                        cutoff_fn=_parse_cutoff_fn(
                            pot.cutoff_fn, self._global_cutoff_fn
                        ),
                        **pot_kwargs,
                    )
                )
            kwargs = {"pairwise_potentials": potentials}
        else:
            kwargs = {}
        return self._model_type(
            aev_computer=featurizer,
            energy_shifter=shifter,
            elements=self.symbols,
            neural_networks=neural_networks,
            periodic_table_index=self.periodic_table_index,
            **kwargs,
        )


def load_from_neurochem(
    model_name: str,
    model_index: tp.Optional[int],
    use_cuda_extension: bool,
    use_cuaev_interface: bool,
    periodic_table_index: bool,
    pretrained: bool,
) -> BuiltinModel:
    if not pretrained:
        raise ValueError("Non pretrained models are not available from neurochem")
    from torchani.neurochem import modules_from_builtin_name
    components = modules_from_builtin_name(
        model_name,
        model_index,
        use_cuda_extension,
        use_cuaev_interface,
    )
    aev_computer, neural_networks, energy_shifter, elements = components
    return BuiltinModel(
        aev_computer,
        neural_networks,
        energy_shifter,
        elements,
        periodic_table_index=periodic_table_index,
    )


def ANI1x(
    model_index: tp.Optional[int] = None,
    pretrained: bool = True,
    neighborlist: str = "full_pairwise",
    use_cuda_extension: bool = False,
    use_cuaev_interface: bool = False,
    periodic_table_index: bool = True,
    use_neurochem_source: bool = False,
) -> BuiltinModel:
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
    if use_neurochem_source:
        return load_from_neurochem(
            model_name="ani1x",
            model_index=model_index,
            use_cuda_extension=use_cuda_extension,
            use_cuaev_interface=use_cuaev_interface,
            periodic_table_index=periodic_table_index,
            pretrained=pretrained,
        )
    asm = Assembler(ensemble_size=8, periodic_table_index=periodic_table_index)
    asm.set_symbols(ELEMENTS_1X, auto_sort=False)
    asm.set_atomic_maker(atomics.like_1x)
    asm.set_global_cutoff_fn("cosine")
    asm.set_featurizer(
        AEVComputer,
        angular_terms=StandardAngular.like_1x(),
        radial_terms=StandardRadial.like_1x(),
        extra={"use_cuda_extension": use_cuda_extension, "use_cuaev_interface": use_cuaev_interface},
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
    neighborlist: str = "full_pairwise",
    use_cuda_extension: bool = False,
    use_cuaev_interface: bool = False,
    periodic_table_index: bool = True,
    use_neurochem_source: bool = False,
) -> BuiltinModel:
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
    if use_neurochem_source:
        return load_from_neurochem(
            model_name="ani1ccx",
            model_index=model_index,
            use_cuda_extension=use_cuda_extension,
            use_cuaev_interface=use_cuaev_interface,
            periodic_table_index=periodic_table_index,
            pretrained=pretrained,
        )
    asm = Assembler(ensemble_size=8, periodic_table_index=periodic_table_index)
    asm.set_symbols(ELEMENTS_1X, auto_sort=False)
    asm.set_global_cutoff_fn("cosine")
    asm.set_featurizer(
        AEVComputer,
        radial_terms=StandardRadial.like_1ccx(),
        angular_terms=StandardAngular.like_1ccx(),
        extra={"use_cuda_extension": use_cuda_extension, "use_cuaev_interface": use_cuaev_interface},
    )
    asm.set_atomic_maker(atomics.like_1ccx)
    asm.set_neighborlist(neighborlist)
    asm.set_gsaes_as_self_energies("ccsd(t)star-cbs")
    model = asm.assemble()
    if pretrained:
        model.load_state_dict(fetch_state_dict("ani1ccx_state_dict.pt", private=False))
    return model if model_index is None else model[model_index]


def ANI2x(
    model_index: tp.Optional[int] = None,
    pretrained: bool = True,
    neighborlist: str = "full_pairwise",
    use_cuda_extension: bool = False,
    use_cuaev_interface: bool = False,
    periodic_table_index: bool = True,
    use_neurochem_source: bool = False,
) -> BuiltinModel:
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
    if use_neurochem_source:
        return load_from_neurochem(
            model_name="ani2x",
            model_index=model_index,
            use_cuda_extension=use_cuda_extension,
            use_cuaev_interface=use_cuaev_interface,
            periodic_table_index=periodic_table_index,
            pretrained=pretrained,
        )
    asm = Assembler(ensemble_size=8, periodic_table_index=periodic_table_index)
    asm.set_symbols(ELEMENTS_2X, auto_sort=False)
    asm.set_global_cutoff_fn("cosine")
    asm.set_featurizer(
        AEVComputer,
        radial_terms=StandardRadial.like_2x(),
        angular_terms=StandardAngular.like_2x(),
        extra={"use_cuda_extension": use_cuda_extension, "use_cuaev_interface": use_cuaev_interface},
    )
    asm.set_atomic_maker(atomics.like_2x)
    asm.set_neighborlist(neighborlist)
    asm.set_gsaes_as_self_energies("wb97x-631gd")
    model = asm.assemble()
    if pretrained:
        model.load_state_dict(fetch_state_dict("ani2x_state_dict.pt", private=False))
    return model if model_index is None else model[model_index]


def ANIala(
    model_index: tp.Optional[int] = None,
    pretrained: bool = True,
    neighborlist: str = "full_pairwise",
    use_cuda_extension: bool = False,
    use_cuaev_interface: bool = False,
    periodic_table_index: bool = True,
) -> BuiltinModel:
    r"""Experimental Model fine tuned to solvated frames of Ala dipeptide"""
    if model_index is not None:
        raise ValueError("Model index is not supported for ANIala")
    asm = Assembler(ensemble_size=1, periodic_table_index=periodic_table_index)
    asm.set_symbols(ELEMENTS_2X, auto_sort=False)
    asm.set_global_cutoff_fn("cosine")
    asm.set_featurizer(
        AEVComputer,
        radial_terms=StandardRadial.like_2x(),
        angular_terms=StandardAngular.like_2x(),
        extra={"use_cuda_extension": use_cuda_extension, "use_cuaev_interface": use_cuaev_interface},
    )
    asm.set_atomic_maker(atomics.like_ala)
    asm.set_neighborlist(neighborlist)
    asm.set_gsaes_as_self_energies("wb97x-631gd")
    model = asm.assemble()
    if pretrained:
        model.load_state_dict(fetch_state_dict("aniala_state_dict.pt", private=True))
    return model


def ANIdr(
    model_index: tp.Optional[int] = None,
    pretrained: bool = True,
    neighborlist: str = "full_pairwise",
    use_cuda_ops: bool = False,
    periodic_table_index: bool = True,
) -> BuiltinModel:
    """ANI model trained with both dispersion and repulsion

    The level of theory is B973c, it is an ensemble of 7 models.
    It predicts
    energies on HCNOFSCl elements
    """
    asm = Assembler(ensemble_size=7, periodic_table_index=periodic_table_index)
    asm.set_symbols(ELEMENTS_2X, auto_sort=False)
    asm.set_global_cutoff_fn("smooth2")
    asm.set_featurizer(
        AEVComputer,
        angular_terms=StandardAngular.like_2x(),
        radial_terms=StandardRadial.like_2x(),
        extra={"use_cuda_extension": use_cuda_ops, "use_cuaev_interface": use_cuda_ops},
    )
    asm.set_atomic_maker(atomics.like_dr)
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
    atomic_maker: tp.Union[str, tp.Callable[[str, int], torch.nn.Module]],
    activation: tp.Union[str, torch.nn.Module],
    bias: bool,
    use_cuda_ops: bool,
    periodic_table_index: bool,
) -> BuiltinModel:
    r"""
    Flexible builder to create ANI-style models
    """
    asm = Assembler(ensemble_size=ensemble_size, periodic_table_index=periodic_table_index)
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
    _atomic_maker = functools.partial(
        atomics._parse_atomics(atomic_maker),
        activation=atomics._parse_activation(activation),
        bias=bias,
    )
    asm.set_atomic_maker(_atomic_maker)
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
    atomic_maker: tp.Union[str, tp.Callable[[str, int], torch.nn.Module]] = "ani1x",
    activation: tp.Union[str, torch.nn.Module] = "gelu",
    bias: bool = False,
    use_cuda_ops: bool = False,
    periodic_table_index: bool = True,
) -> BuiltinModel:
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
    angular_precision: float = 12.5,
    radial_precision: float = 19.7,
    angular_zeta: float = 14.1,
    cutoff_fn: CutoffArg = "smooth2",
    neighborlist: NeighborlistArg = "full_pairwise",
    dispersion: bool = False,
    repulsion: bool = True,
    atomic_maker: tp.Union[str, tp.Callable[[str, int], torch.nn.Module]] = "ani2x",
    activation: tp.Union[str, torch.nn.Module] = "gelu",
    bias: bool = False,
    use_cuda_ops: bool = False,
    periodic_table_index: bool = True,
) -> BuiltinModel:
    r"""
    Builder that uses defaults similar to ANI2x
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

    if private:
        url = "http://moria.chem.ufl.edu/animodel/private/"
    else:
        url = "https://github.com/roitberg-group/torchani_model_zoo/releases/download/v0.1/"
    dict_ = torch.hub.load_state_dict_from_url(
        f"{url}/{state_dict_file}",
        model_dir=str(STATE_DICTS_DIR),
        map_location=torch.device("cpu"),
    )
    # if "energy_shifter.atomic_numbers" not in dict_:
    # dict_["energy_shifter.atomic_numbers"] = deepcopy(dict_["atomic_numbers"])
    return OrderedDict(dict_)
