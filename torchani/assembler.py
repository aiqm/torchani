r"""
WARNING: This module is currently experimental and untested, use under your own risk!

The assembler's responsibility is to build an ANI-style model from the
different necessary parts, in such a way that all the parts of the model
interact in the correct way and there are no compatibility issues among them.

An energy-predicting ANI-style model consists of:

- Featurizer (typically an AEVComputer, or subclass)
- Container for atomic networks (typically ANIModel)
- Atomic Networks Dict {"H": torch.nn.Module(), "C": torch.nn.Module, ...}
- Self Energies Dict (In Ha) {"H": -12.0, "C": -75.0, ...}
- Shifter (typically EnergyShifter, or subclass)

One or more PairwisePotentials (Typically RepulsionXTB, TwoBodyDispersion)
TBA, VDW potential, Coulombic

Each of the potentials will have their own cutoff, and the Featurizer has two
cutoffs, an angular and a radial cutoff (the radial cutoff must be larger than
the angular cutoff, and it is recommended that the angular cutoff is kept
small, roughly 3.5 Ang or less).

These pieces are assembled into a Model, which is a subclass of BuiltinModel
(or BuiltinModelPairInteractions if it has PairwisePotentials).

Some of the Featurizers support custom made cuda operators that accelerate them
"""
from copy import deepcopy
from pathlib import Path
import warnings
import math
from dataclasses import dataclass
from collections import OrderedDict
import typing as tp

import torch
from torch import Tensor

from torchani import atomics
from torchani.models import BuiltinModel, BuiltinModelPairInteractions
from torchani.neighbors import BaseNeighborlist
from torchani.cutoffs import _parse_cutoff_fn, Cutoff
from torchani.potentials import (
    PairwisePotential,
    RepulsionXTB,
    TwoBodyDispersionD3,
)
from torchani.aev import AEVComputer, StandardAngular, StandardRadial
from torchani.nn import ANIModel, Ensemble
from torchani.utils import EnergyShifter, GSAES, ATOMIC_NUMBERS

ModelType = tp.Type[BuiltinModel]
NeighborlistType = tp.Type[BaseNeighborlist]
FeaturizerType = tp.Type[AEVComputer]
PairwisePotentialType = tp.Type[PairwisePotential]
AtomicContainerType = tp.Type[ANIModel]
ShifterType = tp.Type[EnergyShifter]

SFCl = ("S", "F", "Cl")
ELEMENTS_1X = ("H", "C", "N", "O")
ELEMENTS_2X = ELEMENTS_1X + SFCl


def sort_by_element(it: tp.Iterable[str]) -> tp.Tuple[str, ...]:
    return tuple(sorted(it, key=lambda x: ATOMIC_NUMBERS[x]))


# "global" cutoff means the global cutoff_fn will be used
# Otherwise, a specific cutoff fn can be specified
class FeaturizerWrapper:
    def __init__(
        self,
        cls: FeaturizerType,
        radial_terms: torch.nn.Module,
        angular_terms: torch.nn.Module,
        cutoff_fn: tp.Union[Cutoff, str] = "global",
        cuda_ops: bool = False,
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
        self.cuda_ops = cuda_ops
        self.extra = extra

    @property
    def angular_cutoff(self) -> float:
        return tp.cast(float, self.angular_terms.cutoff)

    @property
    def radial_cutoff(self) -> float:
        return tp.cast(float, self.radial_terms.cutoff)


@dataclass
class PotentialWrapper:
    cls: PairwisePotentialType
    extra: tp.Optional[tp.Dict[str, tp.Any]] = None
    cutoff_fn: tp.Union[Cutoff, str] = "global"
    cutoff: float = math.inf


class Assembler:
    def __init__(
        self,
        ensemble_size: int = 1,
        symbols: tp.Sequence[str] = (),
        atomic_container_type: AtomicContainerType = ANIModel,
        shifter_type: ShifterType = EnergyShifter,
        model_type: ModelType = BuiltinModel,
        featurizer: tp.Optional[FeaturizerWrapper] = None,
        neighborlist: tp.Union[NeighborlistType, str] = "full_pairwise",
        periodic_table_index: bool = True,
    ) -> None:
        self._global_cutoff_fn: tp.Optional[Cutoff] = None

        self._neighborlist_type = neighborlist
        self._featurizer = featurizer
        self._pairwise_potentials: tp.List[PotentialWrapper] = []

        # This part of the assembler organizes the self-energies, the
        # symbols and the atomic networks
        self._self_energies: tp.Dict[str, float] = {}
        self._fn_for_networks: tp.Optional[
            tp.Callable[[str, int], torch.nn.Module]
        ] = None
        self._atomic_networks: tp.Dict[str, torch.nn.Module] = {}
        self._shifter_type: ShifterType = shifter_type
        self._atomic_container_type: AtomicContainerType = atomic_container_type
        self._symbols: tp.Tuple[str, ...] = tuple(symbols)
        self._ensemble_size: int = ensemble_size

        # This is the general container for all the parts of the model
        self._model_type: ModelType = model_type

        # This is a deprecated feature, it should probably not be used
        self.periodic_table_index = periodic_table_index

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

    @symbols.setter
    def symbols(self, symbols: tp.Sequence[str]) -> None:
        self._symbols = sort_by_element(symbols)

    @property
    def atomic_networks(self) -> tp.OrderedDict[str, torch.nn.Module]:
        odict = OrderedDict()
        for k in self.symbols:
            odict[k] = deepcopy(self._atomic_networks[k])
        return odict

    def set_atomic_maker(
        self,
        fn: tp.Callable[[str, int], torch.nn.Module],
        symbols: tp.Sequence[str] = (),
    ) -> None:
        if not self.symbols:
            self.symbols = sort_by_element(symbols)
        elif symbols:
            if set(self.symbols) != set(symbols):
                raise ValueError(
                    f"Atomic networks don't match supported elements {self._symbols}"
                )
        self._fn_for_networks = fn

    @property
    def self_energies(self) -> tp.OrderedDict[str, float]:
        odict = OrderedDict()
        for k in self.symbols:
            odict[k] = self._self_energies[k]
        return odict

    @self_energies.setter
    def self_energies(self, value: tp.Mapping[str, float]) -> None:
        if not self.symbols:
            self.symbols = sort_by_element(value.keys())
        elif set(self.symbols) != set(value.keys()):
            raise ValueError(
                f"Self energies don't match supported elements {self._symbols}"
            )
        self._self_energies = {k: v for k, v in value.items()}

    def set_gsaes_as_self_energies(
        self,
        lot: str = "",
        functional: str = "",
        basis_set: str = "",
        symbols: tp.Iterable[str] = (),
    ) -> None:
        if (functional and basis_set) and not lot:
            lot = f"{functional}-{basis_set}"
        elif not (functional or basis_set) and lot:
            pass
        else:
            raise ValueError(
                "Incorrect specification, either specify only lot, or both functional and basis set"
            )

        if not symbols:
            symbols = self.symbols
        gsaes = GSAES[lot.lower()]
        self.self_energies = OrderedDict([(s, gsaes[s]) for s in symbols])

    def set_shifter(self, shifter_type: ShifterType) -> None:
        self._shifter_type = shifter_type

    def set_atomic_container(
        self,
        atomic_container_type: AtomicContainerType,
    ) -> None:
        self._atomic_container_type = atomic_container_type

    def set_featurizer(
        self,
        featurizer_type: FeaturizerType,
        angular_terms: torch.nn.Module,
        radial_terms: torch.nn.Module,
        cutoff_fn: tp.Union[Cutoff, str] = "global",
        cuda_ops: bool = False,
    ) -> None:
        self._featurizer = FeaturizerWrapper(
            featurizer_type,
            cutoff_fn=cutoff_fn,
            angular_terms=angular_terms,
            radial_terms=radial_terms,
            cuda_ops=cuda_ops,
        )

    def set_neighborlist(
        self,
        neighborlist_type: tp.Union[NeighborlistType, str],
    ) -> None:
        if isinstance(neighborlist_type, str) and neighborlist_type not in [
            "full_pairwise",
            "cell_list",
        ]:
            raise ValueError("Unsupported neighborlist")
        self._neighborlist_type = neighborlist_type

    def set_global_cutoff_fn(
        self,
        cutoff_fn: tp.Union[Cutoff, str],
    ) -> None:
        self._global_cutoff_fn = _parse_cutoff_fn(cutoff_fn)

    def add_pairwise_potential(
        self,
        pair_type: PairwisePotentialType,
        cutoff: float = math.inf,
        cutoff_fn: tp.Union[Cutoff, str] = "global",
        extra: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> None:
        if not issubclass(self._model_type, BuiltinModelPairInteractions):
            # Override the model if it is exactly equal to this class
            if self._model_type == BuiltinModel:
                self._model_type = BuiltinModelPairInteractions
            else:
                raise ValueError(
                    "The model class must support pairwise potentials in order to add potentials"
                )
        self._pairwise_potentials.append(
            PotentialWrapper(
                pair_type,
                cutoff=cutoff,
                cutoff_fn=cutoff_fn,
                extra=extra,
            )
        )

    def assemble(self) -> BuiltinModel:
        # Here it is necessary to get the largest cutoff to attach to the neighborlist, right?
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

        cuts = [pot.cutoff for pot in self._pairwise_potentials]
        cuts.extend([self._featurizer.angular_cutoff, self._featurizer.radial_cutoff])
        max_cutoff = max(cuts)

        feat_cutoff_fn = _parse_cutoff_fn(
            self._featurizer.cutoff_fn, self._global_cutoff_fn
        )

        self._featurizer.angular_terms.cutoff_fn = feat_cutoff_fn
        self._featurizer.radial_terms.cutoff_fn = feat_cutoff_fn
        if self._featurizer.cls == AEVComputer:
            feat_kwargs = {
                "use_cuda_extension": self._featurizer.cuda_ops,
                "use_cuaev_interface": self._featurizer.cuda_ops,
            }
        else:
            feat_kwargs = {}
        if self._featurizer.extra is not None:
            feat_kwargs.update(self._featurizer.extra)

        featurizer = self._featurizer.cls(
            neighborlist=self._neighborlist_type,
            cutoff_fn=feat_cutoff_fn,
            angular_terms=self._featurizer.angular_terms,
            radial_terms=self._featurizer.radial_terms,
            num_species=self.elements_num,
            **feat_kwargs,  # type: ignore
        )
        # This fails because the attribute is marked as final, but it should not be
        featurizer.neighborlist.cutoff = max_cutoff  # type: ignore
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
                containers.append(self._atomic_container_type(self.atomic_networks))
            neural_networks = Ensemble(containers)
        else:
            neural_networks = self._atomic_container_type(self.atomic_networks)
        shifter = self._shifter_type(tuple(self.self_energies.values()))

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


def ANI1x(
    pretrained: bool = True,
    neighborlist: str = "full_pairwise",
    cuda_ops: bool = False,
) -> BuiltinModel:
    asm = Assembler(ensemble_size=8)
    asm.symbols = ELEMENTS_1X
    asm.set_atomic_maker(atomics.like_1x)
    asm.set_global_cutoff_fn("cosine")
    asm.set_featurizer(
        AEVComputer,
        angular_terms=StandardAngular.like_1x(),
        radial_terms=StandardRadial.like_1x(),
        cuda_ops=cuda_ops,
    )
    asm.set_neighborlist(neighborlist)
    asm.set_gsaes_as_self_energies("wb97x-631gd")
    model = asm.assemble()
    if pretrained:
        model.load_state_dict(fetch_state_dict("ani1x_state_dict.pt", private=False))
    return model


def ANI1ccx(
    pretrained: bool = True,
    neighborlist: str = "full_pairwise",
    cuda_ops: bool = False,
) -> BuiltinModel:
    asm = Assembler(ensemble_size=8)
    asm.symbols = ELEMENTS_1X
    asm.set_global_cutoff_fn("cosine")
    asm.set_featurizer(
        AEVComputer,
        radial_terms=StandardRadial.like_1ccx(),
        angular_terms=StandardAngular.like_1ccx(),
        cuda_ops=cuda_ops,
    )
    asm.set_atomic_maker(atomics.like_1ccx)
    asm.set_neighborlist(neighborlist)
    asm.set_gsaes_as_self_energies("ccsd(t)star-cbs")
    model = asm.assemble()
    if pretrained:
        model.load_state_dict(fetch_state_dict("ani1ccx_state_dict.pt", private=False))
    return model


def ANI2x(
    pretrained: bool = True,
    neighborlist: str = "full_pairwise",
    cuda_ops: bool = False,
) -> BuiltinModel:
    asm = Assembler(ensemble_size=8)
    asm.symbols = ELEMENTS_2X
    asm.set_global_cutoff_fn("cosine")
    asm.set_featurizer(
        AEVComputer,
        radial_terms=StandardRadial.like_2x(),
        angular_terms=StandardAngular.like_2x(),
        cuda_ops=cuda_ops,
    )
    asm.set_atomic_maker(atomics.like_2x)
    asm.set_neighborlist(neighborlist)
    asm.set_gsaes_as_self_energies("wb97x-631gd")
    model = asm.assemble()
    if pretrained:
        model.load_state_dict(fetch_state_dict("ani2x_state_dict.pt", private=False))
    return model


def ANIala(
    pretrained: bool = True,
    neighborlist: str = "full_pairwise",
    cuda_ops: bool = False,
) -> BuiltinModel:
    r"""Experimental Model fine tuned to solvated frames of Ala dipeptide"""
    asm = Assembler(ensemble_size=1)
    asm.symbols = ELEMENTS_2X
    asm.set_global_cutoff_fn("cosine")
    asm.set_featurizer(
        AEVComputer,
        radial_terms=StandardRadial.like_2x(),
        angular_terms=StandardAngular.like_2x(),
        cuda_ops=cuda_ops,
    )
    asm.set_atomic_maker(atomics.like_ala)
    asm.set_neighborlist(neighborlist)
    asm.set_gsaes_as_self_energies("wb97x-631gd")
    model = asm.assemble()
    if pretrained:
        model.load_state_dict(fetch_state_dict("aniala_state_dict.pt", private=True))
    return model


def ANIdr(
    pretrained: bool = True,
    neighborlist: str = "full_pairwise",
    cuda_ops: bool = False,
) -> BuiltinModel:
    asm = Assembler(ensemble_size=7)
    asm.symbols = ELEMENTS_2X
    asm.set_global_cutoff_fn("smooth2")
    asm.set_featurizer(
        AEVComputer,
        angular_terms=StandardAngular.like_2x(),
        radial_terms=StandardRadial.like_2x(),
        cuda_ops=cuda_ops,
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
    return model


def FlexibleANI(
    lot: str,  # functional-basis
    symbols: tp.Sequence[str],
    ensemble_size: int = 1,
    radial_cutoff: float = 5.2,
    angular_cutoff: float = 3.5,
    radial_shifts: int = 16,
    angular_shifts: int = 8,
    angle_sections: int = 4,
    cutoff_fn: tp.Union[Cutoff, str] = "smooth2",
    neighborlist: str = "full_pairwise",
    dispersion: bool = True,
    repulsion: bool = True,
    atomic_maker: tp.Callable[[str, int], torch.nn.Module] = atomics.like_dr,
    cuda_ops: bool = False,
) -> BuiltinModel:
    asm = Assembler(ensemble_size=ensemble_size)
    asm.symbols = tuple(symbols)
    asm.set_global_cutoff_fn(cutoff_fn)
    asm.set_featurizer(
        AEVComputer,
        radial_terms=StandardRadial.cover_linearly(
            start=0.9,
            cutoff=radial_cutoff,
            eta=19.7,
            num_shifts=radial_shifts,
        ),
        angular_terms=StandardAngular.cover_linearly(
            start=0.9,
            eta=12.5,
            zeta=14.1,
            num_shifts=angular_shifts,
            num_angle_sections=angle_sections,
            cutoff=angular_cutoff,
        ),
        cuda_ops=cuda_ops,
    )
    asm.set_atomic_maker(atomic_maker)
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
            cutoff=9.0,
            extra={"functional": lot.split("-")[0]},
        )
    return asm.assemble()


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

    model_dir = (Path.home() / ".local") / "torchani"

    if private:
        url = "http://moria.chem.ufl.edu/animodel/private/"
    else:
        url = "https://github.com/roitberg-group/torchani_model_zoo/releases/download/v0.1/"
    dict_ = torch.hub.load_state_dict_from_url(
        f"{url}/{state_dict_file}",
        model_dir=str(model_dir),
        map_location=torch.device("cpu"),
    )
    return OrderedDict(dict_)
