r"""
This module is part of the *Legacy API* of TorchANI 2 and should not be used in new
code. It should only be used if you need to interface with previously trained ANI models
in the NeuroChem format. It contains tools for loading files in the NeuroChem format,
the original file format used in the first `ANI`_ article.

.. _ANI:
    http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
"""

import itertools
from pathlib import Path
from dataclasses import dataclass
import typing as tp
import struct
import bz2
import shutil
from collections import OrderedDict

import torch
import typing_extensions as tpx

from torchani.paths import neurochem_dir
from torchani.arch import ANI
from torchani.aev import AEVComputer
from torchani.nn import (
    ANINetworks,
    Ensemble,
    AtomicNetwork,
    AtomicContainer,
    TightCELU,
)
from torchani.cutoffs import CutoffArg
from torchani.neighbors import NeighborlistArg
from torchani.sae import SelfEnergy
from torchani.utils import download_and_extract
from torchani.annotations import StrPath


def model_dir_from_prefix(prefix: Path, idx: int) -> Path:
    network_path = (prefix.parent / f"{prefix.name}{idx}") / "networks"
    return network_path


class NeurochemParseError(RuntimeError):
    pass


@dataclass
class NeurochemLayerSpec:
    nodes: int
    activation: int
    kind: int
    blocksize: int
    dropout: int
    dropset: float
    maskupdate: float
    maxnorm: int
    norm: float
    normupdate: float
    l2norm: int
    l2value: float
    batchnorm: int
    weights: str
    weight_numel: int
    biases: str
    bias_numel: int


def load_aev_computer_and_symbols(
    consts_file: StrPath,
    strategy: str = "pyaev",
    neighborlist: NeighborlistArg = "all_pairs",
    cutoff_fn: CutoffArg = "cosine",
) -> tp.Tuple[AEVComputer, tp.Tuple[str, ...]]:
    consts, symbols = load_aev_constants_and_symbols(consts_file)
    aev_computer = AEVComputer.from_constants(
        radial_cutoff=consts.radial_cutoff,
        angular_cutoff=consts.angular_cutoff,
        radial_eta=consts.radial_eta,
        radial_shifts=consts.radial_shifts,
        angular_eta=consts.angular_eta,
        angular_zeta=consts.angular_zeta,
        angular_shifts=consts.angular_shifts,
        sections=consts.sections,
        num_species=len(symbols),
        strategy=strategy,
        cutoff_fn=cutoff_fn,
        neighborlist=neighborlist,
    )
    return aev_computer, symbols


@dataclass
class AEVConstants:
    radial_cutoff: float
    radial_eta: float
    radial_shifts: tp.Tuple[float, ...]
    angular_cutoff: float
    angular_eta: float
    angular_zeta: float
    angular_shifts: tp.Tuple[float, ...]
    sections: tp.Tuple[float, ...]


def load_aev_constants_and_symbols(
    consts_file: StrPath,
) -> tp.Tuple[AEVConstants, tp.Tuple[str, ...]]:
    aev_floats: tp.Dict[str, float] = {}
    aev_seqs: tp.Dict[str, tp.Tuple[float, ...]] = {}
    file_name_mapping = {
        "Rcr": "radial_cutoff",
        "Rca": "angular_cutoff",
        "EtaR": "radial_eta",
        "ShfR": "radial_shifts",
        "ShfA": "angular_shifts",
        "ShfZ": "sections",
        "EtaA": "angular_eta",
        "Zeta": "angular_zeta",
    }
    with open(consts_file) as f:
        for i in f:
            try:
                line = [x.strip() for x in i.split("=")]
                name = line[0]
                value = line[1]
                if name in ["Rcr", "Rca"]:
                    aev_floats[file_name_mapping[name]] = float(value)
                elif name in ["EtaR", "ShfR", "Zeta", "ShfZ", "EtaA", "ShfA"]:
                    float_values = tuple(
                        float(x.strip())
                        for x in value.replace("[", "").replace("]", "").split(",")
                    )
                    if name in ["EtaR", "Zeta", "EtaA"]:
                        assert len(float_values) == 1
                        aev_floats[file_name_mapping[name]] = float_values[0]
                    else:
                        aev_seqs[file_name_mapping[name]] = float_values
                elif name == "Atyp":
                    symbols = tuple(
                        x.strip()
                        for x in value.replace("[", "").replace("]", "").split(",")
                    )
            except Exception:
                raise NeurochemParseError(
                    f"Unable to parse const file {consts_file}"
                ) from None
    constants = AEVConstants(
        radial_cutoff=aev_floats["radial_cutoff"],
        angular_cutoff=aev_floats["angular_cutoff"],
        radial_eta=aev_floats["radial_eta"],
        angular_eta=aev_floats["angular_eta"],
        angular_zeta=aev_floats["angular_zeta"],
        radial_shifts=aev_seqs["radial_shifts"],
        angular_shifts=aev_seqs["angular_shifts"],
        sections=aev_seqs["sections"],
    )
    return constants, symbols


def load_sae(filename: StrPath) -> SelfEnergy:
    r"""Returns a self-energy calculator, with self energies from NeuroChem sae file

    The constructed object is of class `torchani.sae.SelfEnergy`, and can
    be used to calculate the self atomic energies of a group of molecules.
    """
    _self_energies = []
    _symbols = []
    with open(Path(filename).resolve(), mode="rt", encoding="utf-8") as f:
        for i in f:
            line = [x.strip() for x in i.split("=")]
            symbol = line[0].split(",")[0].strip()
            idx = int(line[0].split(",")[1].strip())
            energy = float(line[1])
            _symbols.append((idx, symbol))
            _self_energies.append((idx, energy))
    self_energies = [e for _, e in sorted(_self_energies)]
    symbols = [s for _, s in sorted(_symbols)]
    return SelfEnergy(symbols, self_energies)


def _get_activation(activation_index: int) -> torch.nn.Module:
    # Activation defined in:
    # https://github.com/Jussmith01/NeuroChem/blob
    #   /stable1/src-atomicnnplib/cunetwork/cuannlayer_t.cu#L920
    if activation_index == 9:  # CELU
        return TightCELU()
    elif activation_index == 5:  # Gaussian
        raise NeurochemParseError(
            "Activation index 5 corresponds to a Gaussian which is not supported"
        )
    raise NeurochemParseError(f"Unsupported activation index {activation_index}")


def _decompress_nnf(buffer_: bytes) -> str:
    while buffer_[0] != ord("="):
        buffer_ = buffer_[1:]
    buffer_ = buffer_[2:]
    return bz2.decompress(buffer_)[:-1].decode("ascii").strip()


def _parse_nnf(nnf_str: str) -> tp.List[NeurochemLayerSpec]:
    # Hack: replace tokens so the file can be evale'd as a list of python dicts
    # This is unsafe and hacky but since neurochem is legacy it is not a problem
    layers = [
        layer.strip()
        for layer in nnf_str.replace("\n", "")
        .replace("$", "")
        .replace("FILE:", "'")
        .replace("];]", ")")
        .replace("]", "")
        .replace(";", ",")
        .replace("wparam[", "wparam',weight_numel=")
        .replace("bparam[", "bparam',bias_numel=")
        .replace("type", "kind")
        .replace("l2valu", "l2value")
        .replace("btchnorm", "batchnorm")
        .replace("[", "dict(")
        .replace("=-nan", "=float('nan')")
        .replace("=nan", "=float('nan')")
        .split("layer")[1:]
    ]
    return [NeurochemLayerSpec(**eval(layer)) for layer in layers]


def load_atomic_network(filename: StrPath) -> AtomicNetwork:
    """Returns a `torchani.nn.AtomicNetwork`

    Hyperparams and params are loaded from NeuroChem's ``.nnf``, ``.wparam`` and
    ``.bparam`` files.
    """
    filename = Path(filename).resolve()

    with open(filename, "rb") as f:
        nnf_compressed_buffer = f.read()

    try:
        nnf_str = _decompress_nnf(nnf_compressed_buffer)
    except Exception:
        raise NeurochemParseError(f"Could not decompress nnf file {filename}") from None

    try:
        layer_specs = _parse_nnf(nnf_str)
    except Exception:
        raise NeurochemParseError(f"Could not parse nnf file {filename}") from None

    activations: tp.List[int] = []
    layer_dims: tp.List[int] = []
    weight_files: tp.List[Path] = []
    bias_files: tp.List[Path] = []
    for j, spec in enumerate(layer_specs):
        # construct linear layer and load parameters
        _in = spec.blocksize
        _out = spec.nodes
        weight_filename = spec.weights
        weights_numel = spec.weight_numel
        bias_filename = spec.biases
        biases_numel = spec.bias_numel
        if _in * _out != weights_numel:
            raise NeurochemParseError(
                f"Bad parameter shape in {filename}:"
                f" blocksize * nodes=({_in} * {_out})"
                f" should be equal to weights_numel={weights_numel}"
            )
        if _out != biases_numel:
            raise NeurochemParseError(
                f"Bad parameter shape in {filename}:"
                f" nodes={_out}"
                f" should be equal to biases_numel={biases_numel}"
            )
        if j == 0:
            layer_dims.extend([_in, _out])
        else:
            if layer_dims[-1] != _in:
                raise NeurochemParseError(f"Bad layer dimension in {filename}")
            layer_dims.append(_out)
        weight_files.append(filename.parent / weight_filename)
        bias_files.append(filename.parent / bias_filename)
        activations.append(spec.activation)

    assert activations[-1] == 6, "Last activation must have index 6"
    assert len(set(activations[:-1])) == 1, "All activations must be equal"

    network = AtomicNetwork(
        layer_dims,
        activation=_get_activation(activations[0]),
        bias=True,
    )

    for linear, wfile, bfile in zip(
        itertools.chain(network.layers, [network.final_layer]), weight_files, bias_files
    ):
        _in = linear.in_features
        _out = linear.out_features
        with open(wfile, mode="rb") as wf:
            _w = struct.unpack("{}f".format(_in * _out), wf.read())
            linear.weight.data = torch.tensor(_w).view(_out, _in)
        with open(bfile, mode="rb") as bf:
            _b = struct.unpack("{}f".format(_out), bf.read())
            linear.bias.data = torch.tensor(_b).view(_out)

    return network


def load_member(symbols: tp.Sequence[str], model_dir: StrPath) -> ANINetworks:
    """Returns a `torchani.nn.ANINetworks` loaded from NeuroChem's network directory.

    Args:
        symbols: |symbols|
        model_dir: Dir storing network configurations.
    """
    model_dir = Path(model_dir).resolve()
    return ANINetworks(
        OrderedDict(
            [(s, load_atomic_network(model_dir / f"ANN-{s}.nnf")) for s in symbols]
        )
    )


def load_ensemble(
    symbols: tp.Sequence[str], prefix: StrPath, count: int
) -> Ensemble:
    r"""Loads `torchani.nn.Ensemble` from NeuroChem's dirs with a given prefix

    Args:
        symbols: |symbols|
        prefix: Prefix of paths of directory where networks configurations are stored.
        count: Number of models in the ensemble.
    """
    prefix = Path(prefix)
    return Ensemble(
        [load_member(symbols, model_dir_from_prefix(prefix, i)) for i in range(count)]
    )


SUPPORTED_MODELS = {"ani1x", "ani2x", "ani1ccx"}


@dataclass
class NeurochemInfo:
    sae: Path
    const: Path
    ensemble_prefix: Path
    ensemble_size: int

    @classmethod
    def from_info_file(cls, info_file_path: Path) -> tpx.Self:
        with open(info_file_path, mode="rt", encoding="utf-8") as f:
            lines: tp.List[str] = [x.strip() for x in f.readlines()][:4]
            _const_file, _sae_file, _ensemble_prefix, _ensemble_size = lines

            ensemble_size: int = int(_ensemble_size)
            const_file_path: Path = Path(neurochem_dir(), *_const_file.split("/"))
            sae_file_path: Path = Path(neurochem_dir(), *_sae_file.split("/"))
            ensemble_prefix: Path = Path(neurochem_dir(), *_ensemble_prefix.split("/"))
        return cls(sae_file_path, const_file_path, ensemble_prefix, ensemble_size)

    @classmethod
    def from_model_name(cls, model_name: str) -> tpx.Self:
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Neurochem model {model_name} not supported,"
                f" supported models are: {SUPPORTED_MODELS}",
            )
        suffix = model_name.replace("ani", "")
        info_file_path = Path(neurochem_dir(), f"ani-{suffix}_8x.info")
        if not info_file_path.is_file():
            download_model_parameters()
        info = cls.from_info_file(info_file_path)
        return info


def download_model_parameters(
    root: tp.Optional[Path] = None, verbose: bool = True
) -> None:
    root = root or neurochem_dir()
    if any(root.iterdir()):
        if verbose:
            print("Found existing files in directory, assuming params already present")
        return
    repo = "ani-model-zoo"
    tag = "ani-2x"
    extracted_dirname = f"{repo}-{tag}"
    url = f"https://github.com/aiqm/{repo}/archive/{tag}.zip"
    download_and_extract(url, "neurochem-builtins.zip", root, verbose=verbose)
    extracted_dir = Path(root) / extracted_dirname
    for f in (extracted_dir / "resources").iterdir():
        shutil.move(str(f), root / f.name)
    shutil.rmtree(extracted_dir)


def modules_from_info(
    info: NeurochemInfo,
    model_index: tp.Optional[int] = None,
    strategy: str = "pyaev",
) -> tp.Tuple[AEVComputer, AtomicContainer, SelfEnergy, tp.Sequence[str]]:
    aev_computer, symbols = load_aev_computer_and_symbols(
        info.const,
        strategy=strategy,
    )
    adder = load_sae(info.sae)

    neural_networks: AtomicContainer
    if model_index is None:
        neural_networks = load_ensemble(
            symbols, info.ensemble_prefix, info.ensemble_size
        )
    else:
        if model_index >= info.ensemble_size:
            raise ValueError(
                f"Model index {model_index} should be <= {info.ensemble_size}"
            )
        neural_networks = load_member(
            symbols,
            model_dir_from_prefix(
                info.ensemble_prefix,
                model_index,
            ),
        )
    return aev_computer, neural_networks, adder, symbols


def modules_from_model_name(
    model_name: str,
    model_index: tp.Optional[int] = None,
    strategy: str = "pyaev",
) -> tp.Tuple[AEVComputer, AtomicContainer, SelfEnergy, tp.Sequence[str]]:
    r"""Creates the necessary modules to generate a pre-trained ANI model

    Parses data from legacy neurochem files, which are fetched according to the model
    name.
    """
    return modules_from_info(
        NeurochemInfo.from_model_name(model_name),
        model_index,
        strategy=strategy,
    )


def modules_from_info_file(
    info_file: Path,
    model_index: tp.Optional[int] = None,
    strategy: str = "pyaev",
) -> tp.Tuple[AEVComputer, AtomicContainer, SelfEnergy, tp.Sequence[str]]:
    r"""
    Creates the necessary modules to generate a pre-trained ANI model, parsing the data
    from legacy neurochem files.
    """
    return modules_from_info(
        NeurochemInfo.from_info_file(info_file),
        model_index,
        strategy=strategy,
    )


def load_model_from_info_file(
    info_file: StrPath,
    model_index: tp.Optional[int] = None,
    strategy: str = "pyaev",
    periodic_table_index: bool = True,
) -> ANI:
    info_file = Path(info_file).resolve()
    components = modules_from_info_file(
        info_file,
        model_index,
        strategy=strategy,
    )
    aev_computer, neural_networks, self_energy_potential, symbols = components
    return ANI(
        symbols,
        aev_computer,
        neural_networks,
        self_energy_potential,
        periodic_table_index=periodic_table_index,
    )


def load_model_from_name(
    model_name: str,
    model_index: tp.Optional[int] = None,
    strategy: str = "pyaev",
    periodic_table_index: bool = True,
) -> ANI:
    components = modules_from_model_name(
        model_name,
        model_index,
        strategy=strategy,
    )
    aev_computer, neural_networks, self_energy_potential, symbols = components
    return ANI(
        symbols,
        aev_computer,
        neural_networks,
        self_energy_potential,
        periodic_table_index=periodic_table_index,
    )


__all__ = [
    "load_aev_constants_and_symbols",
    "load_aev_computer_and_symbols",
    "load_sae",
    "load_member",
    "load_ensemble",
    "load_model_from_name",
    "load_model_from_info_file",
    "modules_from_model_name",
    "modules_from_info_file",
    "download_model_parameters",
]
