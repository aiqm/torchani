from pathlib import Path
import typing as tp
import struct
import bz2
import math
import os
from collections import OrderedDict

import torch
from torch import Tensor

try:
    import lark
except ImportError:
    raise ImportError(
        "Error when trying to import 'torchani.neurochem':"
        " The 'lark-parser' package could not be found. 'torchani.neurochem'"
        " won't be available. Please install 'lark-parser' if you want to use it."
        " ('conda install lark-parser' or 'pip install lark-parser')"
    ) from None

from torchani.aev import AEVComputer
from torchani.nn import ANIModel, Ensemble
from torchani.cutoffs import CutoffArg
from torchani.neighbors import NeighborlistArg
from torchani.potentials import EnergyAdder
from torchani.tuples import SpeciesEnergies
from torchani.neurochem.utils import model_dir_from_prefix


class NeurochemParseError(RuntimeError):
    pass


def load_aev_computer_and_symbols(
    consts_file: tp.Union[str, Path],
    use_cuda_extension: bool = False,
    use_cuaev_interface: bool = False,
    neighborlist: NeighborlistArg = "full_pairwise",
    cutoff_fn: CutoffArg = "cosine",
) -> tp.Tuple[AEVComputer, tp.Tuple[str, ...]]:
    aev_consts, aev_cutoffs, symbols = load_constants(consts_file)
    aev_computer = AEVComputer.from_constants(
        Rcr=aev_cutoffs["Rcr"],
        Rca=aev_cutoffs["Rca"],
        num_species=len(symbols),
        cutoff_fn=cutoff_fn,
        neighborlist=neighborlist,
        use_cuda_extension=use_cuda_extension,
        use_cuaev_interface=use_cuaev_interface,
        **aev_consts,
    )
    return aev_computer, symbols


def load_constants(
    consts_file: tp.Union[Path, str]
) -> tp.Tuple[tp.Dict[str, Tensor], tp.Dict[str, float], tp.Tuple[str, ...]]:
    aev_consts: tp.Dict[str, Tensor] = {}
    aev_cutoffs: tp.Dict[str, float] = {}
    with open(consts_file) as f:
        for i in f:
            try:
                line = [x.strip() for x in i.split("=")]
                name = line[0]
                value = line[1]
                if name in ["Rcr", "Rca"]:
                    aev_cutoffs[name] = float(value)
                elif name in ["EtaR", "ShfR", "Zeta", "ShfZ", "EtaA", "ShfA"]:
                    float_values = [
                        float(x.strip())
                        for x in value.replace("[", "").replace("]", "").split(",")
                    ]
                    aev_consts[name] = torch.tensor(float_values)
                elif name == "Atyp":
                    symbols = tuple(
                        x.strip()
                        for x in value.replace("[", "").replace("]", "").split(",")
                    )
            except Exception:
                raise NeurochemParseError(
                    f"Unable to parse const file {consts_file}"
                ) from None
    return aev_consts, aev_cutoffs, symbols


def load_energy_adder(filename: tp.Union[Path, str]) -> EnergyAdder:
    """Returns an object of :class:`EnergyAdder` with self energies from
    NeuroChem sae file"""
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
    return EnergyAdder(symbols, self_energies)


class EnergyShifter(torch.nn.Module):
    def __init__(self, adder: EnergyAdder) -> None:
        super().__init__()
        self._adder = adder

    def forward(
        self,
        species_energies: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> SpeciesEnergies:
        species, energies = species_energies
        self_energies = self._adder(species)
        return SpeciesEnergies(species, energies + self_energies)


# This function is kept for backwards compatibility
def load_sae(filename: tp.Union[Path, str]):
    """Returns an object of :class:`EnergyShifter` with self energies from
    NeuroChem sae file"""
    return EnergyShifter(load_energy_adder(filename))


def _get_activation(activation_index: int) -> tp.Optional[torch.nn.Module]:
    # Activation defined in:
    # https://github.com/Jussmith01/NeuroChem/blob
    #   /stable1/src-atomicnnplib/cunetwork/cuannlayer_t.cu#L920
    if activation_index == 6:
        return None
    elif activation_index == 9:  # CELU
        return torch.nn.CELU(alpha=0.1)
    elif activation_index == 5:  # Gaussian
        raise NeurochemParseError(
            "Activation index 5 corresponds to a Gaussian which is not supported"
        )
    raise NeurochemParseError(f"Unsupported activation index {activation_index}")


def load_atomic_network(filename: tp.Union[Path, str]) -> torch.nn.Sequential:
    """Returns an instance of :class:`torch.nn.Sequential` with hyperparameters
    and parameters loaded from NeuroChem's .nnf, .wparam and .bparam files."""
    filename = Path(filename).resolve()

    def decompress_nnf(buffer_):
        while buffer_[0] != b"="[0]:
            buffer_ = buffer_[1:]
        buffer_ = buffer_[2:]
        return bz2.decompress(buffer_)[:-1].decode("ascii").strip()

    def parse_nnf(nnf_file):
        # parse input file
        parser = lark.Lark(
            r"""
        identifier : CNAME

        inputsize : "inputsize" "=" INT ";"

        assign : identifier "=" value ";"

        layer : "layer" "[" assign * "]"

        atom_net : "atom_net" WORD "$" layer * "$"

        start: inputsize atom_net

        nans: "-"?"nan"

        value : SIGNED_INT
              | SIGNED_FLOAT
              | nans
              | "FILE" ":" FILENAME "[" INT "]"

        FILENAME : ("_"|"-"|"."|LETTER|DIGIT)+

        %import common.SIGNED_NUMBER
        %import common.LETTER
        %import common.WORD
        %import common.DIGIT
        %import common.INT
        %import common.SIGNED_INT
        %import common.SIGNED_FLOAT
        %import common.CNAME
        %import common.WS
        %ignore WS
        """,
            parser="lalr",
        )
        tree = parser.parse(nnf_file)

        # execute parse tree
        class TreeExec(lark.Transformer):
            def identifier(self, v):
                v = v[0].value
                return v

            def value(self, v):
                if len(v) == 1:
                    v = v[0]
                    if isinstance(v, lark.tree.Tree):
                        assert v.data == "nans"
                        return math.nan
                    assert isinstance(v, lark.lexer.Token)
                    if v.type == "FILENAME":
                        v = v.value
                    elif v.type == "SIGNED_INT" or v.type == "INT":
                        v = int(v.value)
                    elif v.type == "SIGNED_FLOAT" or v.type == "FLOAT":
                        v = float(v.value)
                    else:
                        raise NeurochemParseError(
                            f"Type should be one of"
                            " [SIGNED]_INT, [SIGNED]_FLOAT or FILENAME"
                            f" but found {v.type} in file {nnf_file}"
                        )
                elif len(v) == 2:
                    v = self.value([v[0]]), self.value([v[1]])
                else:
                    raise NeurochemParseError(
                        f"len(value) should be 1 or 2 but found {len(v)} in {nnf_file}"
                    )
                return v

            def assign(self, v):
                name = v[0]
                value = v[1]
                return name, value

            def layer(self, v):
                return dict(v)

            def atom_net(self, v):
                layers = v[1:]
                return layers

            def start(self, v):
                return v[1]

        layer_setups = TreeExec().transform(tree)
        return layer_setups

    def load_param_file(linear: torch.nn.Linear, in_size: int, out_size: int, wfn, bfn):
        """Load `.wparam` and `.bparam` files"""
        wsize = in_size * out_size
        with open(wfn, mode="rb") as fw:
            _w = struct.unpack("{}f".format(wsize), fw.read())
            w = torch.tensor(_w).view(out_size, in_size)
            linear.weight.data = w
        with open(bfn, mode="rb") as fb:
            _b = struct.unpack("{}f".format(out_size), fb.read())
            b = torch.tensor(_b).view(out_size)
            linear.bias.data = b

    with open(filename, "rb") as f:
        buffer_ = f.read()
        buffer_ = decompress_nnf(buffer_)
        layer_setups = parse_nnf(buffer_)

        layers: tp.List[torch.nn.Module] = []
        network_dir = str(filename.parent)
        for s in layer_setups:
            # construct linear layer and load parameters
            in_size = s["blocksize"]
            out_size = s["nodes"]
            wfn, wsz = s["weights"]
            bfn, bsz = s["biases"]
            if in_size * out_size != wsz:
                raise NeurochemParseError(
                    f"Bad parameter shape in {filename}:"
                    f" in_size * out_size=({in_size} * {out_size})"
                    f" should be equal to wsz={wsz}"
                )
            if out_size != bsz:
                raise NeurochemParseError(
                    f"Bad parameter shape in {filename}:"
                    f" out_size={out_size}"
                    f" should be equal to bsz={bsz}"
                )
            layer = torch.nn.Linear(in_size, out_size)
            wfn = os.path.join(network_dir, wfn)
            bfn = os.path.join(network_dir, bfn)
            load_param_file(layer, in_size, out_size, wfn, bfn)
            layers.append(layer)
            activation = _get_activation(s["activation"])
            if activation is not None:
                layers.append(activation)

        return torch.nn.Sequential(*layers)


def load_model(symbols: tp.Sequence[str], model_dir: tp.Union[Path, str]) -> ANIModel:
    """Returns an instance of :class:`torchani.nn.ANIModel` loaded from
    NeuroChem's network directory.

    Arguments:
        symbols (:class:`collections.abc.Sequence`): Sequence of strings for
            chemical symbols of each supported atom type in correct order.
        model_dir (str): String for directory storing network configurations.
    """
    model_dir = Path(model_dir).resolve()
    return ANIModel(
        OrderedDict(
            [(s, load_atomic_network(model_dir / f"ANN-{s}.nnf")) for s in symbols]
        )
    )


def load_model_ensemble(
    symbols: tp.Sequence[str], prefix: tp.Union[Path, str], count: int
) -> Ensemble:
    """Returns an instance of :class:`torchani.nn.Ensemble` loaded from
    NeuroChem's network directories beginning with the given prefix.

    Arguments:
        symbols (:class:`collections.abc.Sequence`): Sequence of strings for
            chemical symbols of each supported atom type in correct order.
        prefix (str): Prefix of paths of directory that networks configurations
            are stored.
        count (int): Number of models in the ensemble.
    """
    prefix = Path(prefix)
    return Ensemble(
        [load_model(symbols, model_dir_from_prefix(prefix, i)) for i in range(count)]
    )


__all__ = [
    "load_constants",
    "load_aev_computer_and_symbols",
    "load_sae",
    "load_energy_adder",
    "load_model",
    "load_model_ensemble",
]
