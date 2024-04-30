import struct
import bz2
import math
import collections.abc
import os
from collections import OrderedDict

import torch
import lark

from torchani.nn import ANIModel, Ensemble
from torchani.utils import EnergyShifter, ChemicalSymbolsToInts


class Constants(collections.abc.Mapping):
    """NeuroChem constants. Objects of this class can be used as arguments
    to :class:`torchani.AEVComputer`, like ``torchani.AEVComputer(**consts)``.

    Attributes:
        species_to_tensor (:class:`ChemicalSymbolsToInts`): call to convert
            string chemical symbols to 1d long tensor.
    """

    def __init__(self, filename):
        self.filename = filename
        with open(filename) as f:
            for i in f:
                try:
                    line = [x.strip() for x in i.split('=')]
                    name = line[0]
                    value = line[1]
                    if name == 'Rcr' or name == 'Rca':
                        setattr(self, name, float(value))
                    elif name in ['EtaR', 'ShfR', 'Zeta',
                                  'ShfZ', 'EtaA', 'ShfA']:
                        float_values = [float(x.strip()) for x in value.replace(
                            '[', '').replace(']', '').split(',')]
                        setattr(self, name, torch.tensor(float_values))
                    elif name == 'Atyp':
                        str_values = [x.strip() for x in value.replace(
                            '[', '').replace(']', '').split(',')]
                        self.species = str_values
                except Exception:
                    raise ValueError('unable to parse const file')
        self.num_species = len(self.species)
        self.species_to_tensor = ChemicalSymbolsToInts(self.species)

    def __iter__(self):
        yield 'Rcr'
        yield 'Rca'
        yield 'EtaR'
        yield 'ShfR'
        yield 'EtaA'
        yield 'Zeta'
        yield 'ShfA'
        yield 'ShfZ'
        yield 'num_species'

    def __len__(self):
        return 8

    def __getitem__(self, item):
        return getattr(self, item)


def load_sae(filename, return_dict=False):
    """Returns an object of :class:`EnergyShifter` with self energies from
    NeuroChem sae file"""
    _self_energies = []
    d = {}
    with open(filename) as f:
        for i in f:
            line = [x.strip() for x in i.split('=')]
            species = line[0].split(',')[0].strip()
            index = int(line[0].split(',')[1].strip())
            value = float(line[1])
            d[species] = value
            _self_energies.append((index, value))
    self_energies = [i for _, i in sorted(_self_energies)]
    if return_dict:
        return EnergyShifter(self_energies), d
    return EnergyShifter(self_energies)


def _get_activation(activation_index):
    # Activation defined in:
    # https://github.com/Jussmith01/NeuroChem/blob/stable1/src-atomicnnplib/cunetwork/cuannlayer_t.cu#L920
    if activation_index == 6:
        return None
    elif activation_index == 5:  # Gaussian
        raise NotImplementedError("Gaussian activations should not be used in NN models")
    elif activation_index == 9:  # CELU
        return torch.nn.CELU(alpha=0.1)
    else:
        raise NotImplementedError(
            'Unexpected activation {}'.format(activation_index))


def load_atomic_network(filename):
    """Returns an instance of :class:`torch.nn.Sequential` with hyperparameters
    and parameters loaded NeuroChem's .nnf, .wparam and .bparam files."""

    def decompress_nnf(buffer_):
        while buffer_[0] != b'='[0]:
            buffer_ = buffer_[1:]
        buffer_ = buffer_[2:]
        return bz2.decompress(buffer_)[:-1].decode('ascii').strip()

    def parse_nnf(nnf_file):
        # parse input file
        parser = lark.Lark(r'''
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
        ''', parser='lalr')
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
                        assert v.data == 'nans'
                        return math.nan
                    assert isinstance(v, lark.lexer.Token)
                    if v.type == 'FILENAME':
                        v = v.value
                    elif v.type == 'SIGNED_INT' or v.type == 'INT':
                        v = int(v.value)
                    elif v.type == 'SIGNED_FLOAT' or v.type == 'FLOAT':
                        v = float(v.value)
                    else:
                        raise ValueError('unexpected type')
                elif len(v) == 2:
                    v = self.value([v[0]]), self.value([v[1]])
                else:
                    raise ValueError('length of value can only be 1 or 2')
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

    def load_param_file(linear, in_size, out_size, wfn, bfn):
        """Load `.wparam` and `.bparam` files"""
        wsize = in_size * out_size
        fw = open(wfn, 'rb')
        _w = struct.unpack('{}f'.format(wsize), fw.read())
        w = torch.tensor(_w).view(out_size, in_size)
        linear.weight.data = w
        fw.close()
        fb = open(bfn, 'rb')
        _b = struct.unpack('{}f'.format(out_size), fb.read())
        b = torch.tensor(_b).view(out_size)
        linear.bias.data = b
        fb.close()

    networ_dir = os.path.dirname(filename)

    with open(filename, 'rb') as f:
        buffer_ = f.read()
        buffer_ = decompress_nnf(buffer_)
        layer_setups = parse_nnf(buffer_)

        layers = []
        for s in layer_setups:
            # construct linear layer and load parameters
            in_size = s['blocksize']
            out_size = s['nodes']
            wfn, wsz = s['weights']
            bfn, bsz = s['biases']
            if in_size * out_size != wsz or out_size != bsz:
                raise ValueError('bad parameter shape')
            layer = torch.nn.Linear(in_size, out_size)
            wfn = os.path.join(networ_dir, wfn)
            bfn = os.path.join(networ_dir, bfn)
            load_param_file(layer, in_size, out_size, wfn, bfn)
            layers.append(layer)
            activation = _get_activation(s['activation'])
            if activation is not None:
                layers.append(activation)

        return torch.nn.Sequential(*layers)


def load_model(species, dir_):
    """Returns an instance of :class:`torchani.ANIModel` loaded from
    NeuroChem's network directory.

    Arguments:
        species (:class:`collections.abc.Sequence`): Sequence of strings for
            chemical symbols of each supported atom type in correct order.
        dir_ (str): String for directory storing network configurations.
    """
    models = OrderedDict()
    for i in species:
        filename = os.path.join(dir_, 'ANN-{}.nnf'.format(i))
        models[i] = load_atomic_network(filename)
    return ANIModel(models)


def load_model_ensemble(species, prefix, count):
    """Returns an instance of :class:`torchani.Ensemble` loaded from
    NeuroChem's network directories beginning with the given prefix.

    Arguments:
        species (:class:`collections.abc.Sequence`): Sequence of strings for
            chemical symbols of each supported atom type in correct order.
        prefix (str): Prefix of paths of directory that networks configurations
            are stored.
        count (int): Number of models in the ensemble.
    """
    models = []
    for i in range(count):
        network_dir = os.path.join('{}{}'.format(prefix, i), 'networks')
        models.append(load_model(species, network_dir))
    return Ensemble(models)


__all__ = ['Constants', 'load_sae', 'load_model', 'load_model_ensemble']
