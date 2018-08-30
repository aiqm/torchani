# -*- coding: utf-8 -*-
"""Tools for loading NeuroChem input files."""

import pkg_resources
import torch
import os
import bz2
import lark
import struct
from collections.abc import Mapping
from ..nn import ANIModel, Ensemble, Gaussian
from ..utils import EnergyShifter
from ..aev import AEVComputer


class Constants(Mapping):
    """NeuroChem constants. Objects of this class can be used as arguments
    to :class:`torchani.AEVComputer`, like ``torchani.AEVComputer(**consts)``.
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
                        setattr(self, name, torch.tensor(float(value)))
                    elif name in ['EtaR', 'ShfR', 'Zeta',
                                  'ShfZ', 'EtaA', 'ShfA']:
                        value = [float(x.strip()) for x in value.replace(
                            '[', '').replace(']', '').split(',')]
                        setattr(self, name, torch.tensor(value))
                    elif name == 'Atyp':
                        value = [x.strip() for x in value.replace(
                            '[', '').replace(']', '').split(',')]
                        self.species = value
                except Exception:
                    raise ValueError('unable to parse const file')
        self.num_species = len(self.species)
        self.rev_species = {}
        for i in range(len(self.species)):
            s = self.species[i]
            self.rev_species[s] = i

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

    def species_to_tensor(self, species):
        """Convert species from squence of strings to 1D tensor"""
        rev = [self.rev_species[s] for s in species]
        return torch.tensor(rev, dtype=torch.long)


def load_sae(filename):
    """Returns an object of :class:`EnergyShifter` with self energies from
    NeuroChem sae file"""
    self_energies = []
    with open(filename) as f:
        for i in f:
            line = [x.strip() for x in i.split('=')]
            index = int(line[0].split(',')[1].strip())
            value = float(line[1])
            self_energies.append((index, value))
    self_energies = [i for _, i in sorted(self_energies)]
    return EnergyShifter(self_energies)


def _get_activation(activation_index):
    # Activation defined in:
    # https://github.com/Jussmith01/NeuroChem/blob/master/src-atomicnnplib/cunetwork/cuannlayer_t.cu#L868
    if activation_index == 6:
        return None
    elif activation_index == 5:  # Gaussian
        return Gaussian()
    elif activation_index == 9:  # CELU
        return torch.nn.CELU(alpha=0.1)
    else:
        raise NotImplementedError(
            'Unexpected activation {}'.format(activation_index))


def load_atomic_network(filename):
    """Returns an instance of :class:`torch.nn.Sequential` with hyperparameters
    and parameters loaded NeuroChem's .nnf, .wparam and .bparam files."""

    def decompress_nnf(buffer):
        while buffer[0] != b'='[0]:
            buffer = buffer[1:]
        buffer = buffer[2:]
        return bz2.decompress(buffer)[:-1].decode('ascii').strip()

    def parse_nnf(nnf_file):
        # parse input file
        parser = lark.Lark(r'''
        identifier : CNAME

        inputsize : "inputsize" "=" INT ";"

        assign : identifier "=" value ";"

        layer : "layer" "[" assign * "]"

        atom_net : "atom_net" WORD "$" layer * "$"

        start: inputsize atom_net

        value : INT
              | FLOAT
              | "FILE" ":" FILENAME "[" INT "]"

        FILENAME : ("_"|"-"|"."|LETTER|DIGIT)+

        %import common.SIGNED_NUMBER
        %import common.LETTER
        %import common.WORD
        %import common.DIGIT
        %import common.INT
        %import common.FLOAT
        %import common.CNAME
        %import common.WS
        %ignore WS
        ''')
        tree = parser.parse(nnf_file)

        # execute parse tree
        class TreeExec(lark.Transformer):

            def identifier(self, v):
                v = v[0].value
                return v

            def value(self, v):
                if len(v) == 1:
                    v = v[0]
                    if v.type == 'FILENAME':
                        v = v.value
                    elif v.type == 'INT':
                        v = int(v.value)
                    elif v.type == 'FLOAT':
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
        w = struct.unpack('{}f'.format(wsize), fw.read())
        w = torch.tensor(w).view(out_size, in_size)
        linear.weight.data = w
        fw.close()
        fb = open(bfn, 'rb')
        b = struct.unpack('{}f'.format(out_size), fb.read())
        b = torch.tensor(b).view(out_size)
        linear.bias.data = b
        fb.close()

    networ_dir = os.path.dirname(filename)

    with open(filename, 'rb') as f:
        buffer = f.read()
        buffer = decompress_nnf(buffer)
        layer_setups = parse_nnf(buffer)

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


def load_model(species, dir):
    """Returns an instance of :class:`torchani.ANIModel` loaded from
    NeuroChem's network directory.

    Arguments:
        species (:class:`collections.abc.Sequence`): Sequence of strings for
            chemical symbols of each supported atom type in correct order.
        dir (str): String for directory storing network configurations.
    """
    models = []
    for i in species:
        filename = os.path.join(dir, 'ANN-{}.nnf'.format(i))
        models.append(load_atomic_network(filename))
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


class Buildins:
    """Container for all builtin stuffs.

    Attributes:
        const_file (:class:`str`): Path to the builtin constant file.
        consts (:class:`Constants`): Constants loaded from builtin constant
            file.
        aev_computer (:class:`torchani.AEVComputer`): AEV computer with builtin
            constants.
        sae_file (:class:`str`): Path to the builtin self atomic energy file.
        energy_shifter (:class:`torchani.EnergyShifter`): AEV computer with
            builtin constants.
        ensemble_size (:class:`int`): Number of models in model ensemble.
        ensemble_prefix (:class:`str`): Prefix of directories of models.
        models (:class:`torchani.Ensemble`): Ensemble of models.
    """

    def __init__(self):
        parent_name = '.'.join(__name__.split('.')[:-1])
        self.const_file = pkg_resources.resource_filename(
            parent_name,
            'resources/ani-1x_dft_x8ens/rHCNO-5.2R_16-3.5A_a4-8.params')
        self.consts = Constants(self.const_file)
        self.aev_computer = AEVComputer(**self.consts)

        self.sae_file = pkg_resources.resource_filename(
            parent_name, 'resources/ani-1x_dft_x8ens/sae_linfit.dat')
        self.energy_shifter = load_sae(self.sae_file)

        self.ensemble_size = 8
        self.ensemble_prefix = pkg_resources.resource_filename(
            parent_name, 'resources/ani-1x_dft_x8ens/train')
        self.models = load_model_ensemble(self.consts.species,
                                          self.ensemble_prefix,
                                          self.ensemble_size)


from .trainer import Trainer  # noqa: F401
