from .aev_base import AEVComputer
import torch
import torch.nn as nn
import bz2
import os
import lark
import struct
import copy
import math


class NeuralNetworkOnAEV(nn.Module):

    def __init__(self, aev_computer, **kwargs):
        super(NeuralNetworkOnAEV, self).__init__()
        if not isinstance(aev_computer, AEVComputer):
            raise TypeError(
                "NeuralNetworkPotential: aev_computer must be a subclass of AEVComputer")
        self.aev_computer = aev_computer
        self.aev_length = aev_computer.radial_length() + aev_computer.angular_length()

        if 'from_pync' in kwargs:
            self._from_pync(kwargs['from_pync'])
        elif 'sizes' in kwargs:
            sizes = kwargs['sizes']
            if isinstance(sizes, list):
                sz = sizes
                sizes = {}
                for i in aev_computer.species:
                    sizes[i] = copy.copy(sz)
            activation = kwargs['activation'] if 'activation' in kwargs else lambda x: torch.exp(
                -x**2)
            reducer = kwargs['reducer'] if 'reducer' in kwargs else torch.sum
            self._from_config(sizes, activation, reducer)
        else:
            raise ValueError(
                'bad arguments when initializing NeuralNetworkOnAEV')

    def _load_params_from_param_file(self, linear, in_size, out_size, wfn, bfn):
        wsize = in_size * out_size
        fw = open(wfn, 'rb')
        float_w = struct.unpack('{}f'.format(wsize), fw.read())
        linear.weight = torch.nn.parameter.Parameter(torch.FloatTensor(
            float_w).type(self.aev_computer.dtype).view(out_size, in_size))
        fw.close()
        fb = open(bfn, 'rb')
        float_b = struct.unpack('{}f'.format(out_size), fb.read())
        linear.bias = torch.nn.parameter.Parameter(torch.FloatTensor(
            float_b).type(self.aev_computer.dtype).view(out_size))
        fb.close()

    def _construct_layers_from_neurochem_cfgfile(self, species, setups, dirname):
        # activation defined in file https://github.com/Jussmith01/NeuroChem/blob/master/src-atomicnnplib/cunetwork/cuannlayer_t.cu#L868
        self.activation_index = None
        self.activation = None
        self.layers[species] = len(setups)
        for i in range(self.layers[species]):
            s = setups[i]
            in_size = s['blocksize']
            out_size = s['nodes']
            activation = s['activation']
            wfn, wsz = s['weights']
            bfn, bsz = s['biases']
            if i == self.layers[species]-1:
                if activation != 6:  # no activation
                    raise ValueError('activation in the last layer must be 6')
            else:
                if self.activation_index is None:
                    if activation != 5:
                        raise NotImplementedError('only gaussian is supported')
                    else:
                        self.activation_index = activation
                        self.activation = lambda x: torch.exp(-x**2)
                elif self.activation_index != activation:
                    raise NotImplementedError(
                        'different activation on different layers are not supported')
            linear = nn.Linear(in_size, out_size).type(self.aev_computer.dtype)
            name = '{}{}'.format(species, i)
            setattr(self, name, linear)
            if in_size * out_size != wsz or out_size != bsz:
                raise ValueError('bad parameter shape')
            wfn = os.path.join(dirname, wfn)
            bfn = os.path.join(dirname, bfn)
            self._load_params_from_param_file(
                linear, in_size, out_size, wfn, bfn)

    def _read_nnf_file(self, species, filename):
        # decompress nnf file
        f = open(filename, 'rb')
        d = f.read()
        f.close()
        while d[0] != b'='[0]:
            d = d[1:]
        d = d[2:]
        d = bz2.decompress(d)[:-1].decode('ascii').strip()

        # parse input size
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
        tree = parser.parse(d)

        # execute parse tree
        class TreeExec(lark.Transformer):
            def __init__(self, outerself, species):
                self.outerself = outerself
                self.species = species

            def inputsize(self, v):
                v = int(v[0])
                if self.outerself.aev_length != v:
                    raise ValueError('aev size of network file ({}) mismatch aev computer ({})'.format(
                        v, self.outerself.aev_length))

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
                s = v[0].value
                layers = v[1:]
                if self.species != s:
                    raise ValueError(
                        'network file does not store expected species')
                return layers

            def start(self, v):
                return v[1]

        layer_setups = TreeExec(self, species).transform(tree)
        self._construct_layers_from_neurochem_cfgfile(
            species, layer_setups, os.path.dirname(filename))

    def _from_pync(self, network_dir):
        self.reducer = torch.sum
        self.layers = {}
        for i in self.aev_computer.species:
            filename = os.path.join(network_dir, 'ANN-{}.nnf'.format(i))
            self._read_nnf_file(i, filename)

    def _from_config(self, sizes, activation, reducer):
        self.activation = activation
        self.reducer = reducer
        for i in self.aev_computer.species:
            sizes[i] = [self.aev_length] + sizes[i]
            self.layers[i] = len(sizes[i])
            for j in range(self.layers[i]):
                linear = nn.Linear(sizes[j], sizes[j+1]
                                   ).type(self.aev_computer.dtype)
                setattr(self, '{}{}'.format(i, j), linear)

    def forward(self, coordinates, species):
        per_atom_outputs = self.get_activations(coordinates, species, math.inf)
        per_atom_outputs = torch.stack(per_atom_outputs)
        molecule_output = self.reducer(per_atom_outputs, dim=0)
        return torch.squeeze(molecule_output)

    def get_activations(self, coordinates, species, layer):
        radial_aev, angular_aev = self.aev_computer(coordinates, species)
        fullaev = torch.cat([radial_aev, angular_aev], dim=2)
        atoms = len(species)
        per_atom_outputs = []
        for i in range(atoms):
            s = species[i]
            y = fullaev[:, i, :]
            for j in range(self.layers[s]-1):
                linear = getattr(self, '{}{}'.format(s, j))
                y = linear(y)
                y = self.activation(y)
                if j == layer:
                    break
            if layer >= self.layers[s]-1:
                linear = getattr(self, '{}{}'.format(s, self.layers[s]-1))
                y = linear(y)
            per_atom_outputs.append(y)
        return per_atom_outputs

    def reset_parameters(self):
        for s in self.aev_computer.species:
            for j in range(self.layers[s]):
                getattr(self, '{}{}'.format(s, j)).reset_parameters()
