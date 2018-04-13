from .aev_base import AEVComputer
import torch
import torch.nn as nn
import bz2
import os
import lark

class NeuralNetworkOnAEV(nn.Module):

    def __init__(self, aev_computer, **kwargs):
        super(NeuralNetworkOnAEV, self).__init__()
        if not isinstance(aev_computer, AEVComputer):
            raise TypeError("NeuralNetworkPotential: aev_computer must be a subclass of AEVComputer")
        self.aev_computer = aev_computer
        self.aev_length = aev_computer.radial_length() + aev_computer.angular_length()

        if 'from_pync' in kwargs:
            self._from_pync(kwargs['from_pync'])
        elif 'sizes' in kwargs:
            sizes = kwargs['sizes']
            activation = kwargs['activation']
            reducer = kwargs['reducer'] if 'reducer' in kwargs else torch.sum
            self._from_config(sizes, activation, reducer)
        else:
            raise ValueError('bad arguments when initializing NeuralNetworkOnAEV')

    def _construct_layers_from_setups(self, species, setups):
        print(setups)
        #TODO

    def _read_nnf_file(self, species, filename):

        # decompress nnf file
        f = open(filename, 'rb').read()
        while f[0] != b'='[0]:
            f = f[1:]
        f = f[2:]
        d = bz2.decompress(f)[:-1].decode('ascii').strip()

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
                    raise ValueError('aev size of network file ({}) mismatch aev computer ({})'.format(v, self.outerself.aev_length))

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
                return name,value

            def layer(self, v):
                return dict(v)

            def atom_net(self, v):
                s = v[0].value
                layers = v[1:]
                if self.species != s:
                    raise ValueError('network file does not store expected species')
                return layers

            def start(self, v):
                return v[1]

        layer_setups = TreeExec(self, species).transform(tree)
        self._construct_layers_from_setups(species, layer_setups)

    def _from_pync(self, network_dir):
        for i in self.aev_computer.species:
            filename = os.path.join(network_dir, 'ANN-{}.nnf'.format(i))
            self._read_nnf_file(i, filename)

    def _from_config(self, sizes, activation, reducer):
        self.layers = len(sizes)
        self.activation = activation
        self.reducer = reducer
        sizes = [self.aev_length] + sizes
        for i in self.aev_computer.species:
            for j in range(self.layers):
                linear = nn.Linear(sizes[j], sizes[j+1]).type(self.aev_computer.dtype)
                setattr(self, '{}{}'.format(i,j), linear)

    def forward(self, coordinates, species):
        radial_aev, angular_aev = self.aev_computer(coordinates, species)
        fullaev = torch.cat([radial_aev, angular_aev], dim=2)
        atoms = len(species)
        per_atom_outputs = []
        for i in range(atoms):
            s = species[i]
            y = fullaev[:,i,:]
            for j in range(self.layers-1):
                linear = getattr(self, '{}{}'.format(s,j))
                y = linear(y)
                y = self.activation(y)
            linear = getattr(self, '{}{}'.format(s,self.layers-1))
            y = linear(y)
            per_atom_outputs.append(y)
        per_atom_outputs = torch.stack(per_atom_outputs)
        molecule_output = self.reducer(per_atom_outputs, dim=0)
        return torch.squeeze(molecule_output)

    def reset_parameters(self):
        for s in self.aev_computer.species:
            for j in range(self.layers):
                getattr(self, '{}{}'.format(s,j)).reset_parameters()