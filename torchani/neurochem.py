import pkg_resources
import torch
import os
import bz2
import lark
import struct
from collections.abc import Mapping
from .models import ANIModel, Ensemble


buildin_const_file = pkg_resources.resource_filename(
    __name__, 'resources/ani-1x_dft_x8ens/rHCNO-5.2R_16-3.5A_a4-8.params')

buildin_sae_file = pkg_resources.resource_filename(
    __name__, 'resources/ani-1x_dft_x8ens/sae_linfit.dat')

buildin_network_dir = pkg_resources.resource_filename(
    __name__, 'resources/ani-1x_dft_x8ens/train0/networks/')

buildin_model_prefix = pkg_resources.resource_filename(
    __name__, 'resources/ani-1x_dft_x8ens/train')

buildin_ensemble = 8


class Constants(Mapping):

    def __init__(self, filename=buildin_const_file):
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
        yield 'species'

    def __len__(self):
        return 8

    def __getitem__(self, item):
        return getattr(self, item)

    def species_to_tensor(self, species, device):
        rev = [self.rev_species[s] for s in species]
        return torch.tensor(rev, dtype=torch.long, device=device)


def load_sae(filename=buildin_sae_file):
    """Load self energies from NeuroChem sae file"""
    self_energies = {}
    with open(filename) as f:
        for i in f:
            try:
                line = [x.strip() for x in i.split('=')]
                name = line[0].split(',')[0].strip()
                value = float(line[1])
                self_energies[name] = value
            except Exception:
                pass  # ignore unrecognizable line
    return self_energies


def load_atomic_network(filename):
    """Load atomic network from NeuroChem's .nnf, .wparam and .bparam files

    Parameters
    ----------
    filename : string
        The file name for the `.nnf` file that store network
        hyperparameters. The `.bparam` and `.wparam` must be
        in the same directory

    Returns
    -------
    torch.nn.Sequential
        The loaded atomic network
    """

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

    class Gaussian(torch.nn.Module):
        def forward(self, x):
            return torch.exp(-x*x)

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

            # Activation defined in:
            # https://github.com/Jussmith01/NeuroChem/blob/master/src-atomicnnplib/cunetwork/cuannlayer_t.cu#L868
            activation = s['activation']
            if activation == 6:
                continue
            elif activation == 5:  # Gaussian
                layers.append(Gaussian())
            elif activation == 9:  # CELU
                layers.append(torch.nn.CELU(alpha=0.1))
            else:
                raise NotImplementedError(
                    'Unexpected activation {}'.format(activation))

        return torch.nn.Sequential(*layers)


def load_model(species, from_=None, ensemble=False):
    """If from_=None then ensemble must be a boolean. If ensemble=False,
        then use buildin network0, else use buildin network ensemble.
    If from_ != None, ensemble must be either False or an integer
        specifying the number of networks in the ensemble.
    """
    if from_ is None:
        if not isinstance(ensemble, bool):
            raise TypeError('ensemble must be boolean')
        if ensemble:
            from_ = buildin_model_prefix
            ensemble = buildin_ensemble
        else:
            from_ = buildin_network_dir
    else:
        if not (ensemble is False or isinstance(ensemble, int)):
            raise ValueError('invalid argument ensemble')

    def load_single_model(from_):
        models = []
        for i in species:
            filename = os.path.join(from_, 'ANN-{}.nnf'.format(i))
            models.append((i, load_atomic_network(filename)))
        return ANIModel(models)

    if ensemble is False:
        return load_single_model(from_)
    else:
        assert isinstance(ensemble, int)
        models = []
        for i in range(ensemble):
            network_dir = os.path.join('{}{}'.format(from_, i), 'networks')
            models.append(load_single_model(network_dir))
        return Ensemble(models)
