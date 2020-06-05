# -*- coding: utf-8 -*-
"""Tools for loading/running NeuroChem input files."""

import torch
import os
import bz2
import lark
import struct
import itertools
import math
import timeit
import collections
import sys
from ..nn import ANIModel, Ensemble, Gaussian, Sequential
from ..utils import EnergyShifter, ChemicalSymbolsToInts
from ..aev import AEVComputer
from torch.optim import AdamW
from collections import OrderedDict
from torchani.units import hartree2kcalmol


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
    self_energies = []
    d = {}
    with open(filename) as f:
        for i in f:
            line = [x.strip() for x in i.split('=')]
            species = line[0].split(',')[0].strip()
            index = int(line[0].split(',')[1].strip())
            value = float(line[1])
            d[species] = value
            self_energies.append((index, value))
    self_energies = [i for _, i in sorted(self_energies)]
    if return_dict:
        return EnergyShifter(self_energies), d
    return EnergyShifter(self_energies)


def _get_activation(activation_index):
    # Activation defined in:
    # https://github.com/Jussmith01/NeuroChem/blob/stable1/src-atomicnnplib/cunetwork/cuannlayer_t.cu#L920
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


if sys.version_info[0] > 2:

    class Trainer:
        """Train with NeuroChem training configurations.

        Arguments:
            filename (str): Input file name
            device (:class:`torch.device`): device to train the model
            tqdm (bool): whether to enable tqdm
            tensorboard (str): Directory to store tensorboard log file, set to
                ``None`` to disable tensorboard.
            checkpoint_name (str): Name of the checkpoint file, checkpoints
                will be stored in the network directory with this file name.
        """

        def __init__(self, filename, device=torch.device('cuda'), tqdm=False,
                     tensorboard=None, checkpoint_name='model.pt'):

            from ..data import load  # noqa: E402

            class dummy:
                pass

            self.imports = dummy()
            self.imports.load = load

            self.filename = filename
            self.device = device
            self.checkpoint_name = checkpoint_name
            self.weights = []
            self.biases = []

            if tqdm:
                import tqdm
                self.tqdm = tqdm.tqdm
            else:
                self.tqdm = None
            if tensorboard is not None:
                import torch.utils.tensorboard
                self.tensorboard = torch.utils.tensorboard.SummaryWriter(
                    log_dir=tensorboard)

                self.training_eval_every = 20
            else:
                self.tensorboard = None

            with open(filename, 'r') as f:
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    network_setup, params = self._parse_yaml(f)
                else:
                    network_setup, params = self._parse(f.read())
                self._construct(network_setup, params)

        def _parse(self, txt):
            parser = lark.Lark(r'''
            identifier : CNAME

            outer_assign : identifier "=" value
            params : outer_assign *

            inner_assign : identifier "=" value ";"
            input_size : "inputsize" "=" INT ";"

            layer : "layer" "[" inner_assign * "]"

            atom_type : WORD

            atom_net : "atom_net" atom_type "$" layer * "$"

            network_setup: "network_setup" "{" input_size atom_net * "}"

            start: params network_setup params

            value : SIGNED_INT
                | SIGNED_FLOAT
                | STRING_VALUE

            STRING_VALUE : ("_"|"-"|"."|"/"|LETTER)("_"|"-"|"."|"/"|LETTER|DIGIT)*

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
            %ignore /!.*/
            ''')  # noqa: E501
            tree = parser.parse(txt)

            class TreeExec(lark.Transformer):

                def identifier(self, v):
                    v = v[0].value
                    return v

                def value(self, v):
                    if len(v) == 1:
                        v = v[0]
                        if v.type == 'STRING_VALUE':
                            v = v.value
                        elif v.type == 'SIGNED_INT' or v.type == 'INT':
                            v = int(v.value)
                        elif v.type == 'SIGNED_FLOAT' or v.type == 'FLOAT':
                            v = float(v.value)
                        else:
                            raise ValueError('unexpected type')
                    else:
                        raise ValueError('length of value can only be 1 or 2')
                    return v

                def outer_assign(self, v):
                    name = v[0]
                    value = v[1]
                    return name, value

                inner_assign = outer_assign

                def params(self, v):
                    return v

                def network_setup(self, v):
                    intput_size = int(v[0])
                    atomic_nets = dict(v[1:])
                    return intput_size, atomic_nets

                def layer(self, v):
                    return dict(v)

                def atom_net(self, v):
                    atom_type = v[0]
                    layers = v[1:]
                    return atom_type, layers

                def atom_type(self, v):
                    return v[0].value

                def start(self, v):
                    network_setup = v[1]
                    del v[1]
                    return network_setup, dict(itertools.chain(*v))

                def input_size(self, v):
                    return v[0].value

            return TreeExec().transform(tree)

        def _parse_yaml(self, f):
            import yaml
            params = yaml.safe_load(f)
            network_setup = params['network_setup']
            del params['network_setup']
            network_setup = (network_setup['inputsize'],
                             network_setup['atom_net'])
            return network_setup, params

        def _construct(self, network_setup, params):
            dir_ = os.path.dirname(os.path.abspath(self.filename))

            # delete ignored params
            def del_if_exists(key):
                if key in params:
                    del params[key]

            def assert_param(key, value):
                if key in params and params[key] != value:
                    raise NotImplementedError(key + ' not supported yet')
                del params[key]

            # weights and biases initialization
            def init_params(m):

                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(m.weight, a=1.0)
                    torch.nn.init.zeros_(m.bias)

            del_if_exists('gpuid')
            del_if_exists('nkde')
            del_if_exists('fmult')
            del_if_exists('cmult')
            del_if_exists('decrate')
            del_if_exists('mu')
            assert_param('pbc', 0)
            assert_param('force', 0)
            assert_param('energy', 1)
            assert_param('moment', 'ADAM')
            assert_param('runtype', 'ANNP_CREATE_HDNN_AND_TRAIN')
            assert_param('adptlrn', 'OFF')
            assert_param('tmax', 0)
            assert_param('ntwshr', 0)

            # load parameters
            self.const_file = os.path.join(dir_, params['sflparamsfile'])
            self.consts = Constants(self.const_file)
            self.aev_computer = AEVComputer(**self.consts)
            del params['sflparamsfile']
            self.sae_file = os.path.join(dir_, params['atomEnergyFile'])
            self.shift_energy, self.sae = load_sae(self.sae_file, return_dict=True)
            del params['atomEnergyFile']
            network_dir = os.path.join(dir_, params['ntwkStoreDir'])
            if not os.path.exists(network_dir):
                os.makedirs(network_dir)
            self.model_checkpoint = os.path.join(network_dir,
                                                 self.checkpoint_name)
            del params['ntwkStoreDir']
            self.max_nonimprove = params['tolr']
            del params['tolr']
            self.init_lr = params['eta']
            del params['eta']
            self.lr_decay = params['emult']
            del params['emult']
            self.min_lr = params['tcrit']
            del params['tcrit']
            self.training_batch_size = params['tbtchsz']
            del params['tbtchsz']
            self.validation_batch_size = params['vbtchsz']
            del params['vbtchsz']
            self.nmax = math.inf if params['nmax'] == 0 else params['nmax']
            del params['nmax']

            # construct networks
            input_size, network_setup = network_setup
            if input_size != self.aev_computer.aev_length:
                raise ValueError('AEV size and input size does not match')
            atomic_nets = OrderedDict()
            for atom_type in self.consts.species:
                layers = network_setup[atom_type]
                modules = []
                i = input_size
                for layer in layers:
                    o = layer['nodes']
                    del layer['nodes']
                    if layer['type'] != 0:
                        raise ValueError('Unsupported layer type')
                    del layer['type']
                    module = torch.nn.Linear(i, o)
                    modules.append(module)
                    activation = _get_activation(layer['activation'])
                    if activation is not None:
                        modules.append(activation)
                    del layer['activation']
                    if 'l2norm' in layer:
                        if layer['l2norm'] == 1:
                            self.weights.append({
                                'params': [module.weight],
                                'weight_decay': layer['l2valu'],
                            })
                        else:
                            self.weights.append({
                                'params': [module.weight],
                            })
                        del layer['l2norm']
                        del layer['l2valu']
                    else:
                        self.weights.append({
                            'params': [module.weight],
                        })
                    self.biases.append({
                        'params': [module.bias],
                    })
                    if layer:
                        raise ValueError(
                            'unrecognized parameter in layer setup')
                    i = o
                atomic_nets[atom_type] = torch.nn.Sequential(*modules)
            self.nn = ANIModel(atomic_nets)

            # initialize weights and biases
            self.nn.apply(init_params)
            self.model = Sequential(self.aev_computer, self.nn).to(self.device)

            # loss functions
            self.mse_se = torch.nn.MSELoss(reduction='none')
            self.mse_sum = torch.nn.MSELoss(reduction='sum')

            if params:
                raise ValueError('unrecognized parameter')

            self.best_validation_rmse = math.inf

        def load_data(self, training_path, validation_path):
            """Load training and validation dataset from file."""
            self.training_set = self.imports.load(training_path).subtract_self_energies(self.sae).species_to_indices().shuffle().collate(self.training_batch_size).cache()
            self.validation_set = self.imports.load(validation_path).subtract_self_energies(self.sae).species_to_indices().shuffle().collate(self.validation_batch_size).cache()

        def evaluate(self, dataset):
            """Run the evaluation"""
            total_mse = 0.0
            count = 0
            for properties in dataset:
                species = properties['species'].to(self.device)
                coordinates = properties['coordinates'].to(self.device).float()
                true_energies = properties['energies'].to(self.device).float()
                _, predicted_energies = self.model((species, coordinates))
                total_mse += self.mse_sum(predicted_energies, true_energies).item()
                count += predicted_energies.shape[0]
            return hartree2kcalmol(math.sqrt(total_mse / count))

        def run(self):
            """Run the training"""
            start = timeit.default_timer()
            no_improve_count = 0

            AdamW_optim = AdamW(self.weights, lr=self.init_lr)
            SGD_optim = torch.optim.SGD(self.biases, lr=self.init_lr)

            AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                AdamW_optim,
                factor=0.5,
                patience=100,
                threshold=0)
            SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                SGD_optim,
                factor=0.5,
                patience=100,
                threshold=0)

            while True:
                rmse = self.evaluate(self.validation_set)
                learning_rate = AdamW_optim.param_groups[0]['lr']
                if learning_rate < self.min_lr or AdamW_scheduler.last_epoch > self.nmax:
                    break

                # checkpoint
                if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
                    no_improve_count = 0
                    torch.save(self.nn.state_dict(), self.model_checkpoint)
                else:
                    no_improve_count += 1

                if no_improve_count > self.max_nonimprove:
                    break

                AdamW_scheduler.step(rmse)
                SGD_scheduler.step(rmse)

                if self.tensorboard is not None:
                    self.tensorboard.add_scalar('validation_rmse', rmse, AdamW_scheduler.last_epoch)
                    self.tensorboard.add_scalar('best_validation_rmse', AdamW_scheduler.best, AdamW_scheduler.last_epoch)
                    self.tensorboard.add_scalar('learning_rate', learning_rate, AdamW_scheduler.last_epoch)
                    self.tensorboard.add_scalar('no_improve_count_vs_epoch', no_improve_count, AdamW_scheduler.last_epoch)

                for i, properties in self.tqdm(
                    enumerate(self.training_set),
                    total=len(self.training_set),
                    desc='epoch {}'.format(AdamW_scheduler.last_epoch)
                ):
                    species = properties['species'].to(self.device)
                    coordinates = properties['coordinates'].to(self.device).float()
                    true_energies = properties['energies'].to(self.device).float()
                    num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
                    _, predicted_energies = self.model((species, coordinates))
                    loss = (self.mse_se(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
                    AdamW_optim.zero_grad()
                    SGD_optim.zero_grad()
                    loss.backward()
                    AdamW_optim.step()
                    SGD_optim.step()

                    # write current batch loss to TensorBoard
                    if self.tensorboard is not None:
                        self.tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * len(self.training_set) + i)

                # log elapsed time
                elapsed = round(timeit.default_timer() - start, 2)
                if self.tensorboard is not None:
                    self.tensorboard.add_scalar('time_vs_epoch', elapsed, AdamW_scheduler.last_epoch)


__all__ = ['Constants', 'load_sae', 'load_model', 'load_model_ensemble', 'Trainer']
