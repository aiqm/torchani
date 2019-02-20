# -*- coding: utf-8 -*-
"""Tools for loading/running NeuroChem input files."""

import pkg_resources
import torch
import os
import bz2
import lark
import struct
import itertools
import ignite
import math
import timeit
from collections.abc import Mapping
from ..nn import ANIModel, Ensemble, Gaussian
from ..utils import EnergyShifter, ChemicalSymbolsToInts
from ..aev import AEVComputer
from ..ignite import Container, MSELoss, TransformedLoss, RMSEMetric, MAEMetric


class Constants(Mapping):
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

        value : SIGNED_INT
              | SIGNED_FLOAT
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
    models = []
    for i in species:
        filename = os.path.join(dir_, 'ANN-{}.nnf'.format(i))
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


class BuiltinsAbstract:
    """Base class for loading ANI neural network from configuration files.

    Arguments:
        parent_name (:class:`str`): Base path that other paths are relative to.
        const_file_path (:class:`str`): Path to constant file for ANI model(s).
        sae_file_path (:class:`str`): Path to sae file for ANI model(s).
        ensemble_size (:class:`int`): Number of models in model ensemble.
        ensemble_prefix_path (:class:`str`): Path to prefix of directories of
            models.

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
    def __init__(
            self,
            parent_name,
            const_file_path,
            sae_file_path,
            ensemble_size,
            ensemble_prefix_path):
        self.const_file = pkg_resources.resource_filename(
            parent_name,
            const_file_path)
        self.consts = Constants(self.const_file)
        self.species = self.consts.species
        self.aev_computer = AEVComputer(**self.consts)
        self.sae_file = pkg_resources.resource_filename(
            parent_name,
            sae_file_path)
        self.energy_shifter = load_sae(self.sae_file)
        self.ensemble_size = ensemble_size
        self.ensemble_prefix = pkg_resources.resource_filename(
            parent_name,
            ensemble_prefix_path)
        self.models = load_model_ensemble(self.consts.species,
                                          self.ensemble_prefix,
                                          self.ensemble_size)


class Builtins(BuiltinsAbstract):
    """Container for the builtin ANI-1x model.

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
        const_file_path = 'resources/ani-1x_8x'\
            '/rHCNO-5.2R_16-3.5A_a4-8.params'
        sae_file_path = 'resources/ani-1x_8x/sae_linfit.dat'
        ensemble_size = 8
        ensemble_prefix_path = 'resources/ani-1x_8x/train'
        super(Builtins, self).__init__(
            parent_name,
            const_file_path,
            sae_file_path,
            ensemble_size,
            ensemble_prefix_path
        )


class BuiltinsANI1CCX(BuiltinsAbstract):
    """Container for the builtin ANI-1ccx model.

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
        const_file_path = 'resources/ani-1ccx_8x'\
            '/rHCNO-5.2R_16-3.5A_a4-8.params'
        sae_file_path = 'resources/ani-1ccx_8x/sae_linfit.dat'
        ensemble_size = 8
        ensemble_prefix_path = 'resources/ani-1ccx_8x/train'
        super(BuiltinsANI1CCX, self).__init__(
            parent_name,
            const_file_path,
            sae_file_path,
            ensemble_size,
            ensemble_prefix_path
        )


def hartree2kcal(x):
    return 627.509 * x


from ..data import BatchedANIDataset  # noqa: E402
from ..data import AEVCacheLoader  # noqa: E402


class Trainer:
    """Train with NeuroChem training configurations.

    Arguments:
        filename (str): Input file name
        device (:class:`torch.device`): device to train the model
        tqdm (bool): whether to enable tqdm
        tensorboard (str): Directory to store tensorboard log file, set to
            ``None`` to disable tensorboardX.
        aev_caching (bool): Whether to use AEV caching.
        checkpoint_name (str): Name of the checkpoint file, checkpoints will be
            stored in the network directory with this file name.
    """

    def __init__(self, filename, device=torch.device('cuda'), tqdm=False,
                 tensorboard=None, aev_caching=False,
                 checkpoint_name='model.pt'):
        self.filename = filename
        self.device = device
        self.aev_caching = aev_caching
        self.checkpoint_name = checkpoint_name
        if tqdm:
            import tqdm
            self.tqdm = tqdm.tqdm
        else:
            self.tqdm = None
        if tensorboard is not None:
            import tensorboardX
            self.tensorboard = tensorboardX.SummaryWriter(log_dir=tensorboard)
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
        ''')
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
        network_setup = (network_setup['inputsize'], network_setup['atom_net'])
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
        assert_param('nmax', 0)
        assert_param('ntwshr', 0)

        # load parameters
        self.const_file = os.path.join(dir_, params['sflparamsfile'])
        self.consts = Constants(self.const_file)
        self.aev_computer = AEVComputer(**self.consts)
        del params['sflparamsfile']
        self.sae_file = os.path.join(dir_, params['atomEnergyFile'])
        self.shift_energy = load_sae(self.sae_file)
        del params['atomEnergyFile']
        network_dir = os.path.join(dir_, params['ntwkStoreDir'])
        if not os.path.exists(network_dir):
            os.makedirs(network_dir)
        self.model_checkpoint = os.path.join(network_dir, self.checkpoint_name)
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

        # construct networks
        input_size, network_setup = network_setup
        if input_size != self.aev_computer.aev_length():
            raise ValueError('AEV size and input size does not match')
        l2reg = []
        atomic_nets = {}
        for atom_type in network_setup:
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
                        # NB: The "L2" implemented in NeuroChem is actually not
                        # L2 but weight decay. The difference of these two is:
                        # https://arxiv.org/pdf/1711.05101.pdf
                        # There is a pull request on github/pytorch
                        # implementing AdamW, etc.:
                        # https://github.com/pytorch/pytorch/pull/4429
                        # There is no plan to support the "L2" settings in
                        # input file before AdamW get merged into pytorch.
                        raise NotImplementedError('L2 not supported yet')
                    del layer['l2norm']
                    del layer['l2valu']
                if layer:
                    raise ValueError('unrecognized parameter in layer setup')
                i = o
            atomic_nets[atom_type] = torch.nn.Sequential(*modules)
        self.model = ANIModel([atomic_nets[s] for s in self.consts.species])
        if self.aev_caching:
            self.nnp = self.model
        else:
            self.nnp = torch.nn.Sequential(self.aev_computer, self.model)
        self.container = Container({'energies': self.nnp}).to(self.device)

        # losses
        def l2():
            return sum([c * (m.weight ** 2).sum() for c, m in l2reg])
        self.mse_loss = TransformedLoss(MSELoss('energies'),
                                        lambda x: x + l2())
        self.exp_loss = TransformedLoss(
            MSELoss('energies'),
            lambda x: 0.5 * (torch.exp(2 * x) - 1) + l2())

        if params:
            raise ValueError('unrecognized parameter')

        self.global_epoch = 0
        self.global_iteration = 0
        self.best_validation_rmse = math.inf

    def evaluate(self, dataset):
        """Evaluate on given dataset to compute RMSE and MAE."""
        evaluator = ignite.engine.create_supervised_evaluator(
            self.container,
            metrics={
                'RMSE': RMSEMetric('energies'),
                'MAE': MAEMetric('energies'),
            }
        )
        evaluator.run(dataset)
        metrics = evaluator.state.metrics
        return hartree2kcal(metrics['RMSE']), hartree2kcal(metrics['MAE'])

    def load_data(self, training_path, validation_path):
        """Load training and validation dataset from file.

        If AEV caching is enabled, then the arguments are path to the cache
        directory, otherwise it should be path to the dataset.
        """
        if self.aev_caching:
            self.training_set = AEVCacheLoader(training_path)
            self.validation_set = AEVCacheLoader(validation_path)
        else:
            self.training_set = BatchedANIDataset(
                training_path, self.consts.species_to_tensor,
                self.training_batch_size, device=self.device,
                transform=[self.shift_energy.subtract_from_dataset])
            self.validation_set = BatchedANIDataset(
                validation_path, self.consts.species_to_tensor,
                self.validation_batch_size, device=self.device,
                transform=[self.shift_energy.subtract_from_dataset])

    def run(self):
        """Run the training"""
        start = timeit.default_timer()

        def decorate(trainer):

            @trainer.on(ignite.engine.Events.STARTED)
            def initialize(trainer):
                trainer.state.no_improve_count = 0
                trainer.state.epoch += self.global_epoch
                trainer.state.iteration += self.global_iteration

            @trainer.on(ignite.engine.Events.COMPLETED)
            def finalize(trainer):
                self.global_epoch = trainer.state.epoch
                self.global_iteration = trainer.state.iteration

            if self.tqdm is not None:
                @trainer.on(ignite.engine.Events.EPOCH_STARTED)
                def init_tqdm(trainer):
                    trainer.state.tqdm = self.tqdm(
                        total=len(self.training_set), desc='epoch')

                @trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
                def update_tqdm(trainer):
                    trainer.state.tqdm.update(1)

                @trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
                def finalize_tqdm(trainer):
                    trainer.state.tqdm.close()

            @trainer.on(ignite.engine.Events.EPOCH_STARTED)
            def validation_and_checkpoint(trainer):
                trainer.state.rmse, trainer.state.mae = \
                    self.evaluate(self.validation_set)
                if trainer.state.rmse < self.best_validation_rmse:
                    trainer.state.no_improve_count = 0
                    self.best_validation_rmse = trainer.state.rmse
                    torch.save(self.model.state_dict(), self.model_checkpoint)
                else:
                    trainer.state.no_improve_count += 1

                if trainer.state.no_improve_count > self.max_nonimprove:
                    trainer.terminate()

            if self.tensorboard is not None:
                @trainer.on(ignite.engine.Events.EPOCH_STARTED)
                def log_per_epoch(trainer):
                    elapsed = round(timeit.default_timer() - start, 2)
                    epoch = trainer.state.epoch
                    self.tensorboard.add_scalar('time_vs_epoch', elapsed,
                                                epoch)
                    self.tensorboard.add_scalar('learning_rate_vs_epoch', lr,
                                                epoch)
                    self.tensorboard.add_scalar('validation_rmse_vs_epoch',
                                                trainer.state.rmse, epoch)
                    self.tensorboard.add_scalar('validation_mae_vs_epoch',
                                                trainer.state.mae, epoch)
                    self.tensorboard.add_scalar(
                        'best_validation_rmse_vs_epoch',
                        self.best_validation_rmse, epoch)
                    self.tensorboard.add_scalar('no_improve_count_vs_epoch',
                                                trainer.state.no_improve_count,
                                                epoch)

                    # compute training RMSE and MAE
                    if epoch % self.training_eval_every == 1:
                        training_rmse, training_mae = \
                            self.evaluate(self.training_set)
                        self.tensorboard.add_scalar('training_rmse_vs_epoch',
                                                    training_rmse, epoch)
                        self.tensorboard.add_scalar('training_mae_vs_epoch',
                                                    training_mae, epoch)

                @trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
                def log_loss(trainer):
                    iteration = trainer.state.iteration
                    loss = trainer.state.output
                    self.tensorboard.add_scalar('loss_vs_iteration',
                                                loss, iteration)

        lr = self.init_lr

        # training using mse loss first until the validation MAE decrease
        # to < 1 Hartree
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        trainer = ignite.engine.create_supervised_trainer(
            self.container, optimizer, self.mse_loss)
        decorate(trainer)

        @trainer.on(ignite.engine.Events.EPOCH_STARTED)
        def terminate_if_smaller_enough(trainer):
            if trainer.state.mae < 1.0:
                trainer.terminate()

        trainer.run(self.training_set, max_epochs=math.inf)

        while lr > self.min_lr:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            trainer = ignite.engine.create_supervised_trainer(
                self.container, optimizer, self.exp_loss)
            decorate(trainer)
            trainer.run(self.training_set, max_epochs=math.inf)
            lr *= self.lr_decay


__all__ = ['Constants', 'load_sae', 'load_model', 'load_model_ensemble',
           'Builtins', 'Trainer']
