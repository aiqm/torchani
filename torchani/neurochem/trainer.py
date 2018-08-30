import itertools
import ignite
import math
import timeit
import torch
import os
import lark
from ..ignite import Container, MSELoss, TransformedLoss, RMSEMetric, MAEMetric
from ..data import BatchedANIDataset
from . import _get_activation, Constants, load_sae
from ..nn import ANIModel
from ..aev import AEVComputer


def hartree2kcal(x):
    return 627.509 * x


class Trainer:
    """Train with NeuroChem training configurations.

    Arguments:
        filename (str): Input file name
        device (:class:`torch.device`): device to train the model
        tqdm (bool): whether to enable tqdm
        tensorboard (str): Directory to store tensorboard log file, set to\
            ``None`` to disable tensorboardX.
    """

    def __init__(self, filename, device=torch.device('cuda'),
                 tqdm=False, tensorboard=None):
        self.filename = filename
        self.device = device
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

        value : INT
              | FLOAT
              | STRING_VALUE

        STRING_VALUE : ("_"|"-"|"."|"/"|LETTER)("_"|"-"|"."|"/"|LETTER|DIGIT)*

        %import common.SIGNED_NUMBER
        %import common.LETTER
        %import common.WORD
        %import common.DIGIT
        %import common.INT
        %import common.FLOAT
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
                    elif v.type == 'INT':
                        v = int(v.value)
                    elif v.type == 'FLOAT':
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

    def _construct(self, network_setup, params):
        dir = os.path.dirname(os.path.abspath(self.filename))

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
        self.const_file = os.path.join(dir, params['sflparamsfile'])
        self.consts = Constants(self.const_file)
        self.aev_computer = AEVComputer(**self.consts)
        del params['sflparamsfile']
        self.sae_file = os.path.join(dir, params['atomEnergyFile'])
        self.shift_energy = load_sae(self.sae_file)
        del params['atomEnergyFile']
        network_dir = os.path.join(dir, params['ntwkStoreDir'])
        if not os.path.exists(network_dir):
            os.makedirs(network_dir)
        self.model_checkpoint = os.path.join(network_dir, 'model.pt')
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
                        l2reg.append((0.5 * layer['l2valu'], module))
                    del layer['l2norm']
                    del layer['l2valu']
                if len(layer) > 0:
                    raise ValueError('unrecognized parameter in layer setup')
                i = o
            atomic_nets[atom_type] = torch.nn.Sequential(*modules)
        self.model = ANIModel([atomic_nets[s] for s in self.consts.species])
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

        if len(params) > 0:
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
        """Load training and validation dataset from file"""
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
