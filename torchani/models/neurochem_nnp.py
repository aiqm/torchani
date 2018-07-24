import os
import torch
from .ani_model import ANIModel
from .neurochem_atomic_network import NeuroChemAtomicNetwork
from ..env import buildin_network_dir, buildin_model_prefix, buildin_ensemble

class NeuroChemNNP(ANIModel):

    def __init__(self, aev_computer, from_=None, ensemble=False,
                 derivative=False, derivative_graph=False, benchmark=False):
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

        if ensemble is False:
            network_dirs = [from_]
            suffixes = ['']
        else:
            assert isinstance(ensemble, int)
            network_prefix = from_
            network_dirs = []
            suffixes = []
            for i in range(ensemble):
                suffix = '{}'.format(i)
                network_dir = os.path.join(
                    network_prefix+suffix, 'networks')
                network_dirs.append(network_dir)
                suffixes.append(suffix)

        reducer = torch.sum

        models = {}
        output_length = None
        for network_dir, suffix in zip(network_dirs, self.suffixes):
            for i in aev_computer.species:
                filename = os.path.join(
                    network_dir, 'ANN-{}.nnf'.format(i))
                model_X = NeuroChemAtomicNetwork(
                    aev_computer.dtype, aev_computer.device,
                    filename)
                if output_length is None:
                    output_length = model_X.output_length
                elif output_length != model_X.output_length:
                    raise ValueError(
                        '''output length of each atomic neural networt
                        must match''')
                models['model_' + i + suffix] = model_X
        super(NeuroChemNNP, self).__init__(aev_computer, suffixes, reducer,
                                           output_length, models, derivative,
                                           derivative_graph, benchmark)