import os
import torch
from .ani_model import ANIModel
from .neurochem_atomic_network import NeuroChemAtomicNetwork
from ..env import buildin_network_dir, buildin_model_prefix, buildin_ensemble


class NeuroChemNNP(ANIModel):

    def __init__(self, species, from_=None, ensemble=False, benchmark=False):
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
        for network_dir, suffix in zip(network_dirs, suffixes):
            for i in species:
                filename = os.path.join(
                    network_dir, 'ANN-{}.nnf'.format(i))
                model_X = NeuroChemAtomicNetwork(filename)
                models['model_' + i + suffix] = model_X
        super(NeuroChemNNP, self).__init__(species, suffixes, reducer,
                                           models, benchmark)
