import pkg_resources
import torch
from collections.abc import Mapping


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


def load_sae(filename=buildin_sae_file):
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
