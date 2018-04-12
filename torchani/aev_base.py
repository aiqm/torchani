import torch
import pkg_resources

default_const_file = pkg_resources.resource_filename(__name__, 'data/rHCNO-4.6R_16-3.1A_a4-8_3.params')

class AEVComputer:

    def __init__(self, dtype=torch.cuda.float32, const_file=default_const_file):
        self.dtype = dtype

        # load constants
        self.constants = {}
        with open(const_file) as f:
            for i in f:
                try:
                    line = [x.strip() for x in i.split('=')]
                    name = line[0]
                    value = line[1]
                    if name == 'Rcr' or name == 'Rca':
                        self.constants[name] = float(value)
                    elif name in ['EtaR', 'ShfR', 'Zeta', 'ShfZ', 'EtaA', 'ShfA']:
                        value = [ float(x.strip()) for x in value.replace('[','').replace(']','').split(',')]
                        self.constants[name] = value
                    elif name == 'Atyp':
                        value = [ x.strip() for x in value.replace('[','').replace(']','').split(',')]
                        self.species = value
                except:
                    pass  # ignore unrecognizable line

    def per_species_radial_length(self):
        return len(self.constants['EtaR']) * len(self.constants['ShfR'])

    def radial_length(self):
        return len(self.species) * self.per_species_radial_length()
    
    def per_species_angular_length(self):
        return len(self.constants['EtaA']) * len(self.constants['Zeta']) * len(self.constants['ShfA']) * len(self.constants['ShfZ'])

    def angular_length(self):
        species = len(self.species)
        return int((species * (species + 1)) / 2) * self.per_species_angular_length()

    def __call__(self, coordinates, species):
        raise NotImplementedError('subclass must override this method')