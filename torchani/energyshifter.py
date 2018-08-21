import torch
from .neurochem import buildin_sae_file


class EnergyShifter(torch.nn.Module):

    def __init__(self, species, self_energy_file=buildin_sae_file):
        super(EnergyShifter, self).__init__()
        # load self energies
        self.self_energies = {}
        with open(self_energy_file) as f:
            for i in f:
                try:
                    line = [x.strip() for x in i.split('=')]
                    name = line[0].split(',')[0].strip()
                    value = float(line[1])
                    self.self_energies[name] = value
                except Exception:
                    pass  # ignore unrecognizable line
        self_energies_tensor = [self.self_energies[s] for s in species]
        self.register_buffer('self_energies_tensor',
                             torch.tensor(self_energies_tensor,
                                          dtype=torch.double))

    def sae(self, species):
        self_energies = self.self_energies_tensor[species]
        self_energies[species == -1] = 0
        return self_energies.sum(dim=1)

    def subtract_from_dataset(self, species, coordinates, properties):
        dtype = properties['energies'].dtype
        device = properties['energies'].device
        properties['energies'] -= self.sae(species).to(dtype).to(device)
        return species, coordinates, properties

    def forward(self, species_energies):
        species, energies = species_energies
        sae = self.sae(species).to(energies.dtype).to(energies.device)
        return species, energies + sae
