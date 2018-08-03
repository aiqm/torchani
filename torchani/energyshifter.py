import torch
from .env import buildin_sae_file


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

    def sae_from_list(self, species):
        energies = [self.self_energies[i] for i in species]
        return sum(energies)

    def sae_from_tensor(self, species):
        return self.self_energies_tensor[species].sum().item()

    def subtract_from_dataset(self, data):
        sae = self.sae_from_list(data['species'])
        data['energies'] -= sae
        return data

    def forward(self, species_energies):
        species, energies = species_energies
        sae = self.sae_from_tensor(species)
        return species, energies + sae
