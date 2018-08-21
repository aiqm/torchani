import torch


class EnergyShifter(torch.nn.Module):

    def __init__(self, species, self_energies):
        super(EnergyShifter, self).__init__()
        self_energies_tensor = [self_energies[s] for s in species]
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
