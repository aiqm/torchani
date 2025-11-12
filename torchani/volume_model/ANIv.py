import torch
import sys
sys.path.append("/home/jolmos/programas/torchani_sandbox/")
import torchani

class ANIv(torch.nn.Module):
    """
    Predicts atomic mbis volumes as absolute values
    """
    def __init__(self, 
        species_converter, 
        aev_computer, 
        volume_networks, 
    ):
        super().__init__()
        self.species_converter = species_converter
        self.aev_computer = aev_computer
        self.neighborlist = aev_computer.neighborlist
        self.volume_networks = volume_networks       


        
    def forward(self, atomic_nums, coords, ensemble_values = False):
        elem_idxs = self.species_converter(atomic_nums)    #(BxN)
        neighbors = self.neighborlist(self.aev_computer.radial.cutoff, elem_idxs, coords, cell = None, pbc = None)
        aevs = self.aev_computer.compute_from_neighbors(elem_idxs, coords, neighbors)  #(BxNxD)
        volumes = coords.new_zeros(elem_idxs.shape)
        volumes += self.volume_networks(elem_idxs, aevs, atomic = True, ensemble_values = False) #(BxN)
        volumes = torch.nn.GELU()(volumes)  #gelu para remover sacar valores negativos
        
        return volumes


class ANIdv(torch.nn.Module):
    """
    Predicts atomic mbis volumes as delta values
    """
    def __init__(self, 
        species_converter, 
        aev_computer, 
        volume_networks,
        volume_shifter,
    ):
        super().__init__()
        self.species_converter = species_converter
        self.aev_computer = aev_computer
        self.neighborlist = aev_computer.neighborlist
        self.volume_networks = volume_networks       
        self.volume_shifter = volume_shifter

        
    def forward(self, atomic_nums, coords, ensemble_values = False):
        elem_idxs = self.species_converter(atomic_nums)    #(BxN)
        neighbors = self.neighborlist(self.aev_computer.radial.cutoff, elem_idxs, coords, cell = None, pbc = None)
        aevs = self.aev_computer.compute_from_neighbors(elem_idxs, coords, neighbors)  #(BxNxD)
        volumes = coords.new_zeros(elem_idxs.shape)
        volumes += self.volume_networks(elem_idxs, aevs, atomic = True, ensemble_values = False) #(BxN)
        volumes += self.volume_shifter(elem_idxs, atomic = True)

        return volumes

#The following class predicts energys and volumes, its untested, dont use it.
class _ANIv(torch.nn.Module):
    def __init__(self, 
        species_converter, 
        aev_computer, 
        #energy_networks, 
        volume_networks, 
        #energy_shifter,
        volume_shifter = None,
    ):
        super().__init__()
        self.species_converter = species_converter
        self.aev_computer = aev_computer
        self.neighborlist = aev_computer.neighborlist
        #self.energy_networks = energy_networks
        self.volume_networks = volume_networks
        #self.energy_shifter = energy_shifter
        
        if volume_shifter != None:
            self.has_vol_shift = True
            self.volume_shifter = volume_shifter
        else:
            self.has_vol_shift = None
        
    def forward(self, atomic_nums, coords, atomic = False, ensemble_values = False):
        elem_idxs = self.species_converter(atomic_nums)    #(BxN)
        neighbors = self.neighborlist(self.aev_computer.radial.cutoff, elem_idxs, coords, cell = None, pbc = None)
        aevs = self.aev_computer.compute_from_neighbors(elem_idxs, coords, neighbors)  #(BxNxD)

        #energies = coords.new_zeros(elem_idxs.shape[0])
        volumes = coords.new_zeros(elem_idxs.shape)

        #if atomic:
        #    energies = energies.unsqueeze(1)
        #if ensemble_values:
        #    energies = energies.unsqueeze(0)

        #energies += self.energy_networks(elem_idxs, aevs, atomic, ensemble_values)
        volumes += self.volume_networks(elem_idxs, aevs, atomic = True, ensemble_values = False) #(BxN)
        volumes = torch.nn.GELU(volumes)  #gelu para remover sacar valores negativos

        #aca faltaria aplicarle un relu/celu o algo para que no prediga valores negativos

        #energies += self.energy_shifter(elem_idxs, atomic=atomic)
        
        if self.has_vol_shift:
            volumes += self.volume_shifter(elem_idxs, atomic=True)
        #ojo que si estoy prediciendo DV en vez de volumenes entonces tengo que permitir valores negativos
        

        return volumes


