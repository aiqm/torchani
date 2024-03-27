#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 23:59:10 2023

@author: rkb19187
"""
import torch, torchani

class Logger:
    def __init__(self, logfile=None, verbose=True):
        if type(logfile) == str:
            self.Logfile = open(logfile, 'w')
        else:
            self.Logfile = False
        
        self.verbose = bool(verbose)
    
    def Log(self, string):
        string = str(string)
        if self.Logfile != False:
            self.Logfile.write(string)
            self.Logfile.write("\n")
            self.Logfile.flush()
        if self.verbose:
            print(string)
        
    def close(self):
        if self.Logfile != False:
            self.Logfile.close()
            self.Logfile = False
            
            
def get_networks(aev_dim, celu0, celu1, noutputs):
    networks = {}
    networks["H_network"] = torch.nn.Sequential(
        torch.nn.Linear(aev_dim, 256),
        torch.nn.CELU(celu0),
        torch.nn.Linear(256, 192),
        torch.nn.CELU(celu1),
        torch.nn.Linear(192, 160),
        torch.nn.CELU(celu1),
        torch.nn.Linear(160, noutputs)
    )
    networks["C_network"] = torch.nn.Sequential(
        torch.nn.Linear(aev_dim, 224),
        torch.nn.CELU(celu0),
        torch.nn.Linear(224, 192),
        torch.nn.CELU(celu1),
        torch.nn.Linear(192, 160),
        torch.nn.CELU(celu1),
        torch.nn.Linear(160, noutputs)
    )
    networks["N_network"] = torch.nn.Sequential(
        torch.nn.Linear(aev_dim, 192),
        torch.nn.CELU(celu0),
        torch.nn.Linear(192, 160),
        torch.nn.CELU(celu1),
        torch.nn.Linear(160, 128),
        torch.nn.CELU(celu1),
        torch.nn.Linear(128, noutputs)
    )
    networks["O_network"] = torch.nn.Sequential(
        torch.nn.Linear(aev_dim, 192),
        torch.nn.CELU(celu0),
        torch.nn.Linear(192, 160),
        torch.nn.CELU(celu1),
        torch.nn.Linear(160, 128),
        torch.nn.CELU(celu1),
        torch.nn.Linear(128, noutputs)
    )
    networks["Cl_network"] = torch.nn.Sequential(
        torch.nn.Linear(aev_dim, 160),
        torch.nn.CELU(celu0),
        torch.nn.Linear(160, 128),
        torch.nn.CELU(celu1),
        torch.nn.Linear(128, 96),
        torch.nn.CELU(celu1),
        torch.nn.Linear(96, noutputs)
    )
    networks["F_network"] = torch.nn.Sequential(
        torch.nn.Linear(aev_dim, 160),
        torch.nn.CELU(celu0),
        torch.nn.Linear(160, 128),
        torch.nn.CELU(celu1),
        torch.nn.Linear(128, 96),
        torch.nn.CELU(celu1),
        torch.nn.Linear(96, noutputs)
    )
    networks["K_network"] = torch.nn.Sequential(
        torch.nn.Linear(aev_dim, 256),
        torch.nn.CELU(celu0),
        torch.nn.Linear(256, 192),
        torch.nn.CELU(celu1),
        torch.nn.Linear(192, 160),
        torch.nn.CELU(celu1),
        torch.nn.Linear(160, noutputs)
    )
    networks["S_network"] = torch.nn.Sequential(
        torch.nn.Linear(aev_dim, 256),
        torch.nn.CELU(celu0),
        torch.nn.Linear(256, 192),
        torch.nn.CELU(celu1),
        torch.nn.Linear(192, 160),
        torch.nn.CELU(celu1),
        torch.nn.Linear(160, noutputs)
    )
    networks["Ir_network"] = torch.nn.Sequential(
        torch.nn.Linear(aev_dim, 256),
        torch.nn.CELU(celu0),
        torch.nn.Linear(256, 192),
        torch.nn.CELU(celu1),
        torch.nn.Linear(192, 160),
        torch.nn.CELU(celu1),
        torch.nn.Linear(160, noutputs)
    )
    return networks

def get_aev_comp(Rcr, Rca, num_species, device):
    EtaR = torch.tensor([1.6000000e+01], device=device)
    ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
    Zeta = torch.tensor([3.2000000e+01], device=device)
    ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
    EtaA = torch.tensor([8.0000000e+00], device=device)
    ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.8500000e+00, 2.2000000e+00], device=device)
    return torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
