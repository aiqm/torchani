#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 23:59:10 2023

@author: rkb19187
"""
import torch

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