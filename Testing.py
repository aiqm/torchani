# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:57:30 2024

@author: rkb19187
"""

import pathlib, json, pandas, sys
import torch, torchani
from colour import Color
import orca_parser
import matplotlib.pyplot as plt
import numpy as np
from Logger import *

pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)

device = torch.device("cpu")

test_mol = orca_parser.ORCAParse("testing/cubane/wB97X_D4_SVP.out")
test_mol.parse_coords()
test_mol.parse_energies()
test_mol.parse_free_energy()


with open("Training/training_config.json") as jin:
    config = json.load(jin)

selfE = {}

for ds in config["ds"]:
    selfE_file = ds.replace("\\", "/").replace("/", "_") + "_Self_Energies.csv"
    print(ds, selfE_file)
    selfE[ds] = pandas.read_csv(f"Training/{selfE_file}", index_col=0)
    

networks = get_networks(config["aev_dim"], config["celu0"], config["celu1"], noutputs = len(config["ds"]))
NNs = []
for element in config["species_order"]:
    NNs.append(networks[f"{element}_network"])
nn = torchani.ANIModel(NNs, config['Min'], config['Max'], noutputs = len(config["ds"]))

master_checkpoint = torch.load("Training/best.pt", map_location=device)
nn.load_state_dict(master_checkpoint)

species_to_tensor = torchani.utils.ChemicalSymbolsToInts(config["species_order"])
species = species_to_tensor(test_mol.asemol.get_chemical_symbols()).to(device)
species = species.unsqueeze(0)

coordinates = torch.tensor(test_mol.asemol.get_positions())
coordinates = coordinates.to(device).to(next(nn.parameters()).dtype)
coordinates = coordinates.unsqueeze(0)

aev_computer = get_aev_comp(config["Rcr"], config["Rca"], len(config["species_order"]), device)
AEV = aev_computer((species, coordinates))

nn_output = nn(AEV)
nn_output = [x.energies.detach().numpy()[0] for x in nn_output]

predictions = pandas.DataFrame()
predictions["DNN"] = nn_output
predictions["SelfE"] = 0
predictions.index = config["ds"]

#for index in predictions.index:
for index in config["ds"]:
    for atom in test_mol.asemol.get_chemical_symbols():
        predictions.at[index, "SelfE"] += selfE[index].at[atom, "SelfEnergy"]
    predictions.at[index, "DNN"] = predictions.at[index, "DNN"] + predictions.at[index, "SelfE"]

#predictions.at["DataDump/GasZ=0_rmsd=2.h5", "DFT"] = test_mol.Gibbs
predictions.at["DataDump/GasZ=0_rmsd=2.h5", "DFT"] = test_mol.energies[0]
predictions.at["DataDump/ANI1x_OGP/ANI+OGP.h5", "DFT"] = -309.367605064
predictions["err"] = (predictions["DNN"] - predictions["DFT"]) * 627.5

print(predictions)

#sys.exit()

#### Add in results for individually training neural networks
networks = get_networks(config["aev_dim"], config["celu0"], config["celu1"], noutputs = 1)
NNs = []
for element in config["species_order"]:
    NNs.append(networks[f"{element}_network"])
nn = torchani.ANIModel(NNs, config['Min'], config['Max'], noutputs = 1)
checkpoint = torch.load("Training/Individual/Gas0SVP/best.pt", map_location=device)
nn.load_state_dict(checkpoint)
nn_output = nn(AEV)
predictions.at["DataDump/GasZ=0_rmsd=2.h5", "Single"] = float(nn(AEV)[0].energies[0].detach().numpy())
predictions.at["DataDump/GasZ=0_rmsd=2.h5", "Single"] += predictions.at[index, "SelfE"]
print(predictions)



# OGP
ds = 'DataDump/ANI1x_OGP/ANI+OGP.h5'
with open("Training/Individual/ANI_OGP/training_config.json") as jin:
    config = json.load(jin)
networks = get_networks(config["aev_dim"], config["celu0"], config["celu1"], noutputs = 1)
NNs = []
for element in config["species_order"]:
    NNs.append(networks[f"{element}_network"])
nn = torchani.ANIModel(NNs, config['Min'], config['Max'], noutputs = 1)
checkpoint = torch.load("Training/Individual/ANI_OGP/best.pt", map_location=device)
# =============================================================================
# for key in list(master_checkpoint.keys()):
#     print(key, master_checkpoint[key].shape, checkpoint[key].shape)
# =============================================================================
nn.load_state_dict(checkpoint)
nn_output = nn(AEV)
predictions.at[ds, "Single"] = float(nn(AEV)[0].energies[0].detach().numpy())
predictions.at[ds, "Single"] += predictions.at[index, "SelfE"]
print(predictions)

#SZ
ds = 'DataDump/Surajit_Nandi/SZ.h5'
with open("Training/Individual/Nandi_SZ/training_config.json") as jin:
    config = json.load(jin)
networks = get_networks(config["aev_dim"], config["celu0"], config["celu1"], noutputs = 1)
networks = get_networks(config["aev_dim"], config["celu0"], config["celu1"], noutputs = 1)
NNs = []
for element in config["species_order"]:
    NNs.append(networks[f"{element}_network"])
nn = torchani.ANIModel(NNs, config['Min'], config['Max'], noutputs = 1)
checkpoint = torch.load("Training/Individual/Nandi_SZ/best.pt", map_location=device)
nn.load_state_dict(checkpoint)
nn_output = nn(AEV)
predictions.at[ds, "Single"] = float(nn(AEV)[0].energies[0].detach().numpy())
predictions.at[ds, "Single"] += predictions.at[index, "SelfE"]
print(predictions)


