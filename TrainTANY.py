# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:50:25 2024

@author: Alex
"""

# =============================================================================
# Import all the requirements
# =============================================================================
import matplotlib, os, sys, tqdm
if os.name != "nt":
    matplotlib.use('Agg')
import torch
import torchani

# Temp patch, somethings wrong with the way torchany installs
from torchani import nn as torachani_nn
import math, pandas, h5py, pickle, time, json, copy, argparse
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
### TorchAni uses the 'random' module to do the .shuffle of the data
import random, time

# helper function to convert energy unit from Hartree to kcal/mol
from torchani.units import hartree2kcalmol

# sub-modules
from Logger import *

def indices2atoms(specie):
    return [config["species_order"][x] for x in specie if x >= 0]

def dump_json():
    with open(os.path.join(config["output"], "training_config.json"), 'w') as jout:
        json.dump(config, jout, indent=4)
    


# =============================================================================
# Configure the basic variables that we need
# =============================================================================
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--cuda', action='store_true', help='Use CUDA if possible', default=True)
parser.add_argument('--epochs', type=int, help='max Epochs', default=10000, required=False)
parser.add_argument('--multigpu', action='store_true', help='Use CUDA if possible', default=False)
parser.add_argument('--KMP_DUPLICATE_LIB_OK', action='store_true', default=True)
parser.add_argument('--output', type=str, help='Output folder', required=True)
parser.add_argument('--mixed', action='store_true', help='Does the dataset contains mixed explicit charges', default=False, required=False)
#parser.add_argument('-Z', type=int, help='Charge', required=True)
parser.add_argument('--batch_size', type=int, help='batch_size', required=True)
parser.add_argument('--training_frac', type=float, help='Fraction of the dataset to be used for training', default=0.8)
parser.add_argument('--ds', type=str, help='HDF5 dataset(s) for training', required=True, action='append')
parser.add_argument('--valds', type=str, help='HDF5 dataset for validation', required=False)
parser.add_argument('--hard_reset', action='store_true', help='Delete all the file from previous run', default=False, required=False)
parser.add_argument('--log', type=str, help='Verbose Log file', default="Log.txt", required=False)
parser.add_argument('--logfile', type=str, help='Training Log file', default="Training.log", required=False)
parser.add_argument('--model_n', type=int, help='Model number', default=1, required=False)
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate', default=False, required=False)
parser.add_argument('--fc', type=float, help='Force coefficient', default=0.0, required=False)
parser.add_argument('--GraphEveryEpoch', type=int, help='Graph Every N Epochs', default=10, required=False)
parser.add_argument('--HasForces', action='store_true', help='Do the datasets have forces and are we training against them?', default=False, required=False)
parser.add_argument('--lr', type=float, help='initial learning rate', default=1e-3, required=False)
parser.add_argument('--preLoadANI',  action='store_true', help='Transfer learning from another DNN', default=False, required=False)
parser.add_argument('--preLoadANI_model_path', type=str, help='Folder containing the network information', default="ANIWEIGHTS/ani-2x_8x/train0/networks", required=False)
parser.add_argument('--early_stop', type=float, help='early stop (kcal/mol)', default=0.1, required=False)
parser.add_argument('--early_stopping_learning_rate', type=float, help='early_stopping_learning_rate', default=1e-5, required=False)
parser.add_argument('--Rcr', type=float, help='Rcr', default=5.2, required=False)
parser.add_argument('--Rca', type=float, help='Rca', default=3.1, required=False)
parser.add_argument('--remove_self_energy',  action='store_true', help='remove_self_energy', default=True, required=False)
parser.add_argument('--preAEV',  action='store_true', help='preAEV', default=False, required=False)
parser.add_argument('--sigmoid',  action='store_true', help='Apply a sigmoid and scale it between Min and Max over the output', default=False, required=False)

parser.add_argument('--celu0',  type=float, help='celu0', default=0.1, required=False)
parser.add_argument('--celu1',  type=float, help='celu1', default=0.1, required=False)

parser.add_argument('--L1',  action='store_true',  help='Use L1 loss function', default=False)

# Stratified DS
parser.add_argument('--Stratified',  action='store_true', help='Stratified', default=False, required=False)
parser.add_argument('--TrainDS', type=str, help='Training dataset for Stratified training', required=False)
parser.add_argument('--TestDS', type=str, help='Training dataset for Stratified training', required=False)

args = parser.parse_args()
print(args)


config = vars(args) # converts it to dictionary
config["species_order"] = ['C','H','N','O',"F","Cl","K"]
config["species_order"].sort()
config["random_seed"] = int(time.time())
random.seed(config["random_seed"])
os.makedirs(config["output"], exist_ok=True)
if os.path.exists(os.path.join(config["output"], "training_config.json")) and not config["hard_reset"]:
    with open(os.path.join(config["output"], "training_config.json")) as jin:
        jin = json.load(jin)
        if "species_list" in jin:
            config["species_list"] = jin["species_list"]
    
print(config)
dump_json()

# =============================================================================
# Define the 'Logger' class so that we can save information that we need
# =============================================================================
Log = Logger(f"{config['output']}/{config['log']}" if type(config["log"]) == str else None, verbose=True)

for ds in config["ds"]:
    assert os.path.exists(ds), "Dataset not found"

Log.Log("Figuring out which datasets contain which atoms")
species_list = {}
for ds in tqdm.tqdm(config["ds"]):
    if "species_list" in config:
        if ds in config["species_list"]:
            Log.Log(ds+" already done: " + " ".join(config["species_list"][ds]))
            continue
    inputs = h5py.File(ds, 'r')
    species_list[ds] = []
    for entry in [x[0] for x in list(inputs.items())]:
        #print(entry)
        species = inputs[entry]["species"][()]
        species = species.astype("<U2")
        species_list[ds] = np.unique(np.hstack((species_list[ds], species)))
        #print(species_list)
    inputs.close()
    species_list[ds] = species_list[ds].tolist()
    species_list[ds].sort()
    config["species_list"] = species_list
    dump_json()


config["cmd"] = " ".join(sys.argv[1:])

           


pickled_training = f"{config['output']}/Training_{config['model_n']}.pkl"
pickled_testing = f"{config['output']}/Testing_{config['model_n']}.pkl"
pickled_SelfEnergies = f"{config['output']}/SelfEnergies_{config['model_n']}.pkl"

# =============================================================================
# Config the operation of a hard reset if needed
# =============================================================================
TrainingLog = config['output']+"/Training.log"

if config["hard_reset"] == True:
    Log.Log("Performing a HARD RESET")
    config["hard_reset"] = False
    if type(config["logfile"]) == str and os.path.exists(config["logfile"]):
        os.remove(config["logfile"])
    for file in [TrainingLog, f"{config['output']}/DNN_training.png",
                 f"{config['output']}/best.pt", f"{config['output']}/latest.pt", f"{config['output']}/Verbose.log",
                 pickled_training, pickled_testing, pickled_SelfEnergies]:
        if os.path.exists(file):
            Log.Log(f"Removing: {file}")
            os.remove(file)
    
os.environ["KMP_DUPLICATE_LIB_OK"] = str(config["KMP_DUPLICATE_LIB_OK"])
plt.ioff()

# =============================================================================
# Ensure learning rate and checkpoints reset are consistent with config
# =============================================================================
reset_lr = config["reset_lr"]
latest_checkpoint = config['output']+"/latest.pt"

Log.Log("reset_lr: "+str(reset_lr))
Log.Log("latest_checkpoint:"+latest_checkpoint)

# =============================================================================
# Set the device to cuda or to CPU
# =============================================================================

Log.Log("CUDA availibility: "+str(torch.cuda.is_available()))
if config["cuda"] == False or torch.cuda.is_available() == False:
    device = torch.device('cpu')   
    Log.Log("FORCING TO CPU")
else:
    device = torch.device('cuda')
Log.Log("Running on device: "+str(device))

# =============================================================================
# Now we want to initialise the variables for the aev equations (Eqn 3 from the
# ANI paper)
# =============================================================================
num_species = len(config["species_order"])
aev_computer = get_aev_comp(config["Rcr"], config["Rca"], num_species, device)
cuaev = False
Log.Log("cuaev: "+str(cuaev))
    
#aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
aev_dim = aev_computer.aev_length
energy_shifter = torchani.utils.EnergyShifter(None)


config["aev_dim"] = aev_dim+1 if config["mixed"] else aev_dim





starttime = int(time.time())

colourlist = iter(["blue", "orange", "purple", "red"])
colours = {}
datasets = {}
for ds in config["ds"]:
    colours[ds] = next(colourlist)
    datasets[ds] = {"training": None, "testing": None}
    datasets[ds]["training"], datasets[ds]["testing"] = torchani.data.load(ds)\
                                        .subtract_self_energies(energy_shifter, config["species_order"])\
                                        .species_to_indices(config["species_order"])\
                                        .shuffle()\
                                        .split(config["training_frac"], None)

    datasets[ds]["training"] = datasets[ds]["training"].collate(config["batch_size"]).cache()
    datasets[ds]["testing"]  = datasets[ds]["testing"].collate(config["batch_size"]).cache()
    
    ds_name = ds.replace("\\", "/").replace("/", "_")
    ###Write self energies
    if config["remove_self_energy"]:
        Log.Log('Self atomic energies: '+str(energy_shifter.self_energies))
        with open(os.path.join(config["output"], f"{ds_name}_Self_Energies.csv"), "w") as f:
            selfenergies_tensors = [t for t in energy_shifter.self_energies]
# =============================================================================
#             if len(selfenergies_tensors) != len(config["species_order"]):
#                 Log.Log("len(selfenergies_tensors) != len(config['species_order'])")
#                 Log.Log("Exiting...")
#                 sys.exit()
# =============================================================================
            senergies = [x.item() for x in selfenergies_tensors]
            se_dict = {}
            #f.write("Self Energies for dataset: %s \n" % ds)
            f.write("Atom,SelfEnergy\n")
            for key in species_list[ds]:
                for value in senergies:
                    se_dict[key] = value
                    senergies.remove(value)
                    break
            for k in se_dict.keys():
                f.write("%s, %s \n" % (k, se_dict[k]))
            f.close()
    else:
        SE = pandas.DataFrame(index=config["species_order"], columns=["SelfEnergy"])
        SE[:] = 0
        SE.to_csv(os.path.join(config["output"], "Self_Energies.csv"))

sys.exit()

config['Max'] = config['Min'] = None
dump_json()



networks = get_networks(config["aev_dim"], config["celu0"], config["celu1"], noutputs = len(datasets))
#networks = get_networks(config["aev_dim"], config["celu0"], config["celu1"], noutputs = 1)

NNs = []
for element in config["species_order"]:
    NNs.append(networks[f"{element}_network"])
nn = torchani.ANIModel(NNs, config['Min'], config['Max'], noutputs = len(datasets))




# Function to randomize weights
def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        #torch.nn.init.uniform_(m.bias)
        torch.nn.init.zeros_(m.bias)
# Apply the function to randomize weightd
nn.apply(init_params)


if config["preAEV"]:
    Log.Log("Setting up neural network for pre-cooked AEV dataset")
    model = nn.to(device)
else:
    model = torachani_nn.Sequential(aev_computer, nn).to(device)
    
# =============================================================================
# Now we set up the optimiser Adam with decoupled weight decay updates the
# weights and SGD updates the biases
# =============================================================================
AdamW_params = []
for network in NNs:
    AdamW_params.append({'params': [network[0].weight]})
    AdamW_params.append({'params': [network[2].weight], 'weight_decay': 0.00001})
    AdamW_params.append({'params': [network[4].weight], 'weight_decay': 0.000001})
    AdamW_params.append({'params': [network[6].weight]})


AdamW = torch.optim.AdamW(AdamW_params, lr = config["lr"])

SGD_params = []
for network in NNs:
    for i in range(len(network)):
        if hasattr(network[i], "bias"):
            SGD_params.append({'params': [network[i].bias]})
        
    
    
SGD = torch.optim.SGD(SGD_params, lr = config["lr"])

# =============================================================================
# Set up schedulers to do the updating of the learning rate for us 
# =============================================================================
AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0, verbose=False)
SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.5, patience=100, threshold=0, verbose=False)

# =============================================================================
# Sort out the resumption of training or reset the learning rate (in params)
# =============================================================================

if os.path.isfile(latest_checkpoint):
    if device.type == "cpu":
        checkpoint = torch.load(latest_checkpoint, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(latest_checkpoint)
    nn.load_state_dict(checkpoint['nn'])
    AdamW.load_state_dict(checkpoint['AdamW'])
    SGD.load_state_dict(checkpoint['SGD'])
    AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
    SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])
    L1_started = checkpoint["L1_started"]
else:
    L1_started = False



if reset_lr:
    Log.Log("Reset learning rates, you should only do this at the begining of a continuation!")
    for x in AdamW.param_groups:
        x["lr"] = config["lr"]
    for x in SGD.param_groups:
        x["lr"] = config["lr"]
    AdamW_scheduler._last_lr=[]
    SGD_scheduler._last_lr=[]
    AdamW_scheduler.best = 10000
    
# =============================================================================
# Now we can set up our testing loop
# =============================================================================

def test_sets():
    # run testing set
    if config["L1"] and L1_started:
        LOSSFN = torch.nn.L1Loss(reduction='sum')
    else:
        LOSSFN = torch.nn.MSELoss(reduction='sum')

    model.eval()
    test_energy_rmse = {}
    for i,key in enumerate(datasets.keys()):
        total_loss = 0.0
        count = 0
        for properties in datasets[key]["testing"]:
            species = properties['species'].to(device)
            coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
            true_energies = properties['energies'].to(device).float()
            
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            SpeciesEnergies = model((species, coordinates))
            _, predicted_energies = SpeciesEnergies[i]

            total_loss += LOSSFN(predicted_energies, true_energies).item()
            count += predicted_energies.shape[0]
            test_energy_rmse[key] = hartree2kcalmol(math.sqrt(total_loss / count))
    model.train()
    return test_energy_rmse



Log.Log("training starting from epoch " + str(AdamW_scheduler.last_epoch + 1))
best_model_checkpoint = config['output']+"/best.pt"



if os.path.exists(TrainingLog):
    training_log = pandas.read_csv(TrainingLog, index_col=0)
else:
    training_log = pandas.DataFrame()
    


# Nothing has crashed up to this point so write out to the config file that we
# don't want to restart from scratch again next time

Log.Log("L1_started: " + str(L1_started))
best_i = 0

dataset_i = 0

for _ in range(AdamW_scheduler.last_epoch + 1, config["epochs"]):
    dataset_key = list(datasets.keys())[dataset_i]
    energy_rmse = test_sets()
    EF_coef = sum(energy_rmse.values())
    print(energy_rmse)
    
    Epoch = AdamW_scheduler.last_epoch + 1
    
    if EF_coef < config["early_stop"]:
        print("rmse < early_stop, exiting...")
        break

    learning_rate = AdamW.param_groups[0]['lr']
    if learning_rate < config["early_stopping_learning_rate"]:
        if config["L1"] and not L1_started:
            L1_started = True
            config["early_stopping_learning_rate"] *= 0.5
            best_model_checkpoint = config['output']+"/best_L1.pt"
            best_L1_score = 1000.0
            Log.Log("learning_rate < early_stopping_learning_rate, switching to L1")
        else:
            Log.Log("learning_rate < early_stopping_learning_rate, exiting...")
            break

    # set a checkpoint
    if L1_started:
        if EF_coef <= best_L1_score:
            Log.Log("EF_coef <= best_L1_score")
            try:
                torch.save(nn.state_dict(), best_model_checkpoint)
            except PermissionError: # happens sometimes on windows for no good reason
                torch.save(nn.state_dict(), best_model_checkpoint)
            best_L1_score = EF_coef

    elif AdamW_scheduler.is_better(EF_coef, AdamW_scheduler.best) :
        try:
            torch.save(nn.state_dict(), best_model_checkpoint)
        except PermissionError: # happens sometimes on windows for no good reason
            torch.save(nn.state_dict(), best_model_checkpoint)
       
    
    if config["L1"] and L1_started:
        LOSS = torch.nn.L1Loss(reduction='none')
    else:
        LOSS = torch.nn.MSELoss(reduction='none')

    AdamW_scheduler.step(EF_coef)
    SGD_scheduler.step(EF_coef)
    
    Train_E_rmse = 0
    keys = list(datasets.keys())
    ds = 0
    iterable = [iter(datasets[keys[i]]["training"]) for i in range(len(datasets))]
    #for dataset_key in list(datasets.keys()):
    while 1:
        key = keys[ds]
        try:
            properties = next(iterable[ds])
        except StopIteration:
            del keys[ds]
            del iterable[ds]
            ds = 0
            if len(keys) == 0:
                break
            else:
                continue
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
        true_energies = properties['energies'].to(device).float()
        
        num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
        SpeciesEnergies = model((species, coordinates))
        predicted_energies = {}
        for i,key in enumerate(datasets.keys()):
            _, predicted_energies[key] = SpeciesEnergies[i]

        # Now the total loss has two parts, energy loss and force loss
        loss = (LOSS(predicted_energies[dataset_key], true_energies) / num_atoms.sqrt()).mean()

        Train_E_rmse += np.sqrt(loss.cpu().detach().numpy())

        AdamW.zero_grad()
        SGD.zero_grad()
        loss.backward()
        AdamW.step()
        SGD.step()

    Train_E_rmse = hartree2kcalmol(Train_E_rmse/len(datasets[dataset_key]["training"]))
    training_log.loc[Epoch, "lr"] = learning_rate
    
    # Training metrics (just the ds we have run)
    training_log.loc[Epoch, f"Train ({dataset_key})"] = Train_E_rmse
    # We should have the test data for every ds every time
    for key in list(datasets.keys()):
        training_log.loc[Epoch, f"Test ({key})"] = energy_rmse[key]
    
    
    
    Log.Log(f"Epoch: {Epoch} Train: {round(Train_E_rmse, 3)} Test: {round(energy_rmse[dataset_key], 3)} lr: {learning_rate}")

# =============================================================================
# Need to ensure everything saves nicely and then we can plot some pretty graphs
# =============================================================================

    try:
        torch.save({
            'nn': nn.state_dict(),
            'AdamW': AdamW.state_dict(),
            'SGD': SGD.state_dict(),
            'AdamW_scheduler': AdamW_scheduler.state_dict(),
            'SGD_scheduler': SGD_scheduler.state_dict(),
            "L1_started": L1_started,
        }, latest_checkpoint)
    except PermissionError: # happens sometimes on windows for no good reason
        Log.Log("Permission error in saving latest.pt, we'll just skip this one.")
    except OSError: # happens sometimes on windows for no good reason
        Log.Log("OSerror in saving latest.pt, we'll just skip this one.")
    
    training_log.to_csv(TrainingLog)
    dataset_i += 1
    if dataset_i >= len(datasets):
        dataset_i = 0
    Log.Log("dataset_i switched to: " + str(dataset_i))
    
    if Epoch > 0 and os.name == "nt":
        for col in training_log.columns:
            if col in ["Epoch", "lr"]:
                continue
            ds = col.split("(")[1].split(")")[0]
            colour = colours[ds]
            if "Test" in col:
                plt.plot(training_log[col].dropna(), "--", label=col, c=colour)
            if "Train" in col:
                plt.plot(training_log[col].dropna(), label=col, c=colour)
                plt.scatter(training_log[col].dropna().index, training_log[col].dropna(), marker=5, c=colour) 
        plt.legend()
        plt.show()


t = time.localtime()
Log.Log("Finished training in: " + str(starttime - time.time()) + "s")

# =============================================================================
# Finish the Training and save evrything to the log file
# =============================================================================

Log.close()
