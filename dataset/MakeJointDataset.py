# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:57:58 2024

@author: Alex
"""

import os, sys, pandas, shutil, datetime, tqdm, pickle
import h5py
import scotch_db
import orca_parser
import json, glob
from json import JSONEncoder
import numpy as np
import MySQLdb
from ase import Atoms

### We could not recreate the conformers table on the new server
### So we recreate the raw table and rebuild from that

if os.path.exists("SVP.xyz"):
    os.remove("SVP.xyz")
if os.path.exists("TZVPP.xyz"):
    os.remove("TZVPP.xyz")
    
    
n_samples = 10000

database_name="dft2"
host="130.159.56.71"
user="scotch_db_user"
passwd=os.environ["FTP_PASSWORD"]

db = MySQLdb.connect(host=host,    # your host, usually localhost
                     user=user,         # your username
                     passwd=passwd,  # your password
                     db=database_name)        # name of the data base

# you must create a Cursor object. It will let
#  you execute all the queries you need
cur = db.cursor()
dft_outputs_checker = db.cursor()
cur.execute(f"""SELECT `p`.`e`, `conformers`.`coordinates`, `conformers`.`species`, `dft_outputs`.`charge`,
`dft_outputs`.`solvation`
FROM `energy` `p`, `conformers` `conformers`, `dft_outputs` `dft_outputs`
WHERE `p`.`e` < 0
  AND `conformers`.`no` < 4
  AND `conformers`.`nc` > 0
  AND `conformers`.`nal` = 0
  AND `conformers`.`nb` = 0
  AND `conformers`.`np` = 0
  AND `conformers`.`nf` = 0
  AND `conformers`.`nir` = 0
  AND `conformers`.`ncl` = 0
  AND `dft_outputs`.`functional` = 'WB97X'
  AND `dft_outputs`.`dispersion` = 'D4'
  AND `dft_outputs`.`def2j` = 1
AND `dft_outputs`.`solvation` = 'Gas'
  AND `dft_outputs`.`basisset` = 'def2-SVP'
  AND `dft_outputs`.`charge` = 0
  AND `p`.`conf_id` = `conformers`.`id`
  AND `p`.`dft_output` = `dft_outputs`.`id`
LIMIT {n_samples}""")
data = cur.fetchall()
print(len(data))

species_order = []
svp = h5py.File("SVP.h5", 'w')
mol, E, C, S, F = [],[],[],[],[]
for i,row in enumerate(data):
    mol.append(svp.create_group(str(i)))
    #print(row)
    energy = np.array([row[0]])
    coords = np.frombuffer(row[1]).reshape(1, -1,3)
    
    species = np.array(row[2].split(), dtype="<U2")
    species = np.array(species, dtype = h5py.special_dtype(vlen=str) )
    
    E.append(mol[-1].create_dataset("energies", (energy.shape[0],), dtype='float64'))
    E[-1][()] = energy    
    
    C.append(mol[-1].create_dataset("coordinates", coords.shape, dtype='float64'))
    C[-1][()] = coords    
    species_order = np.hstack((species_order, species))
    S.append(mol[-1].create_dataset("species", data=species))
    asemol = Atoms(species, np.frombuffer(row[1]).reshape(-1,3))
    asemol.write("SVP.xyz", append=True)    
svp.close()
print("SVP:", np.unique(species_order))


cur.execute(f"""SELECT `p`.`e`, `conformers`.`coordinates`, `conformers`.`species`, `dft_outputs`.*,
`dft_outputs`.`solvation`, `dft_outputs`.`basisset`, `dft_outputs`.`functional`                                       
FROM `energy` `p`, `conformers` `conformers`, `dft_outputs` `dft_outputs`
WHERE `p`.`e` < 0
  AND `conformers`.`no` < 4
  AND `conformers`.`nc` > 0
  AND `conformers`.`nal` = 0
  AND `conformers`.`nb` = 0
  AND `conformers`.`np` = 0
  AND `conformers`.`nf` = 0
  AND `conformers`.`nna` = 0
  AND `conformers`.`nir` = 0
  AND `conformers`.`ncl` = 0
  AND `dft_outputs`.`def2j` = 1
  AND `dft_outputs`.`functional` = 'WB97X'
  AND `dft_outputs`.`solvation` = 'Gas'
  AND `dft_outputs`.`basisset` = 'def2-TZVPP'
  AND `dft_outputs`.`charge` = 0 
  AND `p`.`conf_id` = `conformers`.`id`
  AND `p`.`dft_output` = `dft_outputs`.`id`
LIMIT {n_samples}""")

data = cur.fetchall()
print(len(data))

species_order = []
tzvpp = h5py.File("TZVPP.h5", 'w')
mol, E, C, S, F = [],[],[],[],[]
for i,row in enumerate(data):
    mol.append(tzvpp.create_group(str(i)))
    #print(row)
    energy = np.array([row[0]])
    coords = np.frombuffer(row[1]).reshape(1, -1,3)
    
    species = np.array(row[2].split(), dtype="<U2")
    species = np.array(species, dtype = h5py.special_dtype(vlen=str) )
    
    E.append(mol[-1].create_dataset("energies", (energy.shape[0],), dtype='float64'))
    E[-1][()] = energy    
    
    C.append(mol[-1].create_dataset("coordinates", coords.shape, dtype='float64'))
    C[-1][()] = coords    
    
    #print(species)
    species_order = np.hstack((species_order, species))
    S.append(mol[-1].create_dataset("species", data=species))
    asemol = Atoms(species, np.frombuffer(row[1]).reshape(-1,3))
    asemol.write("TZVPP.xyz", append=True)  
tzvpp.close()
print("TZVPP:", np.unique(species_order))





