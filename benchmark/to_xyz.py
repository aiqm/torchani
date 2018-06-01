from selected_system import mols, mol_file
import h5py
import os

fm = h5py.File(os.path.join(mol_file), "r")

for i in mols:
    print('number of atoms:', i)
    smiles = mols[i]
    for s in smiles:
        key = s.replace('/', '_')
        filename = i
        with open('benchmark_xyz/' + filename + '.xyz', 'w') as fxyz:
            coordinates = fm[key][()]
            species = fm[key].attrs['species'].split()
            conformations = coordinates.shape[0]
            atoms = len(species)
            for i in range(conformations):
                fxyz.write('{}\n{}\n'.format(atoms, 'smiles:{}\tconformation:{}'.format(s,i)))
                for j in range(atoms):
                    ss = species[j]
                    xyz = coordinates[i,j,:]
                    x = xyz[0]
                    y = xyz[1]
                    z = xyz[2]
                    fxyz.write('{} {} {} {}\n'.format(ss,x,y,z))
            break