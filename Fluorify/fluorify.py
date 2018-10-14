#!/usr/bin/env python

from energy import FSim
from mol2 import Mol2, MutatedLigand

import time

import numpy as np


class FlorineScanning(object):

    #have some interface for running jobs ie run_florine run_clorine
    def run(self):
        pass

    def free_energy(self, mutant_energy, wildtype_energies):

        kB = 0.008314472471220214
        T = 300
        kT = kB * T
        ans=[]
        for ligand in mutant_energy:
            energies = 0.0
            for i in range(len(wildtype_energies)):
                energies += (np.exp(-(ligand[i] - wildtype_energies[i])/kT))
            ans.append(energies/len(wildtype_energies))
        final = 0.0
        H_idx = 0
        for i, energy in enumerate(ans):
            tmp = -kB * T * np.log(energy) / 1000  # not sure of unit CHECK!!!
            if tmp <= final:
                final = tmp
                H_idx = i
        return final, H_idx

    t4 = Mol2()
    Mol2.get_data(t4, '../input/', 'ligand.tripos.mol2')
    ar_c = Mol2.get_atom_by_string(t4, 'C.ar')
    hydrogens = Mol2.get_atom_by_string(t4, 'H')

    bonded_h = Mol2.get_bonded_hydrogens(t4, hydrogens, ar_c)

    mutated_systems = Mol2.mutate_elements(t4, bonded_h, new_element='F')
    names = ['a', 'b', 'c', 'd', 'e', 'f']  # How to name the ref later**
    mutated_ligands = []


    print('Generating Mol2s...')
    t0 = time.time()
    for index, sys in enumerate(mutated_systems):
        Mol2.write_mol2(sys, '../', names[index])
        mutated_ligands.append(MutatedLigand(file_path='../', file_name='{}.mol2'.format(names[index])))
    t1 = time.time()
    print('Took {} seconds'.format(t1-t0))

    ligand_charges = []
    for ligand in mutated_ligands:
        ligand_charges.append(ligand.get_charge())


    print('Calculating Energies...')
    t0 = time.time()
    F = FSim(ligand_name='MOL')

    #load traj to apply charges to and calculate energies from
    traj = []
    for i in range(1):
        traj.append(F.snapshot)


    wildtype_energy = FSim.get_wildtype_energy(F, traj)
    mutant_energy = FSim.get_mutant_energy(F, ligand_charges, traj)


    print(free_energy(F, mutant_energy, wildtype_energy))

    #biggest energy differnce will be biggest free energy differnce
    for ligand in mutant_energy:
        print(ligand[0] - wildtype_energy[0])

    t1 = time.time()
    print('Took {} seconds'.format(t1 - t0))

    #better file pointing
    #chosse directory structure and point to approriate places

    #calcualte free energy
    #find lowest free energy
    #how can this be called many times for optimiztion?