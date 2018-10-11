#!/usr/bin/env python

from Energy import FSim
from Mol2 import Mol2, MutatedLigand

import time

class FlorineScanning(object):

    t4 = Mol2()
    Mol2.get_data(t4, '../input/', 'ligand.tripos.mol2')
    ar_c = Mol2.get_atom_by_string(t4, 'C.ar')
    hydrogens = Mol2.get_atom_by_string(t4, 'H')

    bonded_h = Mol2.get_bonded_hydrogens(t4, hydrogens, ar_c)

    mutated_systems = Mol2.mutate_elements(t4, bonded_h, new_element='Cl')
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
    print(FSim.get_mutant_energy(F, ligand_charges, F.snapshot))
    t1 = time.time()
    print('Took {} seconds'.format(t1 - t0))
