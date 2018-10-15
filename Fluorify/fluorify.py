#!/usr/bin/env python

from .energy import FSim
from .mol2 import Mol2, MutatedLigand

import os
import time
import mdtraj as md

class FluorineScanning(object):
    def __init__(self, input_folder, output_folder, mol_file, ligand_name,
                 complex_name, solvent_name, job_type):

        mol_file = mol_file + '.mol2'
        complex_sim_dir = input_folder + complex_name + '/'
        solvent_sim_dir = input_folder + solvent_name + '/'

        try:
            os.makedirs(output_folder)
        except:
            raise ValueError('Could not create output folder {}'.format(output_folder))

        mol = Mol2()
        try:
            Mol2.get_data(mol, input_folder, mol_file)
        except:
            raise ValueError('Could not load molecule {}'.format(input_folder + mol_file))

        new_element = job_type[0]
        carbon_type = 'C.' + job_type[1]

        carbons = Mol2.get_atom_by_string(mol, carbon_type)
        hydrogens = Mol2.get_atom_by_string(mol, 'H')
        bonded_h = Mol2.get_bonded_hydrogens(mol, hydrogens, carbons)
        mutated_systems = Mol2.mutate_elements(mol, bonded_h, new_element)

        mutated_ligands = []
        print('Generating Mol2s...')
        t0 = time.time()
        for index, sys in enumerate(mutated_systems):
            Mol2.write_mol2(sys, output_folder, 'molecule'+str(index))
            mutated_ligands.append(MutatedLigand(file_path=output_folder, file_name='{}.mol2'.format('molecule'+str(index))))
        t1 = time.time()
        print('Took {} seconds'.format(t1 - t0))

        ligand_charges = []
        for ligand in mutated_ligands:
            ligand_charges.append(ligand.get_charge())


        print('Calculating Energies...')
        t0 = time.time()

        #COMPLEX
        complex_traj = md.load(complex_sim_dir + complex_name + '.pdb') #PDB FOR TESTING
        complex = FSim(ligand_name=ligand_name, sim_name=complex_name, input_folder=input_folder)
        complex_free_energy = FSim.treat_phase(complex, ligand_charges, complex_traj)
        print(complex_free_energy)

        #SOLVENT
        #solvent_traj = md.load(solvent_sim_dir + solvent_name + '.pdb') #PDB FOR TESTING
        #solvent = FSim(ligand_name=ligand_name, sim_name=solvent_name, input_folder=input_folder)
        #solvent_free_energy = FSim.treat_phase(solvent, ligand_charges, complex_traj)

        t1 = time.time()
        print('Took {} seconds'.format(t1 - t0))

        # How can this be called many times for optimiztion
        # Check free energy calculation