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

        if os.path.isdir(output_folder):
            print('Output folder {} already exists.'
                  ' Will attempt to skip ligand parametrisation, proceed with caution...'.format(output_folder))
        else:
            try:
                os.makedirs(output_folder)
            except:
                print('Could not create output folder {}'.format(output_folder))

        mol = Mol2()
        try:
            Mol2.get_data(mol, input_folder, mol_file)
        except:
            raise ValueError('Could not load molecule {}'.format(input_folder + mol_file))

        #Check ligand atom order is consistent across input topologies.
        mol2_ligand_atoms = get_atom_list(input_folder+mol_file, ligand_name)
        complex_ligand_atoms = get_atom_list(complex_sim_dir+complex_name+'.pdb', ligand_name)
        solvent_ligand_atoms = get_atom_list(solvent_sim_dir+solvent_name+'.pdb', ligand_name)
        if mol2_ligand_atoms != complex_ligand_atoms:
            raise ValueError('Topology of ligand not matched across input files.'
                             'Charges will not be applied where expected')
        if complex_ligand_atoms != solvent_ligand_atoms:
            raise ValueError('Topology of ligand not matched across input files.'
                             'Charges will not be applied where expected')

        #Generate atom selections
        new_element = job_type[0]
        carbon_type = 'C.' + job_type[1]
        carbons = Mol2.get_atom_by_string(mol, carbon_type)
        hydrogens = Mol2.get_atom_by_string(mol, 'H')
        carbons_neighbours = Mol2.get_bonded_neighbours(mol, carbons)
        bonded_h = hydrogens.intersection(carbons_neighbours)

        """
        Write Mol2 files with substitutions of selected atoms.
        Run antechamber and tleap on mol2 files to get prmtop.
        Create OpenMM systems of ligands from prmtop files.
        """
        print('Parametrize mutant ligands...')
        t0 = time.time()
        mutated_ligands = []
        mutated_systems = Mol2.mutate_elements(mol, bonded_h, new_element)
        for index, sys in enumerate(mutated_systems):
            Mol2.write_mol2(sys, output_folder, 'molecule'+str(index))
            mutated_ligands.append(MutatedLigand(file_path=output_folder,
                                                 file_name='{}.mol2'.format('molecule'+str(index))))
        t1 = time.time()
        print('Took {} seconds'.format(t1 - t0))

        """
        Extract ligand charges from OpenMM systems.
        Build OpenMM simulations for complex and solvent phases.
        """
        print('Setup simulations...')
        t0 = time.time()

        ligand_charges = []
        for ligand in mutated_ligands:
            ligand_charges.append(ligand.get_charge())

        #COMPLEX
        complex = FSim(ligand_name=ligand_name, sim_name=complex_name, input_folder=input_folder)
        complex_traj = md.load(complex_sim_dir+complex_name+'.pdb', complex_sim_dir+complex_name+'.dcd')

        #SOLVENT
        solvent = FSim(ligand_name=ligand_name, sim_name=solvent_name, input_folder=input_folder)
        solvent_traj = md.load(solvent_sim_dir+solvent_name+'.pdb', solvent_sim_dir+solvent_name+'.dcd')

        t1 = time.time()
        print('Took {} seconds'.format(t1 - t0))

        """
        Apply ligand charges to OpenMM simulations.
        Calculate potential energy of simulation with mutant charges.
        Calculate free energy change from wild type to mutant.
        """
        print('Calculating Energies...')
        t0 = time.time()

        complex_free_energy = FSim.treat_phase(complex, ligand_charges, complex_traj)
        solvent_free_energy = FSim.treat_phase(solvent, ligand_charges, solvent_traj)

        for i, energy in enumerate(complex_free_energy):
            atom_index = int(list(bonded_h)[i])-1
            print('dG for molecule{}.mol2 with'
                  ' {} substituted for {}'.format(str(i), mol2_ligand_atoms[atom_index], job_type[0]))
            print(energy - solvent_free_energy[i])

        t1 = time.time()
        print('Took {} seconds'.format(t1 - t0))

        # How can this be called many times for optimiztion?


def get_atom_list(file, resname):
    atoms = []
    traj = md.load(file)
    top = traj.topology
    mol = top.select('resname '+resname)
    for idx in mol:
        atoms.append(str(top.atom(idx)).split('-')[1])
    return atoms
