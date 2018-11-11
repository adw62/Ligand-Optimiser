#!/usr/bin/env python

from .energy import FSim
from .mol2 import Mol2, MutatedLigand

import os
import time
import itertools
import mdtraj as md

class Scanning(object):
    def __init__(self, output_folder, mol_file, ligand_name, net_charge, complex_name,
                 solvent_name, job_type, atom_list, num_frames, charge_only):
        """A class for preparation and running scanning analysis

        mol_file: name of input ligand file
        ligand_name: resname for input ligand
        net_charge: charge on ligand
        complex_name: name of complex phase directory inside of input directory. Default, complex
        solvent_name: name of solvent phase directory inside of output directory.  Default solvent
        job_type: name of scanning requested i.e. F_ar replaces hydrogen on aromatic 'ar' carbons with fluorine 'F'
        atom_list: list of atoms to replace overrides selection from jobtype
        output_folder: name for output directory. Default, mol_file + job_type
        """

        if charge_only == True:
            print('Mutating ligand charges only...')
            if job_type[0] == 'N.ar':
                print('Van der Walls must be set for pyridine scanning')
                print('Ignoring charge only option')
        else:
            print('Mutating all ligand parameters...')

        self.net_charge = net_charge
        #Prepare directories/files and read in ligand from mol2 file
        mol_file = mol_file + '.mol2'
        input_folder = './input/'

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

        self.mol = Mol2()
        try:
            Mol2.get_data(self.mol, input_folder, mol_file)
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
        atoms = atom_selection(atom_list)
        if job_type[0] == 'N.ar':
            mutated_systems, target_atoms, atoms_to_mute = add_nitrogens(self.mol, job_type, atoms)
        else:
            mutated_systems, target_atoms = add_fluorines(self.mol, job_type, atoms)
            atoms_to_mute = None

        """
        Write Mol2 files with substitutions of selected atoms.
        Run antechamber and tleap on mol2 files to get prmtop.
        Create OpenMM systems of ligands from prmtop files.
        Extract ligand charges from OpenMM systems.
        """
        print('Parametrize mutant ligands...')
        t0 = time.time()

        mutated_ligands = []
        for index, sys in enumerate(mutated_systems):
            mol_name = 'molecule'+str(index)
            Mol2.write_mol2(sys, output_folder, mol_name)
            mutated_ligands.append(MutatedLigand(file_path=output_folder, mol_name=mol_name, net_charge=net_charge))

        ligand_parameters = []
        for i, ligand in enumerate(mutated_ligands):
            ligand_parameters.append(ligand.get_parameters(atoms_to_mute[i]))

        t1 = time.time()
        print('Took {} seconds'.format(t1 - t0))

        """
        Build OpenMM simulations for complex and solvent phases.
        Apply ligand charges to OpenMM simulations.
        Calculate potential energy of simulation with mutant charges.
        Calculate free energy change from wild type to mutant.
        """
        print('Calculating free energies...')
        t0 = time.time()

        #COMPLEX
        complex = FSim(ligand_name=ligand_name, sim_name=complex_name,
                       input_folder=input_folder, charge_only=charge_only)
        com_top = md.load(complex_sim_dir+complex_name+'.pdb').topology
        com_dcd = complex_sim_dir + complex_name + '.dcd'
        #SOLVENT
        solvent = FSim(ligand_name=ligand_name, sim_name=solvent_name,
                       input_folder=input_folder, charge_only=charge_only)
        sol_top = md.load(solvent_sim_dir+solvent_name+'.pdb').topology
        sol_dcd = solvent_sim_dir + solvent_name + '.dcd'

        print('Computing complex potential energies...')
        complex_free_energy = FSim.treat_phase(complex, ligand_parameters, com_dcd, com_top, num_frames)
        print('Computing solvent potential energies...')
        solvent_free_energy = FSim.treat_phase(solvent, ligand_parameters, sol_dcd, sol_top, num_frames)

        for i, energy in enumerate(complex_free_energy):
            atom_names = []
            atoms = target_atoms[i]
            for atom in atoms:
                atom_index = int(atom)-1
                atom_names.append(mol2_ligand_atoms[atom_index])

            print('dG for molecule{}.mol2 with'
                  ' {} substituted for {}'.format(str(i), atom_names, job_type[0]))
            print(energy - solvent_free_energy[i])

        t1 = time.time()
        print('Took {} seconds'.format(t1 - t0))

def atom_selection(atom_list):
    #generate all permutations
    atom_list = list(itertools.product(*atom_list))
    if len(atom_list) > 0:
        #remove permutations with duplicates
        tmp = []
        for x in atom_list:
            if len(x) == len(set(x)):
                tmp.append(x)
        atom_list = tmp
        atom_list = [list(sorted(q)) for q in atom_list]
        #remove equivilant permutations
        tmp = []
        for x in atom_list:
            if x not in tmp:
                tmp.append(x)
        atom_list = tmp
    return atom_list


def add_fluorines(mol, job_type, atom_list):
    new_element = job_type[0]
    if len(atom_list[0]) == 0:
        carbon_type = 'C.' + job_type[1]
        carbons = Mol2.get_atom_by_string(mol, carbon_type)
        hydrogens = Mol2.get_atom_by_string(mol, 'H')
        carbons_neighbours = Mol2.get_bonded_neighbours(mol, carbons)
        bonded_h = [x for x in hydrogens if x in carbons_neighbours]
        bonded_h = [bonded_h[i:i+1] for i in range(0, len(bonded_h), 1)]
    else:
        hydrogens = Mol2.get_atom_by_string(mol, 'H')
        for pair in atom_list:
            tmp = [x for x in pair if x not in hydrogens]
            if len(tmp) > 0:
                raise ValueError('Atoms {} are not recognised as hydrogens and therefore can not be fluorinated'.format(tmp))
        bonded_h = atom_list
    mutated_systems = Mol2.mutate(mol, bonded_h, new_element)
    return mutated_systems, bonded_h


def add_nitrogens(mol, job_type, atom_list):
    """
    Look for carbons with one hydrogen neighbour.
    Reduce list of carbons to those with one hydrogen neighbour.
    Make a note of which hydrogen is the neighbour.
    Swap the carbon for a nitrogen and hydrogen for dummy.
    This should be making Pyridine.
    """
    new_element = job_type[0]
    if len(atom_list[0]) == 0:
        carbon_type = 'C.' + job_type[1]
        carbons = Mol2.get_atom_by_string(mol, carbon_type)
        hydrogens = Mol2.get_atom_by_string(mol, 'H')
        hydrogens_to_remove = []
        c_tmp = []
        for atom in carbons:
            h_tmp = []
            neighbours = Mol2.get_bonded_neighbours(mol, atom)
            for neighbour in neighbours:
                if neighbour in hydrogens:
                    h_tmp.append(neighbour)
            if len(h_tmp) == 1:
                #need to index from 0 to interface with openmm
                hydrogens_to_remove.append([int(x)-1 for x in h_tmp])
                c_tmp.append([atom])
        carbons = c_tmp
    else:
        #TODO
        pass

    mutated_systems = Mol2.mutate(mol, carbons, new_element)
    for i, mutant in enumerate(mutated_systems):
        mutant.remove_atom(hydrogens_to_remove[i][0])
    return mutated_systems, carbons, hydrogens_to_remove

def get_atom_list(file, resname):
    atoms = []
    traj = md.load(file)
    top = traj.topology
    mol = top.select('resname '+resname)
    for idx in mol:
        atoms.append(str(top.atom(idx)).split('-')[1])
    return atoms

