#!/usr/bin/env python

from .energy import FSim
from .mol2 import Mol2, MutatedLigand

import os
import time
import itertools
import mdtraj as md

class Scanning(object):
    def __init__(self, output_folder, mol_file, ligand_name, net_charge, complex_name,
                 solvent_name, job_type, auto_select, c_atom_list, h_atom_list, num_frames, charge_only):
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

        self.net_charge = net_charge
        self.output_folder = output_folder
        #Prepare directories/files and read in ligand from mol2 file
        mol_file = mol_file + '.mol2'
        input_folder = './input/'

        complex_sim_dir = input_folder + complex_name + '/'
        solvent_sim_dir = input_folder + solvent_name + '/'

        if os.path.isdir(self.output_folder):
            print('Output folder {} already exists.'
                  ' Will attempt to skip ligand parametrisation, proceed with caution...'.format(self.output_folder))
        else:
            try:
                os.makedirs(self.output_folder)
            except:
                print('Could not create output folder {}'.format(self.output_folder))

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

        #Generate mutant systems with selected pertibations
        mutated_systems, mutations = Scanning.perturbation(self, job_type, auto_select, h_atom_list, c_atom_list)

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
            Mol2.write_mol2(sys, self.output_folder, mol_name)
            mutated_ligands.append(MutatedLigand(file_path=self.output_folder, mol_name=mol_name, net_charge=net_charge))

        ligand_parameters = []
        for i, ligand in enumerate(mutated_ligands):
            mute = mutations[i]['subtract']
            ligand_parameters.append(ligand.get_parameters(mute))

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

        #RESULT
        for i, energy in enumerate(complex_free_energy):
            atom_names = []
            replace = mutations[i]['replace']
            for atom in replace:
                atom_index = int(atom)-1
                atom_names.append(mol2_ligand_atoms[atom_index])

            print('ddG for molecule{}.mol2 with'
                  ' {} substituted for {}'.format(str(i), atom_names, job_type[0]))
            binding_free_energy = energy - solvent_free_energy[i]
            print(binding_free_energy)

        t1 = time.time()
        print('Took {} seconds'.format(t1 - t0))

    def perturbation(self, job_type, auto_select, h_atom_list, c_atom_list):
        """
        Takes a job_type with an auto selection or atom_lists and turns this into the correct mutated_systems
        and a list of mutations to keep track of the mutation applied to each system.
        """
        job_type = job_type.split('x')
        c_atoms = atom_selection(c_atom_list)
        h_atoms = atom_selection(h_atom_list)
        if len(job_type) == 1:
            if job_type == ['N']:
                job_type = 'N.ar'
                if h_atoms is not None:
                    raise ValueError('hydrogen atom list provided but pyridine scanning job requested')
                mutated_systems, mutations = add_nitrogens(self.mol, job_type, auto_select, c_atoms)
            else:
                job_type = job_type[0]
                if c_atoms is not None:
                    raise ValueError('carbon atom list provided but fluorine scanning job requested')
                mutated_systems, mutations = add_fluorines(self.mol, job_type, auto_select, h_atoms)
        else:
            if h_atom_list is None or c_atoms is None:
                raise ValueError('Mixed substitution requested must provided carbon and hydrogen atom lists')
            job_type[0] = 'N.ar'
            mutated_systems = []
            mutations = []
            f_mutated_systems, f_mutations = add_fluorines(self.mol, job_type[1], auto_select, h_atoms)
            for i, mol in enumerate(f_mutated_systems):
                p_mutated_systems, p_mutations = add_nitrogens(mol, job_type[0], auto_select,
                                                                                   c_atoms, modified_atom_type=job_type[1])
                #compound flurination and pyridination
                for j in range(len(p_mutations)):
                    p_mutations[j]['add'].extend(f_mutations[i]['add'])
                    p_mutations[j]['subtract'].extend(f_mutations[i]['subtract'])
                    p_mutations[j]['replace'].extend(f_mutations[i]['replace'])
                mutated_systems.extend(p_mutated_systems)
                mutations.extend(p_mutations)
        return mutated_systems, mutations


def atom_selection(atom_list):

    if atom_list is None:
        return None

    #generate all permutations
    atom_list = list(itertools.product(*atom_list))
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


def add_fluorines(mol, new_element, auto_select, atom_list):
    mutations = []
    if atom_list is None:
        carbon_type = 'C.' + auto_select
        carbons = Mol2.get_atom_by_string(mol, carbon_type)
        carbons_neighbours = []
        for atom in carbons:
            carbons_neighbours.extend(Mol2.get_bonded_neighbours(mol, atom))
        hydrogens = Mol2.get_atom_by_string(mol, 'H')
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

    #Build list of dictionaries each dict describes mutation applied corresponding system
    for i, mutant in enumerate(mutated_systems):
        mutations.append({'add': [], 'subtract': [], 'replace': []})
        for atom in bonded_h[i]:
            mutations[i]['replace'].append(int(atom))
    return mutated_systems, mutations


def add_nitrogens(mol, new_element, auto_select, atom_list, modified_atom_type=None):
    """
    Look for carbons with one hydrogen neighbour.
    Reduce list of carbons to those with one hydrogen neighbour.
    Make a note of which hydrogen is the neighbour.
    Swap the carbon for a nitrogen and label hydrogen to be muted
    or use user provided list of carbons
    """
    mutations = []
    if atom_list is None:
        carbon_type = 'C.' + auto_select
        carbons = Mol2.get_atom_by_string(mol, carbon_type)
        hydrogens = Mol2.get_atom_by_string(mol, 'H')
        hydrogens_to_remove = []
        c_tmp = []
        for atom in carbons:
            h_neigh = []
            neighbours = Mol2.get_bonded_neighbours(mol, atom)
            for neighbour in neighbours:
                if neighbour in hydrogens:
                    h_neigh.append(neighbour)
            if len(h_neigh) == 1:
                #need to index from 0 to interface with openmm
                hydrogens_to_remove.append([int(x)-1 for x in h_neigh])
                c_tmp.append([atom])
        carbons = c_tmp
    else:
        carbons = Mol2.get_atom_by_string(mol, 'C.', wild_card=True)
        hydrogens = Mol2.get_atom_by_string(mol, 'H')
        hydrogens_to_remove = []
        #Find any atoms which may have already been fluorinated
        if modified_atom_type is not None:
            modified_atoms = Mol2.get_atom_by_string(mol, modified_atom_type)
        else:
            modified_atoms = []

        #check atom list contains only carbon
        for pair in atom_list:
            tmp = [x for x in pair if x not in carbons]
            if len(tmp) > 0:
                raise ValueError('Atoms {} are not recognised as carbons and therefore can not be pyridinated'.format(tmp))
        carbons = atom_list

        #check if carbons in atom list have one neighbour
        for pair in carbons:
            h_tmp = []
            for atom in pair:
                h_neigh = []
                neighbours = Mol2.get_bonded_neighbours(mol, atom)
                for neighbour in neighbours:
                    if neighbour in hydrogens:
                        h_neigh.append(neighbour)
                    #catches any atoms which may have previously been hydrogens but have already been flourinated
                    if neighbour in modified_atoms:
                        h_neigh.append(neighbour)
                if len(h_neigh) != 1:
                    raise ValueError('Atom {} does not have only one neighbouring hydrogen therefore can not be pyridinated'.format(atom))
                else:
                    # need to index from 0 to interface with openmm
                    h_tmp.extend(int(x) - 1 for x in h_neigh)
            h_tmp.sort(reverse=True)
            hydrogens_to_remove.append(h_tmp)

    #Mutate carbons to nitrogen and remove hydrogens
    mutated_systems = Mol2.mutate(mol, carbons, new_element)
    for i, mutant in enumerate(mutated_systems):
        for atom in hydrogens_to_remove[i]:
            mutant.remove_atom(atom)

    # Build list of dictionaries each dict describes mutation applied corresponding system
    for i, mutant in enumerate(mutated_systems):
        mutations.append({'add': [], 'subtract': [], 'replace': []})
        for atom in carbons[i]:
            mutations[i]['replace'].append(int(atom))
        for atom in hydrogens_to_remove[i]:
            mutations[i]['subtract'].append(int(atom))
    return mutated_systems, mutations

def get_atom_list(file, resname):
    atoms = []
    traj = md.load(file)
    top = traj.topology
    mol = top.select('resname '+resname)
    for idx in mol:
        atoms.append(str(top.atom(idx)).split('-')[1])
    return atoms

