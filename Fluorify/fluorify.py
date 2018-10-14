#!/usr/bin/env python

from .energy import FSim
from .mol2 import Mol2, MutatedLigand

import time

import numpy as np

class FluorineScanning(object):
    def __init__(self, input_folder=None, output_folder=None, mol_file=None, complex_file=None, job_type=None):
        msg = 'No {0} specified using default {1}'

        if input_folder == None:
            self.input = './input/'
            print(msg.format('input', self.input))
        else:
            self.input = input_folder
        if output_folder == None:
            self.output = './output/'
            print(msg.format('output', self.output))
        else:
            self.output = output_folder
        if mol_file == None:
            self.mol_file = 'ligand.mol2'
            print(msg.format('mol2 file', self.mol_file))
        else:
            self.mol_file = mol_file + 'mol2'
        if complex_file == None:
            self.complex_file = 'complex'
            print(msg.format('complex file names', self.complex_file))
        else:
            self.complex_file = complex_file
        if job_type == None:
            self.job_type = ['F', 'ar']
            print(msg.format('job type', self.job_type))
        else:
            allowed_elements = ['F', 'Cl']
            allowed_carbon_types = ['1', '2', '3', 'ar']
            job_type = job_type.split('_')
            if job_type[0] not in allowed_elements:
                raise ValueError('Allowed elements {}'.format(allowed_elements))
            elif job_type[1] not in allowed_carbon_types:
                raise ValueError('Allowed carbon types {}'.format(allowed_carbon_types))
            else:
                self.job_type = job_type

        #NEED TRAJ_FILE options
        #need split COMPLEX INTO .prmtop and .pdb

        FluorineScanning.run(self)

    def free_energy(mutant_energy, wildtype_energies):

        kB = 0.008314472471220214
        T = 300
        kT = kB * T
        ans = []
        for ligand in mutant_energy:
            energies = 0.0
            for i in range(len(wildtype_energies)):
                energies += (np.exp(-(ligand[i] - wildtype_energies[i]) / kT))
            ans.append(energies / len(wildtype_energies))
        final = 0.0
        H_idx = 0
        for i, energy in enumerate(ans):
            tmp = -kB * T * np.log(energy) / 1000  # not sure of unit CHECK!!!
            if tmp <= final:
                final = tmp
                H_idx = i
            return final, H_idx

    def run(self):
        mol = Mol2()
        try:
            Mol2.get_data(mol, self.input, self.mol_file)
        except:
            raise ValueError('Could not load molecule {}'.format(self.input + self.mol_file))

        new_element = self.job_type[0]
        carbon_type = 'C.' + self.job_type[1]

        carbons = Mol2.get_atom_by_string(mol, carbon_type)
        hydrogens = Mol2.get_atom_by_string(mol, 'H')
        bonded_h = Mol2.get_bonded_hydrogens(mol, hydrogens, carbons)
        mutated_systems = Mol2.mutate_elements(mol, bonded_h, new_element)

        mutated_ligands = []
        print('Generating Mol2s...')
        t0 = time.time()
        for index, sys in enumerate(mutated_systems):
            Mol2.write_mol2(sys, self.output, 'molecule'+str(index))
            mutated_ligands.append(MutatedLigand(file_path=self.output, file_name='{}.mol2'.format('molecule'+str(index))))
        t1 = time.time()
        print('Took {} seconds'.format(t1 - t0))

        ligand_charges = []
        for ligand in mutated_ligands:
            ligand_charges.append(ligand.get_charge())

        print('Calculating Energies...')
        t0 = time.time()
        F = FSim(ligand_name='MOL', ligand_atom=None, pdb_file=self.complex_file,
                                 sim_file=self.complex_file, sim_dir=self.input)


        ##########################
        # load traj to apply charges to and calculate energies from
        traj = []
        for i in range(1):
            traj.append(F.snapshot)
        ###########################


        wildtype_energy = FSim.get_wildtype_energy(F, traj)
        mutant_energy = FSim.get_mutant_energy(F, ligand_charges, traj)

        print(FluorineScanning.free_energy(mutant_energy, wildtype_energy))

        # biggest energy differnce will be biggest free energy differnce
        for ligand in mutant_energy:
            print(ligand[0] - wildtype_energy[0])#zeros because only one frame for now

        t1 = time.time()
        print('Took {} seconds'.format(t1 - t0))

        # better file pointing
        # chosse directory structure and point to approriate places

        # calcualte free energy
        # find lowest free energy
        # how can this be called many times for optimiztion



