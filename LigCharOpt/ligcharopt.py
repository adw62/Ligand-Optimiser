#!/usr/bin/env python

from Fluorify.energy import *
from Fluorify.mol2 import *
from Fluorify.mutants import *
from Fluorify.fluorify import *
from .optimize import Optimize

import os
import time
import shutil
from simtk import unit
import logging

logger = logging.getLogger(__name__)

#CONSTANTS
e = unit.elementary_charges

class LigCharOpt(object):
    def __init__(self, output_folder, mol_name, ligand_name, net_charge, complex_name, solvent_name, job_type,
                 auto_select, c_atom_list, h_atom_list, o_atom_list, num_frames, param, gaff_ver, opt, num_gpu,
                 num_fep, equi, central_diff, opt_name, opt_steps, rmsd, exclude_dualtopo):

        self.output_folder = output_folder
        self.net_charge = net_charge
        self.job_type = job_type
        self.num_frames = num_frames
        self.gaff_ver = gaff_ver
        self.num_fep = int(num_fep)

        # Prepare directories/files and read in ligand from mol2 file
        mol_file = mol_name + '.mol2'
        input_folder = './input/'

        complex_sim_dir = input_folder + complex_name + '/'
        solvent_sim_dir = input_folder + solvent_name + '/'

        if os.path.isdir(self.output_folder):
            logger.debug('Output folder {} already exists. '
                  'Will attempt to skip ligand parametrisation, proceed with caution...'.format(self.output_folder))
        else:
            try:
                os.makedirs(self.output_folder)
            except:
                logger.debug('Could not create output folder {}'.format(self.output_folder))
        shutil.copy2(input_folder+mol_file, self.output_folder)
        self.mol = Mol2()
        try:
            Mol2.get_data(self.mol, input_folder, mol_file)
        except:
            raise ValueError('Could not load molecule {}'.format(self.input_folder + mol_file))

        # Check ligand atom order is consistent across input topologies.
        input_files = [input_folder + mol_file, complex_sim_dir + complex_name + '.pdb',
                       solvent_sim_dir + solvent_name + '.pdb']
        self.mol2_ligand_atoms, complex_ligand_atoms, solvent_ligand_atoms = get_atom_list(input_files, ligand_name)
        if self.mol2_ligand_atoms != complex_ligand_atoms:
            raise ValueError('Names and or name casing and or atom order of ligand not matched across input files.'
                             'Charges will not be applied where expected')
        if complex_ligand_atoms != solvent_ligand_atoms:
            raise ValueError('Names and or name casing and or atom order of ligand not matched across input files.'
                             'Charges will not be applied where expected')
        #write out atom names for convenience
        if opt:
            file = open('atom_names', 'w')
            for name in self.mol2_ligand_atoms:
                file.write('{}\n'.format(name))
            file.close()

        input_files = input_files[1:3]
        self.complex_offset, self.solvent_offset = get_ligand_offset(input_files, self.mol2_ligand_atoms, ligand_name)
        logger.debug('Parametrize wild type ligand...')
        wt_ligand = MutatedLigand(file_path=self.output_folder, mol_name=mol_name,
                                  net_charge=self.net_charge, gaff=self.gaff_ver)

        logger.debug('Loading complex and solvent systems...')
        tests = ['SSP_convergence_test', 'FEP_convergence_test', 'FS_test']

        if opt == True:
            if opt_name in tests:
                run_dynamics = False
            else:
                run_dynamics = True
        else:
            run_dynamics = False

        #COMPLEX
        self.complex_sys = []
        self.complex_sys.append(FSim(ligand_name=ligand_name, sim_name=complex_name, input_folder=input_folder,
                                     param=param, num_gpu=num_gpu, offset=self.complex_offset,
                                     opt=opt, exclude_dualtopo=exclude_dualtopo))
        self.complex_sys.append([complex_sim_dir + complex_name + '.dcd'])
        self.complex_sys.append(complex_sim_dir + complex_name + '.pdb')
        if run_dynamics:
            if not os.path.isfile(self.complex_sys[1][0]):
                self.complex_sys[1] = [complex_sim_dir + complex_name + '_gpu' + str(x) + '.dcd' for x in range(num_gpu)]
                for name in self.complex_sys[1]:
                    if not os.path.isfile(name):
                        self.complex_sys[1] = self.complex_sys[0].run_parallel_dynamics(complex_sim_dir, complex_name,
                                                                                        self.num_frames, equi, None)
                        break
        #SOLVENT
        self.solvent_sys = []
        self.solvent_sys.append(FSim(ligand_name=ligand_name, sim_name=solvent_name, input_folder=input_folder,
                                     param=param, num_gpu=num_gpu, offset=self.solvent_offset,
                                     opt=opt, exclude_dualtopo=exclude_dualtopo))
        self.solvent_sys.append([solvent_sim_dir + solvent_name + '.dcd'])
        self.solvent_sys.append(solvent_sim_dir + solvent_name + '.pdb')
        if run_dynamics:
            if not os.path.isfile(self.solvent_sys[1][0]):
                self.solvent_sys[1] = [solvent_sim_dir + solvent_name + '_gpu' + str(x) + '.dcd' for x in range(num_gpu)]
                for name in self.solvent_sys[1]:
                    if not os.path.isfile(name):
                        self.solvent_sys[1] = self.solvent_sys[0].run_parallel_dynamics(solvent_sim_dir, solvent_name,
                                                                                        self.num_frames, equi, None)
                        break

        if opt:
            Optimize(wt_ligand, self.complex_sys, self.solvent_sys, output_folder, self.num_frames, equi, opt_name, opt_steps,
                     param, central_diff, self.num_fep, rmsd, self.mol)
        else:
            LigCharOpt.fep(self, wt_ligand, auto_select, c_atom_list, h_atom_list, o_atom_list)

    def fep(self, wt_ligand, auto_select, c_atom_list, h_atom_list, o_atom_list):
        """preparation and running of free energy calculations
        """

        #Generate mutant systems with selected pertibations
        mutated_systems, mutations = Fluorify.element_perturbation(self, auto_select, c_atom_list, h_atom_list, o_atom_list)

        """
        Write Mol2 files with substitutions of selected atoms.
        Run antechamber and tleap on mol2 files to get prmtop.
        Create OpenMM systems of ligands from prmtop files.
        Extract ligand parameters from OpenMM systems.
        """
        logger.debug('Parametrize mutant ligands...')
        t0 = time.time()

        mutated_ligands = []
        for index, sys in enumerate(mutated_systems):
            mol_name = 'molecule'+str(index)
            Mol2.write_mol2(sys, self.output_folder, mol_name)
            mutated_ligands.append(MutatedLigand(file_path=self.output_folder, mol_name=mol_name,
                                                 net_charge=self.net_charge, gaff=self.gaff_ver))

        wt_parameters = wt_ligand.get_parameters()
        mutant_parameters = []
        for i, ligand in enumerate(mutated_ligands):
            mute = mutations[i]['subtract']
            mutant_parameters.append(ligand.get_parameters(mute))

        #last entry of mutant is wildtype
        mutant_parameters.append(wt_parameters)
        mutations.append({'add': [], 'subtract': [], 'replace': [None], 'replace_insitu': [None]})

        mutant_params = Mutants(mutant_parameters, mutations, self.complex_sys[0], self.solvent_sys[0])
        del mutant_parameters

        t0 = time.time()
        for i, mut in enumerate(mutant_params.complex_params[:-1]):
            atom_names = []
            replace = mutations[i]['replace']
            replace.extend(mutations[i]['replace_insitu'])
            for atom in replace:
                atom_index = int(atom)-1
                atom_names.append(self.mol2_ligand_atoms[atom_index])
            complex_dg, complex_error = self.complex_sys[0].run_parallel_fep(mutant_params, 0, i, 20000, 50, 12)
            solvent_dg, solvent_error = self.solvent_sys[0].run_parallel_fep(mutant_params, 1, i, 20000, 50, 12)
            ddg_fep = complex_dg - solvent_dg
            ddg_error = (complex_error**2+solvent_error**2)**0.5
            logger.debug('Mutant {}:'.format(atom_names))
            logger.debug('ddG FEP = {} +- {}'.format(ddg_fep, ddg_error))
        t1 = time.time()
        logger.debug('Took {} seconds'.format(t1 - t0))

