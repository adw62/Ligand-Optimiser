#!/usr/bin/env python

import os
import logging
import openmoltools as moltool
import simtk.openmm as mm
import copy
import numpy as np
from simtk import unit
import logging

logger = logging.getLogger(__name__)

class Mol2(object):
    def __init__(self, molecule=None, atoms=None, bonds=None, other=None):
        """A class for reading, writing and modifying a ligand in a mol2 file.

        molecule: All information in mol2 file under '@<TRIPOS>MOLECULE' header
        atoms: All information in mol2 file under '@<TRIPOS>ATOM' header
        bonds: All information in mol2 file under '@<TRIPOS>BOND' header
        other: All information in mol2 file not under MOLECULE, ATOM or BOND header
        """
        self.data = copy.deepcopy({'MOLECULE': molecule, 'ATOM': atoms, 'BOND': bonds, 'OTHER': other})

    def get_data(self, ligand_file_path, ligand_file_name):
        ligand = open(ligand_file_path+ligand_file_name)
        data = {'MOLECULE': [], 'ATOM': [], 'BOND': [], 'OTHER': []}
        headers = ['@<TRIPOS>MOLECULE', '@<TRIPOS>ATOM', '@<TRIPOS>BOND']
        key = None
        for line in ligand:
            if line[0] == '@':
                line = line.strip('\n')
                if line == '@<TRIPOS>MOLECULE':
                    key = 'MOLECULE'
                elif line == '@<TRIPOS>ATOM':
                    key = 'ATOM'
                elif line == '@<TRIPOS>BOND':
                    key = 'BOND'
                else:
                    key = 'OTHER'
            if line not in headers:
                if key == 'ATOM' or key == 'BOND':
                    line = line.split()
                    data[key].append(line)
                else:
                    data[key].append(line)
        self.data = data

    def write_mol2(self, file_path, file_name, charges=None):
        mol2 = []
        headers = {'MOLECULE': '@<TRIPOS>MOLECULE\n', 'ATOM': '@<TRIPOS>ATOM\n',
                   'BOND': '@<TRIPOS>BOND\n', 'OTHER': ''}
        for k, v in self.data.items():
            mol2.append(headers[k])
            if k == 'ATOM':
                if charges is not None:
                    for charge, line in zip(charges, v):
                        line[8] = charge
                        line = ('{0:>7} {1:<10} {2:<9} {3:<9} {4:<5} {5:<9} {6:<1} {7:<11} {8}\n'.format(*line))
                        mol2.append(line)
                else:
                    for line in v:
                        line = ('{0:>7} {1:<10} {2:<9} {3:<9} {4:<5} {5:<9} {6:<1} {7:<11} {8}\n'.format(*line))
                        mol2.append(line)
            elif k == 'BOND':
                for line in v:
                    line = ('{0:>6} {1:>4} {2:>4} {3:>1}\n'.format(*line))
                    mol2.append(line)
            else:
                for line in v:
                    mol2.append(line)
        f = open(file_path+file_name+'.mol2', 'w')
        for line in mol2:
            f.write(line)
        f.close()

    def update_data(self, molecule, atoms, bonds, other):
        data = {'MOLECULE': molecule, 'ATOM': atoms, 'BOND': bonds, 'OTHER': other}
        self.data = data

    def get_atom_by_string(self, string, wild_card=False):
        atoms = []
        for line in self.data['ATOM']:
            if wild_card == True:
                if string in line[5]:
                    atoms.append(line[0])
            else:
                if line[5] == string:
                    atoms.append(line[0])
        if len(atoms) == 0:
            raise ValueError('Atom selection {} not recognised or atoms of this type are not in ligand'.format(string))
        return atoms

    def get_bonded_neighbours(self, atom):
        neighbours = []
        for line in self.data['BOND']:
            if int(line[1]) is int(atom):
                neighbours.append(line[2])
            if int(line[2]) is int(atom):
                neighbours.append((line[1]))
        return neighbours

    def get_atom_position(self, atom):
        #mol2 file indexed from 1, data indexed from 0 therefor -1
        line = self.data['ATOM'][atom-1]
        positions = np.array([float(line[2]), float(line[3]), float(line[4])])
        return positions

    def reindex_data(self, num_atoms, num_bonds, atom, bond, remove_atom=False):
        new_molecule_data = []
        for line in self.data['MOLECULE']:
            try:
                if int(line.split()[0]) == num_atoms:
                    if remove_atom:
                        line = line.replace(str(num_atoms), str(num_atoms-1))
                        line = line.replace(str(num_bonds), str(num_bonds-1))
                    else:
                        line = line.replace(str(num_atoms), str(num_atoms+1))
                        line = line.replace(str(num_bonds), str(num_bonds+1))
            except:
                pass
            new_molecule_data.append(line)

        if remove_atom:
            for i, line in enumerate(self.data['ATOM']):
                if i >= (int(atom)-1):
                    line[0] = i+1

            for i, line in enumerate(self.data['BOND']):
                if i >= (int(bond)):
                    line[0] = i+1
                if int(line[1]) >= int(atom):
                    line[1] = str(int(line[1])-1)
                if int(line[2]) >= int(atom):
                    line[2] = str(int(line[2])-1)

        Mol2.update_data(self, new_molecule_data, self.data['ATOM'],
                         self.data['BOND'], self.data['OTHER'])

    def remove_atom(self, atom):
        new_atom_data = []
        num_atoms = len(self.data['ATOM'])
        for line in self.data['ATOM']:
            if int(line[0]) is not int(atom):
                new_atom_data.append(line)

        new_bond_data = []
        num_bonds = len(self.data['BOND'])
        bond = []
        for i, line in enumerate(self.data['BOND']):
            if int(line[1]) is int(atom):
                bond.append(i)
            elif int(line[2]) is int(atom):
                bond.append(i)
            else:
                new_bond_data.append(line)
        if len(bond) > 1:
            raise ValueError('Atom selected to be removed had more than one bond.'
                             'Currently can only remove atoms with one bond')

        Mol2.update_data(self, molecule=self.data['MOLECULE'], atoms=new_atom_data,
                         bonds=new_bond_data, other=self.data['OTHER'])
        Mol2.reindex_data(self, num_atoms, num_bonds, atom, bond[0], remove_atom=True)

    def add_atom(self, atom, element, position):
        num_atoms = len(self.data['ATOM'])
        num_bonds = len(self.data['BOND'])

        line = copy.deepcopy(self.data['ATOM'][-1])
        line[0] = num_atoms + 1
        line[1] = element
        line[2] = position[0]
        line[3] = position[1]
        line[4] = position[2]
        line[5] = element
        self.data['ATOM'].append(line)

        line = copy.deepcopy(self.data['BOND'][-1])
        line[0] = num_bonds + 1
        line[1] = atom
        line[2] = num_atoms + 1
        self.data['BOND'].append(line)

        Mol2.update_data(self, molecule=self.data['MOLECULE'], atoms=self.data['ATOM'],
                         bonds=self.data['BOND'], other=self.data['OTHER'])
        Mol2.reindex_data(self, num_atoms, num_bonds, None, None)

    def mutate_atoms(self, atoms, new_element):
        data = copy.deepcopy(self.data['ATOM'])
        new_element = new_element.split('.')
        if len(new_element) > 1:
            tmp = new_element[0]
            new_element[0] = new_element[0] + '.' + new_element[1]
            new_element[1] = tmp
        else:
            new_element.append(new_element[0])
        for i, atom in enumerate(atoms):
            new_atom_data = []
            for line in data:
                if int(line[0]) is int(atom):
                    line[5] = new_element[0]
                    line[1] = (new_element[1] + str(i+1))
                    new_atom_data.append(line)
                else:
                    new_atom_data.append(line)
                data = new_atom_data
        return Mol2(molecule=self.data['MOLECULE'], atoms=new_atom_data,
                    bonds=self.data['BOND'], other=self.data['OTHER'])

    def mutate(self, mutations, new_element):
        mutated_systems = []
        for atoms in mutations:
            mutated_system = Mol2.mutate_atoms(self, atoms, new_element)
            mutated_systems.append(mutated_system)
        return mutated_systems


class MutatedLigand(object):
    def __init__(self, file_path, mol_name, net_charge):
        """A class for extracting parameters from a mutated ligand in a mol2 file.
        file_path: location of ligand file
        mol_name: name of mol file
        """
        file_name = mol_name+'.mol2'
        run_ante(file_path, file_name, mol_name, net_charge)
        MutatedLigand.create_system(self, file_path, mol_name)

    def create_system(self, file_path, name):
        parameters_file_path = file_path + name + '.prmtop'
        parameters_file = mm.app.AmberPrmtopFile(parameters_file_path)
        self.system = parameters_file.createSystem()

    def get_parameters(self, atoms_to_mute=[]):
        #atoms to mute does not work if there are more than 2 sequential atoms to mute
        system = self.system
        ligand_parameters = []
        for force in system.getForces():
            if isinstance(force, mm.NonbondedForce):
                nonbonded_force = force
        for index in range(system.getNumParticles()):
            if index in atoms_to_mute:
                charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
                ligand_parameters.append([0.0*charge, 1.0*unit.angstrom, epsilon*0.0])
                if index + 1 in atoms_to_mute:
                    ligand_parameters.append([0.0*charge, 1.0*unit.angstrom, epsilon*0.0])
                    if index + 2 in atoms_to_mute:
                        raise ValueError('Work in progress: Too many pyridinations per mutant')
                ligand_parameters.append([charge, sigma, epsilon])
            else:
                charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
                ligand_parameters.append([charge, sigma, epsilon])
        return np.asarray(ligand_parameters)

def run_ante(file_path, file_name, name, net_charge):
    if os.path.exists(file_path+name+'.prmtop'):
        logger.debug('{0} found skipping antechamber and tleap for {1}'.format(file_path+name+'.prmtop', name))
    else:
        moltool.amber.run_antechamber(molecule_name=file_path+name, input_filename=file_path+file_name,
                                      net_charge=net_charge, gaff_version='gaff2')
        moltool.amber.run_tleap(molecule_name=file_path+name, gaff_mol2_filename=file_path+name+'.gaff.mol2',
                                frcmod_filename=file_path+name+'.frcmod', leaprc='leaprc.gaff2')
