#!/usr/bin/env python

import os
import logging
import openmoltools as moltool
import simtk.openmm as mm

class Mol2(object):
    def __init__(self, molecule=None, atoms=None, bonds=None):
        self.data = {'MOLECULE': [], 'ATOM': [], 'BOND': []}
        Mol2.update_data(self, molecule, atoms, bonds)

    def get_data(self, ligand_file_path, ligand_file_name):
        ligand = open('{0}{1}'.format(ligand_file_path, ligand_file_name))
        data = {'MOLECULE': [], 'ATOM': [], 'BOND': []}
        headers = ['@<TRIPOS>MOLECULE', '@<TRIPOS>ATOM', '@<TRIPOS>BOND']
        key = None
        for line in ligand:
            if line[0] == '@':
                line = line.strip('\n')
                if str(line) == '@<TRIPOS>MOLECULE':
                    key = 'MOLECULE'
                elif line == '@<TRIPOS>ATOM':
                    key = 'ATOM'
                elif line == '@<TRIPOS>BOND':
                    key = 'BOND'
            if line not in headers:
                data[key].append(line)
        self.data = data

    def write_mol2(self, file_path, file_name):
        mol2 = []
        headers = {'MOLECULE': '@<TRIPOS>MOLECULE\n', 'ATOM': '@<TRIPOS>ATOM\n', 'BOND': '@<TRIPOS>BOND\n'}
        for k, v in self.data.items():
            mol2.append(headers[k])
            for line in v:
                mol2.append(line)
        f = open('{0}{1}.mol2'.format(file_path, file_name), 'w')
        for line in mol2:
            f.write(line)
        f.close()

    def update_data(self, molecule, atoms, bonds):
        data = {'MOLECULE': [], 'ATOM': [], 'BOND': []}
        data['MOLECULE'] = molecule
        data['ATOM'] = atoms
        data['BOND'] = bonds
        self.data = data

    def get_atom_by_string(self, string):
        if self.data == None:
            logger.error('Please get_data() first')
        aromatic_carbons = []
        for line in self.data['ATOM']:
            if line.split()[5] == string:
                aromatic_carbons.append(line.split()[0])
        if len(aromatic_carbons) == 0:
            err_msg = 'Atom selection {} not reconised or atoms of this type are not in ligand'.format(string)
            logging.error(err_msg)
        return aromatic_carbons

    def get_bonded_hydrogens(self, hydrogens=[], atoms=[]):
        bonds = []
        bonded_h = []
        for line in self.data['BOND']:
            for atom in atoms:
                if atom in line.split():
                    bonds.append(line.split())
                    break
        for bond in bonds:
            if bond[1] in atoms:
                if bond[0] and bond[2] in hydrogens:
                    bonded_h.append(bond[0])
        return bonded_h

    """
        for bond in bonds:
            if bond[1] in atoms:
                if bond[0] in hydrogens:
                    if bond[2] in hydrogens:
                        bonded_h.append(bond[0])
    """

    def mutate_one_element(self, atom, new_element):
        new_atom_data = []
        if new_element == None:
            new_element = 'F'
            logging.info('No elements specified for mutation performing default fluorine scanning')
        new_element = new_element.split('.')

        if len(new_element) > 1:
            tmp = new_element[0]
            new_element[0] = new_element[0] + '.' + new_element[1]
            new_element[1] = tmp

        for line in self.data['ATOM']:
            old_element = []
            if int(line.split()[0]) is int(atom):
                old_element.append(str(line.split()[5]))
                s = str(line.split()[1])
                old_element.append(''.join([i for i in s if not i.isdigit()]))
                for i, entry in enumerate(new_element):
                    line = line.replace(old_element[i], entry)
                new_atom_data.append(line)
            else:
                new_atom_data.append(line)
        return Mol2(molecule=self.data['MOLECULE'], atoms=new_atom_data, bonds=self.data['BOND'])
        # Mol2.update_data(self, atoms=new_atom_data)

    def mutate_elements(self, atoms, new_element=None):
        mutated_systems = []
        for atom in atoms:
            mutated_system = Mol2.mutate_one_element(self, atom, new_element)
            mutated_systems.append(mutated_system)
        return mutated_systems


class MutatedLigand(object):
    def __init__(self, file_path, file_name, prmout=None):
        extension = os.path.splitext(file_name)[1]
        if extension == '.mol2':
            MutatedLigand.run_ante(file_path, file_name, prmout)
            MutatedLigand.create_system(self, file_path, file_name)
        elif extension == '.prmtop':
            MutatedLigand.create_system(self, file_path, file_name)
        else:
            logging.error('Input not reconised')

    def create_system(self, file_path, file_name):
        # PRMOUT MUST BE SET HERE TO HAVE AN EFFECT ON CHRAGE
        parameters_file_path = '/home/cdt1605/FlorineScaning/MOL.prmtop'
        positions_file_path = '/home/cdt1605/FlorineScaning/MOL.inpcrd'
        parameters_file = mm.app.AmberPrmtopFile(parameters_file_path)
        positions_file = mm.app.AmberInpcrdFile(positions_file_path)
        self.system = parameters_file.createSystem()

    def run_ante(file_path, file_name, prmout):
        if prmout == None:
            prmout = '../MOL'
        moltool.amber.run_antechamber(molecule_name='../MOL', input_filename=file_path + file_name)
        moltool.amber.run_tleap(molecule_name=prmout, gaff_mol2_filename='../MOL.gaff.mol2',
                                frcmod_filename='../MOL.frcmod')

    def get_charge(self):
        system = self.system
        ligand_charge = []
        for index in range(system.getNumParticles()):
            for force in system.getForces():
                if isinstance(force, mm.NonbondedForce):
                    nonbonded_force = force
            OG_charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
            ligand_charge.append(OG_charge)
        return (ligand_charge)

    def get_other_ParticleParameters():
        pass
