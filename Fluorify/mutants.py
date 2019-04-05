#!/usr/bin/env python

import time
from simtk import unit
import logging
import copy
import numpy as np

logger = logging.getLogger(__name__)

#CONSTANTS
e = unit.elementary_charges
ee = e*e
nm = unit.nanometer
kj_mol = unit.kilojoules_per_mole


class Mutants(object):
    def __init__(self, params, mutations, complex_sys, solvent_sys):
        """
        A class for applying mutant parameters to openmm system
        :param params: List of lists for mutant parameters
        :param atom_order: list of lists for order of atoms in each sys
        :param exception_order: list of lists for order of exceptions in each sys
        :param bond_order: list of lists for order of bonds in each sys
        :param sys_offset: number of addition atoms between beginning of ligand in each sys
        :param virtual_offset: number of total atoms in each sys, such that this will be the
                                first index of any virtual atoms added to represent dual topology
        """
        atom_order = [complex_sys.ligand_info[0], solvent_sys.ligand_info[0]]
        virt_atoms = [complex_sys.virt_atom_order, solvent_sys.virt_atom_order]
        exception_order = [complex_sys.exceptions_list, solvent_sys.exceptions_list]
        h_virt_excep = [complex_sys.h_virt_excep, solvent_sys.h_virt_excep]
        bond_order = [complex_sys.bond_list, solvent_sys.bond_list]
        torsion_order = [complex_sys.torsion_list, solvent_sys.torsion_list]

        self.offset = [complex_sys.offset, solvent_sys.offset]
        self.virtual_offset = [complex_sys.virt_atom_shift, solvent_sys.virt_atom_shift]
        self.num_mutants = len([x[0] for x in params])

        nonbonded_params, nonbonded_ghosts = Mutants.build_nonbonded(self, params, mutations, virt_atoms, atom_order)
        exception_params, exception_ghosts = Mutants.build_exceptions(self, params, mutations,
                                                                      h_virt_excep, exception_order)
        bonded_params = Mutants.build_bonds(self, params, mutations, bond_order)
        torsion_params = Mutants.build_torsions(self, params, mutations, torsion_order)

        self.complex_params = [[x, y, z, l, k, p] for x, y, z, l, k, p
                          in zip(nonbonded_params[0], nonbonded_ghosts[0], exception_params[0], exception_ghosts[0], bonded_params[0], torsion_params[0])]
        self.solvent_params = [[x, y, z, l, k, p] for x, y, z, l, k, p
                          in zip(nonbonded_params[1], nonbonded_ghosts[1], exception_params[1], exception_ghosts[1], bonded_params[1], torsion_params[1])]

        self.all_systems_params = [self.complex_params, self.solvent_params]

    def build_nonbonded(self, params, mutations, virt_atoms, atom_order):
        #reduce to just nonbonded
        params = copy.deepcopy([x[0] for x in params])

        #Reorder params from ligand only system to match solvent and complex systems
        nonbonded_params = [[], []]
        for i, (sys_atom_order, sys_offset) in enumerate(zip(atom_order, self.offset)):
            sys_nonbonded_params = copy.deepcopy(params)
            for j, mutant_params in enumerate(sys_nonbonded_params):
                map = {x['id']: x for x in mutant_params}
                sys_nonbonded_params[j] = [map[int(atom - sys_offset)] for atom in sys_atom_order]
                nonbonded_params[i].append(sys_nonbonded_params[j])

        # Build nonbonded ghosts which handle parmas for dual topology
        nonbonded_ghosts = [[], []]
        for i, (sys, sys_virt_offset) in enumerate(zip(nonbonded_ghosts, self.virtual_offset)):
            tmp = [{'id': int(sys_virt_offset+i), 'data': [0.0*e, 0.26*unit.nanometer, 0.0*unit.kilojoules_per_mole]} for i in range(len(virt_atoms[0]))]
            nonbonded_ghosts[i] = [copy.deepcopy(tmp) for i in range(len(nonbonded_params[0]))]

        # transfer params from original topology to ghost topology
        for i, (sys, sys_virt_atoms) in enumerate(zip(nonbonded_params, virt_atoms)):
            for j, (mutant_params, mutant) in enumerate(zip(sys, mutations)):
                atom_idxs = mutant['replace']
                if None not in atom_idxs:
                    for atom in atom_idxs:
                        atom = int(atom-1)
                        transfer_params = copy.deepcopy(mutant_params[atom])
                        transfer_params = transfer_params['data']
                        mutant_params[atom] = {'id': atom, 'data': [0.0*e, 0.26*nm, 0.0*kj_mol]}
                        transfer_index = sys_virt_atoms.index(atom)
                        virt_id = nonbonded_ghosts[i][j][transfer_index]['id']
                        nonbonded_ghosts[i][j][transfer_index] = {'id': virt_id, 'data': transfer_params}

        return nonbonded_params, nonbonded_ghosts

    def build_exceptions(self, params, mutations, h_virt_excep, exception_order):
        """

        :param params:
        :param mutations:
        :param virt_exceptions: ghost from complex and solsvent systems
        :param exception_order: order of vannila exceptions for complex and solvent systems
        :return:
        """
        #reduce to just exceptions
        params = copy.deepcopy([x[1] for x in params])
        #find exceptions of subtracted atoms
        excep_to_add = []
        for i in range(len(params)):
            atoms_to_mute = sorted(mutations[i]['subtract'])
            exceptions = []
            for atom_id in atoms_to_mute:
                for excep in exception_order[0]:
                    if atom_id+self.offset[0] in excep:
                        exceptions.append([x-self.offset[0] for x in excep])
            excep_to_add.append(exceptions)
        #iterate over mutants
        for i, x in enumerate(excep_to_add):
            for exception in x:
                params[i].append({'id': frozenset(exception), 'data': [0.0*ee, 0.1*nm, 0.0*kj_mol]})

        #reorder
        exception_params = [[], []]
        for i, (sys_excep_order, sys_offset) in enumerate(zip(exception_order, self.offset)):
            sys_exception_params = copy.deepcopy(params)
            for j, mutant_parmas in enumerate(sys_exception_params):
                map = {x['id']: x for x in mutant_parmas}
                sys_exception_params[j] = [map[frozenset(int(x-sys_offset) for x in atom)] for atom in sys_excep_order]
                exception_params[i].append(sys_exception_params[j])

        """
        #TEST
        for x,y in zip(sys_exception_params[1][0], exception_order[1]):
            if x['id'] != frozenset(int(atom-0) for atom in y):
                raise ValueError(x['id'], frozenset(int(atom-0) for atom in y))
        """

        #Build ghost exceptions
        exception_ghosts = copy.deepcopy(exception_params)
        for i, (sys_exception_params, sys_virt_order, sys_offset) in enumerate(zip(exception_ghosts, h_virt_excep, self.offset)):
            for j, mutant_parmas in enumerate(sys_exception_params):
                map = {x['id']: x for x in mutant_parmas}
                exception_ghosts[i][j] = [map[frozenset(int(x-sys_offset) for x in atom)] for atom in sys_virt_order]

        zero = [0.0*ee, 0.1*nm, 0.0*kj_mol]
        # zero flourines in original topology aka exception params
        for i, sys_exception_params in enumerate(exception_params):
            for j, (mutant_parmas, mutant) in enumerate(zip(sys_exception_params, mutations)):
                atom_idxs = mutant['replace']
                if None not in atom_idxs:
                    for atom in atom_idxs:
                        atom = int(atom-1)
                        for k, excep1 in enumerate(mutant_parmas):
                            if atom in excep1['id']:
                                exception_params[i][j][k]['data'] = zero

        # zero everything but fluorines in dual topology aka exception_ghosts
        for i, sys_exception_ghosts in enumerate(exception_ghosts):
            for j, (mutant_parmas, mutant) in enumerate(zip(sys_exception_ghosts, mutations)):
                atom_idxs = mutant['replace']
                if None not in atom_idxs:
                    for atom in atom_idxs:
                        atom = int(atom-1)
                        for k, excep1 in enumerate(mutant_parmas):
                            #Hinders multi permu (killing non corss terms)
                            if atom not in excep1['id']:
                                exception_ghosts[i][j][k]['data'] = zero
                else:
                    for k, excep1 in enumerate(mutant_parmas):
                        exception_ghosts[i][j][k]['data'] = zero
        """
        for x in exception_params[0][0]:
            if x['data'][0] == 0.0*ee:
                print(x)

        for x in exception_ghosts[0][0]:
            if x['data'][0] != 0.0*ee:
                print(x)
        """
        return exception_params, exception_ghosts

    def build_bonds(self, params, mutations, bond_order):
        #reduce to bonds only
        params = copy.deepcopy([x[2] for x in params])
        #find bonds of subtracted atoms
        bonds_to_add = []
        for i in range(len(params)):
            atoms_to_mute = sorted(mutations[i]['subtract'])
            bonds = []
            for atom_id in atoms_to_mute:
                for bond in bond_order[0]:
                    if atom_id+self.offset[0] in bond:
                        bonds.append([x-self.offset[0] for x in bond])
            bonds_to_add.append(bonds)
        #iterate over mutants
        for i, x in enumerate(bonds_to_add):
            for bond in x:
                params[i].append({'id': frozenset(bond), 'data': ['BOND']})

        bonded_params = [[], []]
        for i, (sys_bond_order, sys_offset) in enumerate(zip(bond_order, self.offset)):
            sys_bonded_params = copy.deepcopy(params)
            for j, mutant_params in enumerate(sys_bonded_params):
                map = {x['id']: x for x in mutant_params}
                sys_bonded_params[j] = [map[frozenset(int(x-sys_offset) for x in atom)] for atom in sys_bond_order]
                bonded_params[i].append(sys_bonded_params[j])
        return bonded_params

    def build_torsions(self, params, mutations, torsion_order):
        #reduce to bonds only
        params = copy.deepcopy([x[3] for x in params])
        #find torsions of subtracted atoms
        tor_to_add = []
        for i in range(len(params)):
            atoms_to_mute = sorted(mutations[i]['subtract'])
            torsions = []
            for atom_id in atoms_to_mute:
                for torsion in torsion_order[0]:
                    if atom_id+self.offset[0] in torsion:
                        torsions.append([x-self.offset[0] for x in torsion])
            tor_to_add.append(torsions)
        #iterate over mutants
        for i, x in enumerate(tor_to_add):
            for torsion in x:
                params[i].append({'id': frozenset(torsion), 'data': ['TORSION']})

        torsion_params = [[], []]
        for i, (sys_torsion_order, sys_offset) in enumerate(zip(torsion_order, self.offset)):
            sys_torsion_params = copy.deepcopy(params)
            for j, mutant_params in enumerate(sys_torsion_params):
                map = {x['id']: x for x in mutant_params}
                sys_torsion_params[j] = [map[frozenset(int(x-sys_offset) for x in atom)] for atom in sys_torsion_order]
                torsion_params[i].append(sys_torsion_params[j])
        return torsion_params

    def build_fep_systems(self, system_idx, mutant_idx, windows, opt, charge_only):
        #build interpolated params
        interpolated_params = []
        for wt_force, mut_force in zip(self.all_systems_params[system_idx][-1], self.all_systems_params[system_idx][mutant_idx]):
            tmp = [[unit_linspace(x['data'][i], y['data'][i], windows) for i in range(len(x['data']))] for x, y in zip(wt_force, mut_force)]
            tmp = [[[x[i][j] for i in range(len(x))] for x in tmp] for j in range(windows)]
            interpolated_params.append(tmp)

        #add back in ids
        for i, (force1, force2) in enumerate(zip(self.all_systems_params[system_idx][-1], interpolated_params)):
            for j, mutant in enumerate(force2):
                for k, (param1, param2) in enumerate(zip(force1, mutant)):
                    interpolated_params[i][j][k] = copy.deepcopy(param1)
                    interpolated_params[i][j][k]['data'] = param2

        mutant_systems = [[x, y, z, l, k, p] for x, y, z, l, k, p in zip(interpolated_params[0], interpolated_params[1],
                                                                         interpolated_params[2], interpolated_params[3],
                                                                         interpolated_params[4], interpolated_params[5])]
        return mutant_systems

def unit_linspace(x, y, i):
    try:
        unit1 = x.unit
    except:
        unit1 = None
    try:
        unit2 = y.unit
    except:
        unit2 = None
    if unit1 != unit2:
        raise ValueError('Unmatched units')
    if unit1 is None:
        ans = np.linspace(x, y, i)
        ans = np.floor(ans)
        ans = [int(x) for x in ans]
        return ans
    else:
        ans = np.linspace(x/unit1, y/unit2, i)
        ans = [x*unit1 for x in ans]
        return ans




