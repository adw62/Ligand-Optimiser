#!/usr/bin/env python


from Fluorify.energy import *
from Fluorify.mol2 import *
from simtk import unit
from scipy.optimize import minimize
import copy
import logging
import numpy as np
import math

from Fluorify.fluorify import Fluorify

logger = logging.getLogger(__name__)

#CONSTANTS
e = unit.elementary_charges
ee = e*e
nm = unit.nanometer

class Optimize(object):
    def __init__(self, wt_ligand, complex_sys, solvent_sys, output_folder, num_frames, equi, name, steps,
                 param, central_diff, num_fep, rmsd, mol, lock_atoms):

        self.complex_sys = complex_sys
        self.solvent_sys = solvent_sys
        self.num_frames = num_frames
        self.equi = equi
        self.steps = steps
        self.output_folder = output_folder
        self.param = param
        self.central = central_diff
        self.num_fep = num_fep
        self.rmsd = rmsd
        self.mol = mol
        self.lock_atoms = lock_atoms

        self.wt_parameters = wt_ligand.get_parameters()
        self.wt_nonbonded, self.wt_nonbonded_ids, self.wt_excep = Optimize.build_params(self)
        self.excep_scaling = Optimize.get_exception_scaling(self)

        if 'charge' in self.param:
            self.net_charge = Optimize.get_net_charge(self, self.wt_nonbonded)

        # concat all params
        og_charges = [x[0] for x in self.wt_nonbonded]
        self.num_atoms = len(og_charges)
        og_sigma = [x[1] for x in self.wt_nonbonded]
        og_vs_weight = [x[2] for x in self.wt_nonbonded]
        self.og_all_params = og_charges + og_sigma + og_vs_weight

        Optimize.optimize(self, name)

    def get_net_charge(self, wt_nonbonded):
        return sum([x[0] for x in wt_nonbonded])

    def translate_concat_to_atomwise(self, params):
        '''
        Helper to translate concatonated params used by scipy in to atomwise lists of params
        :return:
        '''
        charge = params[:self.num_atoms]
        sigma = params[self.num_atoms:self.num_atoms*2]
        vs_weights = params[self.num_atoms*2:self.num_atoms*3]
        atomwise_params = [[x,y,z] for x,y,z in zip(charge, sigma, vs_weights)]

        return atomwise_params

    def translate_atomwise_to_mutant(self, atomwise, exceptions):
        wt_nonbonded = copy.deepcopy(self.wt_parameters[0])
        vs = []
        for new_atom, old_atom in zip(atomwise, wt_nonbonded):
            old_atom['data'] = [new_atom[0], new_atom[1], float('nan')]
            vs.append(new_atom[2])
        wt_excep = copy.deepcopy(self.wt_parameters[1])
        for new_excep, old_excep in zip(exceptions, wt_excep):
            assert old_excep['id'] == new_excep['id']
            old_atom['data'] = new_excep['data']

        return [wt_nonbonded, wt_excep], vs

    def build_params(self):
        '''
        Build sets of params which can be optimized as list of lists
        built in an atom wise form
        [charge, sigma, VS_weight]
        '''
        wt_nonbonded = copy.deepcopy(self.wt_parameters[0])
        wt_excep = copy.deepcopy(self.wt_parameters[1])
        wt_nonbonded_ids = [x['id'] for x in wt_nonbonded]
        wt_nonbonded = [[x['data'][0]/e, x['data'][1]/nm] for x in wt_nonbonded]
        for id, param in zip(wt_nonbonded_ids, wt_nonbonded):
            if id in self.complex_sys[0].virt_atom_order:
                param.extend([1.0])
            else:
                param.extend([float('nan')])
        wt_excep = [{'id': x['id'], 'data': [x['data'][0]/ee, x['data'][1]/nm]} for x in wt_excep]
        return wt_nonbonded, wt_nonbonded_ids, wt_excep

    def get_exception_scaling(self):
        exceptions = copy.deepcopy(self.wt_excep)
        #Charge scaling
        nonbonded = {x: y[0] for (x, y) in zip(self.wt_nonbonded_ids, self.wt_nonbonded)}
        for product in exceptions:
            ids = list(product['id'])
            id0 = ids[0]
            id1 = ids[1]
            param0 = nonbonded[id0]
            param1 = nonbonded[id1]
            product['data'][0] = product['data'][0]/(param0*param1)
        #Sigma scaling
        nonbonded = {x: y[1] for (x, y) in zip(self.wt_nonbonded_ids, self.wt_nonbonded)}
        for product in exceptions:
            ids = list(product['id'])
            id0 = ids[0]
            id1 = ids[1]
            param0 = nonbonded[id0]
            param1 = nonbonded[id1]
            product['data'][1] = product['data'][1]/((param0+param1)/2)

        return exceptions

    def get_exception_params(self, params):
        new_charge_product = copy.deepcopy(self.excep_scaling)
        # Charge scaling
        nonbonded = {x: y[0] for (x, y) in zip(self.wt_nonbonded_ids, params)}
        for product in new_charge_product:
            ids = list(product['id'])
            id0 = ids[0]
            id1 = ids[1]
            param0 = nonbonded[id0]
            param1 = nonbonded[id1]
            product['data'][0] = product['data'][0] * (param0 * param1)
        # Sigma scaling
        nonbonded = {x: y[1] for (x, y) in zip(self.wt_nonbonded_ids, params)}
        for product in new_charge_product:
            ids = list(product['id'])
            id0 = ids[0]
            id1 = ids[1]
            param0 = nonbonded[id0]
            param1 = nonbonded[id1]
            product['data'][1] = product['data'][1] * ((param0 + param1) / 2)

        return new_charge_product


    def optimize(self, name):
        """optimising ligand charges
        """

        if name == 'scipy':
            opt_charges, ddg_fs = Optimize.scipy_opt(self)
            original_charges = [x[0] for x in self.wt_nonbonded]
            charge_diff = [x-y for x,y in zip(original_charges, opt_charges)]
            Mol2.write_mol2(self.mol, './', 'opt_lig', charges=charge_diff)


        for replica in range(self.num_fep):
            logger.debug('Replica {}/{}'.format(replica+1, self.num_fep))
            complex_dg, complex_error, solvent_dg, solvent_error = Optimize.run_fep(self, opt_charges, 20000)
            ddg_fep = complex_dg - solvent_dg
            ddg_error = (complex_error ** 2 + solvent_error ** 2) ** 0.5
            logger.debug('ddG FEP = {} +- {}'.format(ddg_fep, ddg_error))

        if name != 'FEP_only':
            logger.debug('ddG SSP = {}'.format(ddg_fs))

    def run_fep(self, opt_charges, steps):
        original_charges = [x[0] for x in self.wt_nonbonded]
        opt_exceptions = Optimize.get_charge_product(self, opt_charges)
        logger.debug('Original charges: {}'.format(original_charges))
        logger.debug('Optimized charges: {}'.format(opt_charges))
        mut_charges = [opt_charges, original_charges]
        mut_exceptions = [opt_exceptions, self.wt_excep]
        com_mut_param, sol_mut_param = build_opt_params(mut_charges, mut_exceptions, self)
        com_fep_params = Optimize.build_fep_params(self, com_mut_param, 12)
        sol_fep_params = Optimize.build_fep_params(self, sol_mut_param, 12)
        complex_dg, complex_error = self.complex_sys[0].run_parallel_fep(com_fep_params,
                                                                         None, None, steps, 50, None)
        solvent_dg, solvent_error = self.solvent_sys[0].run_parallel_fep(sol_fep_params,
                                                                         None, None, steps, 50, None)

        return complex_dg, complex_error, solvent_dg, solvent_error


    ###NEEDS REWORKING###
    def get_bounds(self, current_charge, periter_change, total_change):
        og_charge = [x[0] for x in self.wt_nonbonded]
        change = [abs(x-y) for x, y in zip(current_charge, og_charge)]
        bnds = []
        for x, y in zip(change, current_charge):
            if x >= total_change:
                bnds.append((y-periter_change, y))
            elif x <= -total_change:
                bnds.append((y, y+periter_change))
            else:
                bnds.append((y-periter_change, y+periter_change))
        return bnds


    def scipy_opt(self):
        og_all_params = self.og_all_params
        all_params = copy.deepcopy(self.og_all_params)

        #constarints
        con2 = {'type': 'ineq', 'fun': rmsd_change_con, 'args': [og_all_params, self.rmsd]}
        if 'charge' in self.param:
            con1 = {'type': 'eq', 'fun': net_charge_con, 'args': [self.net_charge, self.num_atoms]}
            cons = [con1, con2]
        else:
            cons = [con2]

        #optimization loop
        ddg = 0.0
        for step in range(self.steps):
            write_charges('charges_{}'.format(step), all_params)
            bounds = Optimize.get_bounds(self, all_params, 0.01, 0.5)

            ###REMOVED BOUNDS###
            sol = minimize(objective, all_params, options={'maxiter': 1}, jac=gradient,
                           args=(all_params, self), constraints=cons)
            #update params
            prev_params = all_params
            all_params = sol.x

            #Translate to atomwise lists
            atomwise_params = Optimize.translate_concat_to_atomwise(self, all_params)
            exceptions = Optimize.get_exception_params(self, atomwise_params)
            print(atomwise_params)
            print(exceptions)
            print(end)


            #Translate to mutants

            com_mut_param, sol_mut_param = build_opt_params([charges], [exceptions], self)

            #run new dynamics with updated charges
            self.complex_sys[1] = self.complex_sys[0].run_parallel_dynamics(self.output_folder, 'complex',
                                                                            self.num_frames, self.equi, com_mut_param[0])
            self.solvent_sys[1] = self.solvent_sys[0].run_parallel_dynamics(self.output_folder, 'solvent',
                                                                            self.num_frames, self.equi, sol_mut_param[0])
            logger.debug('Computing reverse leg of accepted step...')
            reverse_ddg = -1*objective(prev_charges, charges, self)
            logger.debug('Forward {} and reverse {} steps'.format(sol.fun, reverse_ddg))
            ddg += (sol.fun+reverse_ddg)/2.0
            logger.debug(sol)
            logger.debug("Current binding free energy improvement {0} for step {1}/{2}".format(ddg, step+1, self.steps))
            write_charges('charges_opt', charges)
        return list(charges), ddg

    def build_fep_params(self, params, windows):
        #build interpolated params
        interpolated_params = []
        for wt_force, mut_force in zip(params[-1], params[0]):
            if wt_force is None:
                interpolated_params.append(None)
            else:
                tmp = [[np.linspace(x['data'], y['data'], windows)] for x, y in zip(wt_force, mut_force)]
                tmp = [[[x[i][j] for i in range(len(x))] for x in tmp] for j in range(windows)]
                interpolated_params.append(tmp)
        #add back in ids
        for i, (force1, force2) in enumerate(zip(params[-1], interpolated_params)):
            if force1 is None:
                pass
            else:
                for j, mutant in enumerate(force2):
                    for k, (param1, param2) in enumerate(zip(force1, mutant)):
                        interpolated_params[i][j][k] = copy.deepcopy(param1)
                        interpolated_params[i][j][k]['data'] = param2[0]

        mutant_systems = [[x, None, y, None] for x, y in zip(interpolated_params[0], interpolated_params[2])]
        return mutant_systems

def build_opt_params(charges, sys_exception_params, sim):
    #reorder exceptions
    exception_order = [sim.complex_sys[0].exceptions_list, sim.solvent_sys[0].exceptions_list]
    offset = [sim.complex_sys[0].offset, sim.solvent_sys[0].offset]
    exception_params = [[], []]
    #[[wild_type], [complex]], None id for ghost which is not used in optimization
    for i, (sys_excep_order, sys_offset) in enumerate(zip(exception_order, offset)):
        for j, mutant_parmas in enumerate(sys_exception_params):
            map = {x['id']: x for x in mutant_parmas}
            sys_exception_params[j] = [map[frozenset(int(x-sys_offset) for x in atom)] for atom in sys_excep_order]
            exception_params[i].append(sys_exception_params[j])
    for i, mut in enumerate(charges):
        charges[i] = [{'id': x, 'data': y} for x, y in zip(sim.wt_nonbonded_ids, mut)]
    complex_parameters = [[x, None, y, None] for x, y in zip(charges, exception_params[0])]
    solvent_parameters = [[x, None, y, None] for x, y in zip(charges, exception_params[1])]
    return complex_parameters, solvent_parameters

def objective(peturbed_charges, current_charges, sim):
    #translate to atomwise
    peturbed_charges = Optimize.translate_concat_to_atomwise(sim, peturbed_charges)
    current_charges = Optimize.translate_concat_to_atomwise(sim, current_charges)
    #Build exceptions
    peturbed_exceptions = Optimize.get_exception_params(sim, peturbed_charges)
    current_exceptions = Optimize.get_exception_params(sim, current_charges)

    #translate to mutant
    #NEED discription of MUTANTS {replace [], insitu [] etc ...}
    peturb_mut, peturbed_vs = Optimize.translate_atomwise_to_mutant(sim, peturbed_charges,
                                                                                       peturbed_exceptions)
    current_mut, current_vs = Optimize.translate_atomwise_to_mutant(sim, current_charges,
                                                                                        current_exceptions)

    #build system with arb weights and vs
    FSim.build(sim.complex_sys[0], const_prefactors=None, weights=None)
    FSim.build(sim.solvent_sys[0], const_prefactors=None, weights=None)

    #build mutants, need mutation description
    #HERE CURRENT

    sys_charge_params = [peturbed_charges, current_charges]
    sys_exception_params = [peturbed_exceptions, current_exceptions]
    com_mut_param, sol_mut_param = build_opt_params(sys_charge_params, sys_exception_params, sim)


    logger.debug('Computing Objective...')
    complex_free_energy = FSim.treat_phase(sim.complex_sys[0], com_mut_param,
                                           sim.complex_sys[1], sim.complex_sys[2], sim.num_frames)
    solvent_free_energy = FSim.treat_phase(sim.solvent_sys[0], sol_mut_param,
                                           sim.solvent_sys[1], sim.solvent_sys[2], sim.num_frames)
    binding_free_energy = complex_free_energy[0] - solvent_free_energy[0]
    return binding_free_energy/unit.kilocalories_per_mole


def gradient(peturbed_charges, current_charges, sim):
    num_frames = int(sim.num_frames)
    dh = 1.5e-04
    if sim.central:
        h = [0.5*dh, -0.5*dh]
        logger.debug('Computing Jacobian with central difference...')
    else:
        h = [dh]
        logger.debug('Computing Jacobian with forward difference...')
    ddG = []
    for diff in h:
        binding_free_energy = []
        mutant_parameters = []
        for i in range(len(peturbed_charges)):
            mutant = copy.deepcopy(peturbed_charges)
            mutant[i] = mutant[i] + diff
            mutant_parameters.append(mutant)

        # Remove systems which correspond to locked atoms
        for x in reversed(sim.lock_atoms):
            mutant_parameters.pop(x)

        mutant_exceptions = [Optimize.get_charge_product(sim, x) for x in mutant_parameters]
        mutant_parameters.append(current_charges)
        mutant_exceptions.append(Optimize.get_charge_product(sim, current_charges))
        com_mut_param, sol_mut_param = build_opt_params(mutant_parameters, mutant_exceptions, sim)
        complex_free_energy = FSim.treat_phase(sim.complex_sys[0], com_mut_param,
                                               sim.complex_sys[1], sim.complex_sys[2], num_frames)
        solvent_free_energy = FSim.treat_phase(sim.solvent_sys[0], sol_mut_param,
                                               sim.solvent_sys[1], sim.solvent_sys[2], num_frames)

        for sol, com in zip(solvent_free_energy, complex_free_energy):
            free_energy = com - sol
            if not sim.central:
                free_energy = free_energy/diff
            binding_free_energy.append(free_energy/unit.kilocalories_per_mole)
        ddG.append(binding_free_energy)

    if sim.central:
        binding_free_energy = []
        for forwards, backwards in zip(ddG[0], ddG[1]):
            binding_free_energy.append((forwards - backwards)/dh)

    #add back in energies for systems corrisponing to locked atoms with energy set to zero
    for x in sim.lock_atoms:
        binding_free_energy.insert(x, 0.0)

    print(binding_free_energy)

    return binding_free_energy


def net_charge_con(current_charge, net_charge, num_charges):
    sum_ = net_charge
    for charge in current_charge[:num_charges]:
        sum_ = sum_ - charge
    return sum_


def rmsd_change_con(current_charge, og_charge, rmsd):
    maximum_rmsd = rmsd
    rmsd = (np.average([(x - y) ** 2 for x, y in zip(current_charge, og_charge)])) ** 0.5
    return maximum_rmsd - rmsd


def write_charges(name, charges):
    file = open(name, 'w')
    for q in charges:
        file.write('{}\n'.format(q))
    file.close()


