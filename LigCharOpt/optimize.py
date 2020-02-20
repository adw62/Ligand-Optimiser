#!/usr/bin/env python


from Fluorify.energy import *
from Fluorify.mol2 import *
from Fluorify.mutants import *
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

        self.wt_parameters = wt_ligand.get_parameters()
        #Unused contains bond, angle and torsion parameters which are not optimized
        self.unused_params = self.wt_parameters[2:5]
        self.wt_parameters = self.wt_parameters[0:2]
        self.wt_nonbonded, self.wt_nonbonded_ids, self.wt_excep = Optimize.build_params(self)
        self.excep_scaling = Optimize.get_exception_scaling(self)

        self.lock_atoms = Optimize.make_lock_list(self, lock_atoms)

        if 'charge' in self.param:
            self.net_charge = Optimize.get_net_charge(self, self.wt_nonbonded)

        # concat all params
        og_charges = [x[0] for x in self.wt_nonbonded]
        self.num_atoms = len(og_charges)
        og_sigma = [x[1] for x in self.wt_nonbonded]
        self.og_all_params = og_charges + og_sigma

        Optimize.optimize(self, name)

    def make_lock_list(self, user_locked_atoms):
        # shift from indexed by 1(mol2) to indexed by 0(python)
        user_locked_atoms = [x - 1 for x in user_locked_atoms]
        # locked charges
        user_locked_charges = [x for x in user_locked_atoms]
        shift = len(self.wt_nonbonded)
        # locked sigmas
        user_locked_sigmas = [x + shift for x in user_locked_atoms]
        if 'charge' not in self.param:
            #lock all charges if not optimizing
            locked_charges = [x for x in self.wt_nonbonded_ids]
        else:
            locked_charges = []
        if 'sigma' not in self.param:
            # lock all sigmas if not optimizing
            locked_sigmas = [x + shift for x in self.wt_nonbonded_ids]
        else:
            locked_sigmas = []
        lock_atoms = user_locked_charges + user_locked_sigmas + locked_charges + locked_sigmas
        lock_atoms = list(set(lock_atoms))  # remove dups
        return sorted(lock_atoms)

    def get_net_charge(self, wt_nonbonded):
        return sum([x[0] for x in wt_nonbonded])

    def translate_concat_to_atomwise(self, params):
        '''
        Helper to translate concatenated params used by scipy in to atomwise lists of params
        :return:
        '''
        charge = params[:self.num_atoms]
        sigma = params[self.num_atoms:self.num_atoms*2]
        atomwise_params = [[x,y] for x,y in zip(charge, sigma)]

        return atomwise_params

    def translate_atomwise_to_mutant(self, atomwise, exceptions):
        wt_nonbonded = copy.deepcopy(self.wt_parameters[0])
        for new_atom, old_atom in zip(atomwise, wt_nonbonded):
            old_atom['data'] = [new_atom[0], new_atom[1]]
        wt_excep = copy.deepcopy(self.wt_parameters[1])
        for new_excep, old_excep in zip(exceptions, wt_excep):
            assert old_excep['id'] == new_excep['id']
            old_atom['data'] = new_excep['data']

        return [wt_nonbonded, wt_excep]

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
        """
        optimising ligand params

        """

        if name == 'grad_decent_ssp':
            opt_params, ddg_fs = Optimize.grad_decent_spp(self)

        elif name == 'grad_decent_fep':
            raise NotImplemented()

        elif name == 'scipy':
            raise NotImplemented

        og_all_params = self.og_all_params
        param_diff = [x-y for x,y in zip(og_all_params, opt_params)]
        if 'sigma' not in self.param:
            Mol2.write_mol2(self.mol, './', 'opt_lig', charges=param_diff[:len(self.wt_nonbonded)])
        elif 'charge' not in self.param:
            Mol2.write_mol2(self.mol, './', 'opt_lig', charges=param_diff[len(self.wt_nonbonded):])
        else:
            logger.debug('Can not write mol2 file containing both sigma and charge params. Please use param_opt file to create your own')

        for replica in range(self.num_fep):
            logger.debug('Replica {}/{}'.format(replica+1, self.num_fep))
            ddg_fep, ddg_error = Optimize.run_fep(self, self.og_all_params, opt_params, 20000)
            logger.debug('ddG FEP = {} +- {}'.format(ddg_fep, ddg_error))

        if name != 'FEP_only':
            logger.debug('ddG SSP = {}'.format(ddg_fs))

    def run_fep(self, start_params, end_params, n_steps):

        mutant = [self.process_mutant(end_params),  self.process_mutant(start_params)]
        mutation = [gen_mutations_dicts(), gen_mutations_dicts()]

        mutant_params = Mutants(mutant, mutation, self.complex_sys[0], self.solvent_sys[0])

        complex_dg, complex_error = self.complex_sys[0].run_parallel_fep(mutant_params, 0, 0, n_steps,
                                                                         n_iterations=50, windows=12)
        solvent_dg, solvent_error = self.solvent_sys[0].run_parallel_fep(mutant_params, 1, 0, n_steps,
                                                                         n_iterations=1, windows=12)
        ddg_fep = complex_dg - solvent_dg
        ddg_error = (complex_error ** 2 + solvent_error ** 2) ** 0.5

        return ddg_fep, ddg_error

    def run_dynamics(self, all_params):

        mutant = [self.process_mutant(all_params)]
        mutation = [gen_mutations_dicts()]

        mutant_params = Mutants(mutant, mutation, self.complex_sys[0], self.solvent_sys[0])

        #run dynamics on built system passing arb q and sigma, is there a better way to apply that just passing
        #non bonded here?
        self.complex_sys[1] = self.complex_sys[0].run_parallel_dynamics(self.output_folder, 'complex',
                                                                        self.num_frames, self.equi,
                                                                        mutant_params.complex_params[0])
        self.solvent_sys[1] = self.solvent_sys[0].run_parallel_dynamics(self.output_folder, 'solvent',
                                                                        self.num_frames, self.equi,
                                                                        mutant_params.solvent_params[0])
    '''
    TODO
    multi permu**
    sigma only refactor use line search
    #Can use a line search effeciently with full FEP if we abandon vs optimization
    #because can use many windows without rebuilding
    '''

    #Can use a line search effeciently with full FEP if we abandon vs optimization
    #because can use many windows without rebuilding
    def grad_decent_spp(self):
        all_params = copy.deepcopy(self.og_all_params)
        step = 0
        max_step_size = 0.05
        damp = 0.85
        #optimization loop
        ddg = 0.0
        while step < self.steps:
            step_size = max_step_size
            write_charges('params_{}'.format(step), all_params)
            grad = gradient(all_params, self)
            write_charges('gradient_{}'.format(step), grad)
            grad = np.array(grad)
            constrained_step = constrain_net_charge(grad, len(self.wt_nonbonded))
            norm_const_step = constrained_step / np.linalg.norm(constrained_step)

            count = 0
            #Line search
            forward_ddg = 1.0
            while forward_ddg > 0.0:
                count += 1
                if count > 10:
                    raise ValueError('Line search failed with ddg {}'.format(forward_ddg))
                all_params_plus_one = all_params - step_size * norm_const_step
                logger.debug('Computing objective with step size {}...'.format(step_size))
                forward_ddg = objective(all_params_plus_one, all_params, self)
                step_size = step_size * damp

            #Run some dynamics with new charges
            logger.debug('Computing reverse leg of accepted step...')
            self.run_dynamics(all_params_plus_one)
            reverse_ddg = -1 * objective(all_params, all_params_plus_one, self)
            logger.debug('Forward {} and reverse {} steps'.format(forward_ddg, reverse_ddg))
            ddg += (forward_ddg + reverse_ddg) / 2.0
            logger.debug(
                "Current binding free energy improvement {0} for step {1}/{2}".format(ddg, step + 1, self.steps))
            all_params = all_params_plus_one
            write_charges('params_opt', all_params)
            step += 1

        return list(all_params), ddg

    def process_mutant(self, parameters):
        '''
        :param parameters: List of charge, sigma and vs charges
        :return:
        '''
        #translate to atomwise
        atomwise_params = Optimize.translate_concat_to_atomwise(self, parameters)
        #Build exceptions
        exceptions = Optimize.get_exception_params(self, atomwise_params)

        #translate to mutant format
        mut = Optimize.translate_atomwise_to_mutant(self, atomwise_params, exceptions)

        return mut+self.unused_params


def gen_mutations_dicts(add=[], subtract=[], replace=[None], replace_insitu=[None]):
    return {'add': add, 'subtract': subtract, 'replace': replace, 'replace_insitu': replace_insitu}


def objective(peturbed_params, current_params, sim):
    systems = [peturbed_params, current_params]
    mutants = [sim.process_mutant(x) for x in systems]
    mutations = [gen_mutations_dicts(), gen_mutations_dicts()]

    mutant_params = Mutants(mutants, mutations, sim.complex_sys[0], sim.solvent_sys[0])

    complex_free_energy = FSim.treat_phase(sim.complex_sys[0], mutant_params.complex_params,
                                           sim.complex_sys[1], sim.complex_sys[2], sim.num_frames)
    solvent_free_energy = FSim.treat_phase(sim.solvent_sys[0], mutant_params.solvent_params,
                                           sim.solvent_sys[1], sim.solvent_sys[2], sim.num_frames)
    binding_free_energy = complex_free_energy[0] - solvent_free_energy[0]

    return binding_free_energy/unit.kilocalories_per_mole


def gradient(all_params, sim):
    num_frames = int(sim.num_frames)
    dh = 1.5e-04
    if sim.central:
        h = [0.5*dh, -0.5*dh]
        logger.debug('Computing jacobian with central difference...')
    else:
        h = [dh]
        logger.debug('Computing jacobian with forward difference...')
    ddG = []

    for diff in h:
        binding_free_energy = []
        mutant_parameters = []
        for i in range(len(all_params)):
            # Skip systems which correspond to locked atoms
            if i not in sim.lock_atoms:
                mutant = copy.deepcopy(all_params)
                mutant[i] = mutant[i] + diff
                mutant_parameters.append(mutant)

        #make mutant systems
        mutants = [sim.process_mutant(x) for x in mutant_parameters]

        #make reference system
        current_sys = sim.process_mutant(all_params)

        # Generate dictionaries to discribe mutations
        mutations = [gen_mutations_dicts() for x in mutants]

        #Append wt system to end to be used as central reference state
        mutants.append(current_sys)
        mutations.append(gen_mutations_dicts())

        mutant_params = Mutants(mutants, mutations, sim.complex_sys[0], sim.solvent_sys[0])

        complex_free_energy = FSim.treat_phase(sim.complex_sys[0], mutant_params.complex_params,
                                               sim.complex_sys[1], sim.complex_sys[2], num_frames)
        solvent_free_energy = FSim.treat_phase(sim.solvent_sys[0], mutant_params.solvent_params,
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
    return binding_free_energy

def constrain_net_charge(delta, num_charges):
    delta_q = delta[:num_charges]
    delta_not_q = delta[num_charges:]
    val = np.sum(delta_q) / len(delta_q)
    delta_q = [x - val for x in delta_q]
    return np.append(np.array(delta_q), delta_not_q)


def write_charges(name, charges):
    file = open(name, 'w')
    for q in charges:
        file.write('{}\n'.format(q))
    file.close()


