#!/usr/bin/env python


from .energy import FSim
from simtk import unit
from scipy.optimize import minimize
import copy
import logging
import numpy as np

logger = logging.getLogger(__name__)

#CONSTANTS
e = unit.elementary_charges
ee = e*e

class Optimize(object):
    def __init__(self, wt_ligand, complex_sys, solvent_sys, output_folder, num_frames, equi, name, steps,
                 charge_only, central_diff, num_fep):

        self.complex_sys = complex_sys
        self.solvent_sys = solvent_sys
        self.num_frames = num_frames
        self.equi = equi
        self.steps = steps
        self.output_folder = output_folder
        self.charge_only = charge_only
        self.central = central_diff
        self.num_fep = num_fep
        self.wt_nonbonded, self.wt_nonbonded_ids, self.wt_excep,\
        self.net_charge = Optimize.build_params(self, wt_ligand)
        self.excep_scaling = Optimize.get_exception_scaling(self)
        Optimize.optimize(self, name)

    def build_params(self, wt_ligand):
        if self.charge_only == False:
            #TODO add VDW
            raise ValueError('Can only optimize charge')
        else:
            #get all wt params
            wt_parameters = wt_ligand.get_parameters()
            #trim down to nonbonded and exclusions
            wt_nonbonded = wt_parameters[0]
            wt_excep = wt_parameters[1]
            wt_nonbonded_ids = [x['id'] for x in wt_nonbonded]
            wt_nonbonded = [x['data'][0]/e for x in wt_nonbonded]
            wt_nonbonded = [[x] for x in wt_nonbonded]
            wt_excep = [{'id': x['id'], 'data': x['data'][0]/ee} for x in wt_excep]
            net_charge = sum([x[0] for x in wt_nonbonded])
        return wt_nonbonded, wt_nonbonded_ids, wt_excep, net_charge

    def get_exception_scaling(self):
        exceptions = copy.deepcopy(self.wt_excep)
        if self.charge_only == False:
            pass
        else:
            for atom_id, param in zip(self.wt_nonbonded_ids, self.wt_nonbonded):
                charge = param[0]
                for charge_prod in exceptions:
                    if atom_id in charge_prod['id']:
                        charge_prod['data'] = charge_prod['data']/charge
        return exceptions

    def get_charge_product(self, charges):
        charge_ids = self.wt_nonbonded_ids
        new_charge_product = copy.deepcopy(self.excep_scaling)
        if self.charge_only == False:
            pass
        else:
            for atom_id, charge in zip(charge_ids, charges):
                for charge_prod in new_charge_product:
                    if atom_id in charge_prod['id']:
                        charge_prod['data'] = charge_prod['data']*charge
        return new_charge_product

    def optimize(self, name):
        """optimising ligand charges
        """

        if name == 'scipy':
            opt_charges, ddg_fs = Optimize.scipy_opt(self)

        elif name == 'FEP_only':
            #Get optimized charges from file to calc full FEP ddG
            file = open('charges_opt', 'r')
            logger.debug('Using charges from {}'.format(file.name))
            opt_charges = [float(line) for line in file]

        elif name == 'convergence_test':
            self.num_fep = 0
            perturbation = 0.01
            sampling = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
            og_charges = [x[0] for x in self.wt_nonbonded]
            peturb_charges = copy.deepcopy(og_charges)
            for i in range(0, int(np.ceil(len(peturb_charges) / 2))):
                peturb_charges[i] = peturb_charges[i] + perturbation
            for i in range(int(len(peturb_charges) / 2), int(len(peturb_charges))):
                peturb_charges[i] = peturb_charges[i] - perturbation
            logger.debug('og net charge {}, perturbed net charge {}'.format(sum(og_charges), sum(peturb_charges)))
            if round(sum(og_charges), 5) != round(sum(peturb_charges), 5):
                raise ValueError('Net charge change')
            exceptions = Optimize.get_charge_product(self, og_charges)
            com_mut_param, sol_mut_param = build_opt_params([og_charges], [exceptions], self)
            for num_frames in sampling:
                self.num_frames = num_frames
                for replica in range(0, 3):
                    self.complex_sys[1] = self.complex_sys[0].run_parallel_dynamics(self.output_folder, 'complex', self.num_frames * 2500,
                                                                                    self.equi, com_mut_param[0])
                    self.solvent_sys[1] = self.solvent_sys[0].run_parallel_dynamics(self.output_folder, 'solvent', self.num_frames * 2500,
                                                                                    self.equi, sol_mut_param[0])

                    ddG = objective(og_charges, peturb_charges, self)
                    logger.debug('ddG Fluorine Scanning  for {} frames for replica {} = {} kcal/mol'.format(num_frames, replica, ddG))

        else:
            raise ValueError('No other optimizers implemented')

        for replica in range(self.num_fep):
            logger.debug('Replica {}/{}'.format(replica+1, self.num_fep))
            complex_dg, complex_error, solvent_dg, solvent_error = Optimize.run_fep(self, opt_charges)
            ddg_fep = complex_dg - solvent_dg
            ddg_error = (complex_error ** 2 + solvent_error ** 2) ** 0.5
            logger.debug('ddG FEP = {} +- {}'.format(ddg_fep, ddg_error))
        if name != 'FEP_only':
            logger.debug('ddG Fluorine Scanning = {}'.format(ddg_fs))

    def run_fep(self, opt_charges):
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
                                                                         None, None, 20000, 50, None)
        solvent_dg, solvent_error = self.solvent_sys[0].run_parallel_fep(sol_fep_params,
                                                                         None, None, 20000, 50, None)

        return complex_dg, complex_error, solvent_dg, solvent_error


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
        og_charges = [x[0] for x in self.wt_nonbonded]
        charges = [x[0] for x in self.wt_nonbonded]
        con1 = {'type': 'eq', 'fun': net_charge_con, 'args': [self.net_charge]}
        con2 = {'type': 'ineq', 'fun': rmsd_change_con, 'args': [og_charges]}
        cons = [con1, con2]
        ddg = 0.0
        for step in range(self.steps):
            write_charges('charges_opt', charges)
            write_charges('charges_{}'.format(step), charges)
            bounds = Optimize.get_bounds(self, charges, 0.01, 0.5)
            sol = minimize(objective, charges, bounds=bounds, options={'maxiter': 1}, jac=gradient,
                           args=(charges, self), constraints=cons)
            prev_charges = charges
            charges = sol.x
            exceptions = Optimize.get_charge_product(self, charges)
            com_mut_param, sol_mut_param = build_opt_params([charges], [exceptions], self)

            #run new dynamics with updated charges
            self.complex_sys[1] = self.complex_sys[0].run_parallel_dynamics(self.output_folder, 'complex_step'+str(step),
                                                                            self.num_frames*2500, self.equi, com_mut_param[0])
            self.solvent_sys[1] = self.solvent_sys[0].run_parallel_dynamics(self.output_folder, 'solvent_step'+str(step),
                                                                            self.num_frames*2500, self.equi, sol_mut_param[0])
            logger.debug('Computing reverse leg of accepted step...')
            reverse_ddg = -1*objective(prev_charges, charges, self)
            logger.debug('Forward {} and reverse {} steps'.format(sol.fun, reverse_ddg))
            ddg += (sol.fun+reverse_ddg)/2.0
            logger.debug(sol)
            logger.debug("Current binding free energy improvement {0} for step {1}/{2}".format(ddg, step+1, self.steps))

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
    peturbed_exceptions = Optimize.get_charge_product(sim, peturbed_charges)
    current_exceptions = Optimize.get_charge_product(sim, current_charges)
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
        return binding_free_energy
    else:
        return binding_free_energy


def net_charge_con(current_charge, net_charge):
    sum_ = net_charge
    for charge in current_charge:
        sum_ = sum_ - charge
    return sum_


def rmsd_change_con(current_charge, og_charge):
    maximum_rmsd = 0.05
    rmsd = (np.sum([(x - y) ** 2 for x, y in zip(current_charge, og_charge)])) ** 0.5
    return maximum_rmsd - rmsd


def write_charges(name, charges):
    file = open(name, 'w')
    for q in charges:
        file.write('{}\n'.format(q))
    file.close()


