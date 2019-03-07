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
                 charge_only):

        self.complex_sys = complex_sys
        self.solvent_sys = solvent_sys
        self.num_frames = num_frames
        self.equi = equi
        self.steps = steps
        self.output_folder = output_folder
        self.charge_only = charge_only
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
            logger.debug('Original charges: {}'.format([x[0] for x in self.wt_parameters]))
            logger.debug('Optimized charges: {}'.format(opt_charges))
            opt_charges = [[x] for x in opt_charges]
            fep = True
            lambdas = np.linspace(0.0, 1.0, 10)
            if fep:
                complex_dg = self.complex_sys[0].run_parallel_fep(self.wt_parameters, opt_charges, 20000, 50, lambdas)
                solvent_dg = self.solvent_sys[0].run_parallel_fep(self.wt_parameters, opt_charges, 20000, 50, lambdas)
                ddg_fep = complex_dg - solvent_dg
                logger.debug('ddG Fluorine Scanning = {}'.format(ddg_fs))
                logger.debug('ddG FEP = {}'.format(ddg_fep))
        else:
            raise ValueError('No other optimizers implemented')

    def scipy_opt(self):
        cons = {'type': 'eq', 'fun': constraint, 'args': [self.net_charge]}
        charges = [x[0] for x in self.wt_nonbonded]
        ddg = 0.0
        for step in range(self.steps):
            max_change = 0.1
            bnds = [sorted((x - max_change * x, x + max_change * x)) for x in charges]
            sol = minimize(objective, charges, bounds=bnds, options={'maxiter': 1}, jac=gradient,
                           args=(charges, self), constraints=cons)
            prev_charges = charges
            charges = [[x] for x in sol.x]

            #run new dynamics with updated charges
            self.complex_sys[1] = self.complex_sys[0].run_parallel_dynamics(self.output_folder, 'complex_step'+str(step),
                                                                            self.num_frames*2500, self.equi, [charges])
            self.solvent_sys[1] = self.solvent_sys[0].run_parallel_dynamics(self.output_folder, 'solvent_step'+str(step),
                                                                            self.num_frames*2500, self.equi, [charges])
            prev_charges = [x[0] for x in prev_charges]
            logger.debug('Computing reverse leg of accepted step...')
            reverse_ddg = -1*objective(prev_charges, charges, self.complex_sys, self.solvent_sys, self.num_frames)
            # NOT WORKING !!!
            if abs(sol.fun) >= 1.5*abs(reverse_ddg) or abs(sol.fun) <= 0.6666*abs(reverse_ddg):
                logger.debug('Forward {} and reverse {} are not well agreed. Need more sampling'.format(sol.fun, reverse_ddg))
            ddg += (sol.fun+reverse_ddg)/2.0
            logger.debug(sol)
            logger.debug("Current binding free energy improvement {0} for step {1}/{2}".format(ddg, step+1, self.steps))

        return sol.x, ddg

def build_opt_params(charges, sys_exception_params, sim):
    #reorder exceptions
    exception_order = [sim.complex_sys[0].exceptions_list, sim.solvent_sys[0].exceptions_list]
    offset = [sim.complex_sys[0].offset, sim.solvent_sys[0].offset]
    exception_params = [[], []]
    for i, (sys_excep_order, sys_offset) in enumerate(zip(exception_order, offset)):
        map = {x['id']: x for x in sys_exception_params}
        exception_params[i] = [map[frozenset(int(x-sys_offset) for x in atom)] for atom in sys_excep_order]
    # add ids back in to charges
    charges = [{'id': x, 'data': y} for x, y in zip(sim.wt_nonbonded_ids, charges)]
    #[[wild_type], [complex]], None id for ghost which is not used in optimization
    complex_parameters = [charges, None, exception_params[0], None]
    solvent_parameters = [charges, None, exception_params[1], None]
    return complex_parameters, solvent_parameters

def objective(peturbed_charges, current_charges, sim):
    peturbed_exceptions = Optimize.get_charge_product(sim, peturbed_charges)
    current_exceptions = Optimize.get_charge_product(sim, current_charges)
    com_mut_param, sol_mut_param = build_opt_params(peturbed_charges, peturbed_exceptions, sim)
    com_wt_param, sol_wt_param = build_opt_params(current_charges, current_exceptions, sim)
    com_mut_param.append(com_wt_param)
    sol_mut_param.append(sol_wt_param)
    logger.debug('Computing Objective...')
    complex_free_energy = FSim.treat_phase(sim.complex_sys[0], [com_mut_param],
                                           sim.complex_sys[1], sim.complex_sys[2], sim.num_frames)
    solvent_free_energy = FSim.treat_phase(sim.solvent_sys[0], [sol_mut_param],
                                           sim.solvent_sys[1], sim.solvent_sys[2], sim.num_frames)
    binding_free_energy = complex_free_energy[0] - solvent_free_energy[0]
    return binding_free_energy/unit.kilocalories_per_mole


def gradient(mutant_parameters, wt_parameters, complex_sys, solvent_sys, num_frames):
    num_frames = int(num_frames/2)
    binding_free_energy = []
    og_mutant_parameters = mutant_parameters
    mutant_parameters = []
    h = 1.5e-04
    for i in range(len(og_mutant_parameters)):
        mutant = copy.deepcopy(og_mutant_parameters)
        mutant[i] = mutant[i] + h
        mutant = [[x] for x in mutant]
        mutant_parameters.append([mutant])

    logger.debug('Computing Jacobian...')
    complex_free_energy = FSim.treat_phase(complex_sys[0], [[wt_parameters]], mutant_parameters,
                                           complex_sys[1], complex_sys[2], num_frames)
    solvent_free_energy = FSim.treat_phase(solvent_sys[0], [[wt_parameters]], mutant_parameters,
                                           solvent_sys[1], solvent_sys[2], num_frames)
    for sol, com in zip(solvent_free_energy, complex_free_energy):
        free_energy = com - sol
        free_energy = free_energy/h
        binding_free_energy.append(free_energy/unit.kilocalories_per_mole)
    return binding_free_energy


def constraint(mutant_parameters, net_charge):
    sum_ = net_charge
    for charge in mutant_parameters:
        sum_ = sum_ - charge
    return sum_


