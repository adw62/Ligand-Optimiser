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

dummy = -8888

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

        #Collect indexs for atoms which have no virtual, therefore replced insitu
        #And idexes for atoms which have virtuals, therfore replace
        assert self.complex_sys[0].virt_atom_order == self.solvent_sys[0].virt_atom_order
        self.all_virt = self.complex_sys[0].virt_atom_order
        self.all_insitu = list(set(self.wt_nonbonded_ids).difference(set(self.all_virt)))
        #Convert to indexing from 0(python) to from 1(mol2)
        self.all_insitu = [x+1 for x in self.all_insitu]
        self.all_virt = [x+1 for x in self.all_virt]

        self.lock_atoms = Optimize.make_lock_list(self, lock_atoms)

        if 'charge' in self.param:
            self.net_charge = Optimize.get_net_charge(self, self.wt_nonbonded)

        # concat all params
        og_charges = [x[0] for x in self.wt_nonbonded]
        self.num_atoms = len(og_charges)
        og_sigma = [x[1] for x in self.wt_nonbonded]
        og_vs_weight = [x[2] for x in self.wt_nonbonded]
        self.og_all_params = og_charges + og_sigma + og_vs_weight

        Optimize.optimize(self, name)

    def make_lock_list(self, lock_atoms):
        # shift from indexed by 1(mol2) to indexed by 0(python)
        lock_atoms = [x - 1 for x in lock_atoms]
        shift = len(self.wt_nonbonded)
        # lock sigmas
        lock2 = [x + shift for x in lock_atoms]
        # lock virtual weights
        lock3 = [x + 2 * shift for x in lock_atoms]
        # lock all wights of atoms with no virtuals by default plus convert back to zero indexing
        lock_v = [x + 2 * shift - 1 for x in self.all_insitu]
        lock_atoms = lock_atoms + lock2 + lock3 + lock_v
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
        vs_weights = params[self.num_atoms*2:self.num_atoms*3]
        atomwise_params = [[x,y,z] for x,y,z in zip(charge, sigma, vs_weights)]

        return atomwise_params

    def translate_atomwise_to_mutant(self, atomwise, exceptions):
        wt_nonbonded = copy.deepcopy(self.wt_parameters[0])
        vs = []
        for new_atom, old_atom in zip(atomwise, wt_nonbonded):
            old_atom['data'] = [new_atom[0], new_atom[1]]
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
                param.extend([dummy])
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

    def run_dynamics(self, all_params):
        mutant = self.process_mutant(all_params)
        # pull out constraints
        const = mutant[1]
        const = [x for x in const if x != dummy]
        # pull out params
        mutant = mutant[0]
        #Gen a empty mutatation as no mutations are occuring
        mutation = gen_mutations_dicts()

        #Build system with arb const weights are unimportant
        FSim.build(self.complex_sys[0], const_prefactors=const, weights=const)
        FSim.build(self.solvent_sys[0], const_prefactors=const, weights=const)

        mutant_params = Mutants([mutant], [mutation], self.complex_sys[0], self.solvent_sys[0])

        #run dynamics on built system passing arb q and sigma, is there a better way to apply that just passing
        #non bonded here?
        self.complex_sys[1] = self.complex_sys[0].run_parallel_dynamics(self.output_folder, 'complex',
                                                                        self.num_frames, self.equi,
                                                                        mutant_params.complex_params[0])
        self.solvent_sys[1] = self.solvent_sys[0].run_parallel_dynamics(self.output_folder, 'solvent',
                                                                        self.num_frames, self.equi,
                                                                        mutant_params.solvent_params[0])

    #Can use a line search effeciently with full FEP if we abandon vs optimization
    #because can use many windows without rebuilding
    def scipy_opt(self):
        og_all_params = self.og_all_params
        all_params = copy.deepcopy(self.og_all_params)
        step = 0
        max_step_size = 0.1
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
                all_params_plus_one = all_params - step_size * norm_const_step
                logger.debug('Computing objective with step size {}...'.format(step_size))
                forward_ddg = objective(all_params_plus_one, all_params, self)
                step_size = step_size * damp
                if count > 20:
                    raise ValueError('Line search failed with ddg {}'.format(forward_ddg))

            #Run some dynamics with new charges
            self.run_dynamics(all_params_plus_one)
            logger.debug('Computing reverse leg of accepted step...')
            reverse_ddg = -1 * objective(all_params, all_params_plus_one, self)
            logger.debug('Forward {} and reverse {} steps'.format(forward_ddg, reverse_ddg))
            ddg += (forward_ddg + reverse_ddg) / 2.0
            logger.debug(
                "Current binding free energy improvement {0} for step {1}/{2}".format(ddg, step + 1, self.steps))
            all_params = all_params_plus_one
            write_charges('params_opt', all_params)
            step += 1

        return list(charges), ddg

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
        mut, mut_vs = Optimize.translate_atomwise_to_mutant(self, atomwise_params, exceptions)

        return [mut+self.unused_params, mut_vs]


def gen_mutations_dicts(add=[], subtract=[], replace=[None], replace_insitu=[None]):
    return {'add': add, 'subtract': subtract, 'replace': replace, 'replace_insitu': replace_insitu}


def objective(peturbed_params, current_params, sim):
    systems = [peturbed_params, current_params]
    mutants_systems = [sim.process_mutant(x) for x in systems]

    #pull out params
    mutants = [x[0] for x in mutants_systems]
    #pull out virtual sites
    virt_sites = [x[1] for x in mutants_systems]

    #remove weight not pertaining to atoms with vitual sites
    current_vs = [x for x in virt_sites[1] if x != dummy]
    peturbed_vs = [x for x in virt_sites[0] if x != dummy]

    #build system with arb weights and vs
    FSim.build(sim.complex_sys[0], const_prefactors=current_vs, weights=peturbed_vs)
    FSim.build(sim.solvent_sys[0], const_prefactors=current_vs, weights=peturbed_vs)

    #in peturbed state all atoms must be replaced insitu except those with virtual sites which are replaced
    mutations = [gen_mutations_dicts(replace=sim.all_virt, replace_insitu=sim.all_insitu)]
    mutations.append(gen_mutations_dicts())

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
    virtual_weights = all_params[-len(sim.wt_nonbonded):]
    virtual_bool = np.array([False for i in range(len(virtual_weights))])
    q_sig_params = all_params[0:2*len(sim.wt_nonbonded)]
    for diff in h:
        binding_free_energy = []
        mutant_parameters = []
        for i in range(len(q_sig_params)):
            # Skip systems which correspond to locked atoms
            if i not in sim.lock_atoms:
                mutant = copy.deepcopy(q_sig_params)
                mutant[i] = mutant[i] + diff
                mutant_parameters.append(np.append(mutant, virtual_bool))
        for i, weight in enumerate(virtual_weights):
            shift = len(q_sig_params)
            i = i+shift
            if i not in sim.lock_atoms:
                mutant = copy.deepcopy(virtual_bool)
                mutant[i-shift] = True
                mutant_parameters.append(np.append(q_sig_params, mutant))

        mutants_systems = [sim.process_mutant(x) for x in mutant_parameters]

        # pull out params
        mutants = [x[0] for x in mutants_systems]
        # pull out virtual sites
        virtual_bool = [x[1] for x in mutants_systems]

        #make reference system
        current_sys, current_vs = sim.process_mutant(all_params)

        #Reduce to only wights which pertain to virtual sites
        current_vs = [x for x in current_vs if x != dummy]
        # add finite diff
        virt_sites = [vsw+diff for vsw in virtual_weights if vsw != dummy]
        assert len(current_vs) == len(virt_sites) == len(sim.all_virt)

        # build system with arb weights and vs
        FSim.build(sim.complex_sys[0], const_prefactors=current_vs, weights=virt_sites)
        FSim.build(sim.solvent_sys[0], const_prefactors=current_vs, weights=virt_sites)

        # convert true and false to mutations, this determines which atoms are going to be on dual or og topo.
        # Also converting to indexing by one to match mol2 file
        replace = [[x+1 for (x, y) in zip(sim.wt_nonbonded_ids,y1) if y] for y1 in virtual_bool]
        virtual_bool = [[not i for i in x] for x in virtual_bool]
        replace_insitu = [[x+1 for (x, y) in zip(sim.wt_nonbonded_ids,y1) if y] for y1 in virtual_bool]

        # Generate dictionaries to discribe mutations
        mutations = [gen_mutations_dicts(replace=x, replace_insitu=y) for (x, y) in zip(replace, replace_insitu)]
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


