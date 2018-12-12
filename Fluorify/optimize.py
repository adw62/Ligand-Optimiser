#!/usr/bin/env python

from .energy import FSim
from simtk import unit
from scipy.optimize import minimize

#CONSTANTS
e = unit.elementary_charges

class Optimize(object):
    def __init__(self, wt_ligand, complex_sys, solvent_sys, num_frames):

        self.complex_sys = complex_sys
        self.solvent_sys = solvent_sys
        self.num_frames = num_frames
        self.wt_parameters = [[x[0] / e] for x in wt_ligand.get_parameters()]
        self.wt_energy_complex = FSim.get_mutant_energy(self.complex_sys[0], [self.wt_parameters], self.complex_sys[1],
                                                        self.complex_sys[2], self.num_frames, True)
        self.wt_energy_solvent = FSim.get_mutant_energy(self.solvent_sys[0], [self.wt_parameters], self.solvent_sys[1],
                                                        self.solvent_sys[2], self.num_frames, True)
        Optimize.optimize(self)

    def optimize(self):
        """optimising ligand charges
        """
        max_change = 0.5
        bnds = [sorted((x[0]-max_change*x[0], x[0]+max_change*x[0])) for x in self.wt_parameters]
        cons = {'type': 'eq', 'fun': constraint}
        sol = minimize(objective, self.wt_parameters, bounds=bnds,
                       args=(self.wt_energy_complex, self.wt_energy_solvent, self.complex_sys, self.solvent_sys, self.num_frames), constraints=cons)


def objective(mutant_parameters, wt_energy_complex, wt_energy_solvent, complex_sys, solvent_sys, num_frames):
    mutant_parameters = [[[x] for x in mutant_parameters]]
    print(mutant_parameters)
    wt_parameters = None
    print('Computing complex potential energies...')
    complex_free_energy = FSim.treat_phase(complex_sys[0], wt_parameters, mutant_parameters,
                                           complex_sys[1], complex_sys[2], num_frames, wt_energy_complex)
    print('Computing solvent potential energies...')
    solvent_free_energy = FSim.treat_phase(solvent_sys[0], wt_parameters, mutant_parameters,
                                           solvent_sys[1], solvent_sys[2], num_frames, wt_energy_solvent)
    binding_free_energy = complex_free_energy[0] - solvent_free_energy[0]
    print('Current binding free energy = ', binding_free_energy)
    return binding_free_energy


def constraint(mutant_parameters):
    # sum = net_charge
    sum_ = 1.0
    for charge in mutant_parameters:
        sum_ = sum_ - charge
    return sum_


