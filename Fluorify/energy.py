#!/usr/bin/env python

import simtk.openmm as mm
from simtk.openmm import app
from simtk import unit
import mdtraj as md
import numpy as np
import copy
from sys import stdout
import math
from functools import partial
from multiprocess import Pool
import logging

from pymbar import MBAR, timeseries

logger = logging.getLogger(__name__)

#CONSTANTS
e = unit.elementary_charges
kB = 0.008314472471220214
T = 300
kT = kB * T # Unit: kJ/mol

class FSim(object):
    def __init__(self, ligand_name, sim_name, input_folder, charge_only, num_gpu,
                 temperature=300*unit.kelvin, friction=0.3/unit.picosecond, timestep=2.0*unit.femtosecond):
        """ A class for creating OpenMM context from input files and calculating free energy
        change when modifying the parameters of the system in the context.

        ligand_name : resname of ligand in input files
        sim_name: complex or solvent
        input_folder: input directory
        """
        self.temperature = temperature
        self.friction = friction
        self.timestep = timestep
        self.num_gpu = num_gpu
        self.charge_only = charge_only
        self.name = sim_name
        #Create system from input files
        sim_dir = input_folder + sim_name + '/'
        snapshot = md.load(sim_dir + sim_name + '.pdb')
        self.pdb = mm.app.pdbfile.PDBFile(sim_dir + sim_name + '.pdb')
        parameters_file_path = sim_dir + sim_name + '.prmtop'
        self.parameters_file = mm.app.AmberPrmtopFile(parameters_file_path)
        system = self.parameters_file.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometers,
                                              constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=0.0005)

        for force_index, force in enumerate(system.getForces()):
            if isinstance(force, mm.NonbondedForce):
                self.nonbonded_index = force_index
            force.setForceGroup(force_index)

        self.wt_system = system
        self.ligand_atoms = get_ligand_atoms(ligand_name, snapshot)

    def run_parallel_fep(self, wt_parameters, mutant_parameters, n_steps, n_iterations, lambdas):
        logger.debug('Computing FEP for {}...'.format(self.name))
        mutant_systems = []
        nstates = len(lambdas)
        if self.charge_only:
            param_diff = [[x[0]-y[0]] for x, y in zip(wt_parameters, mutant_parameters)]
            for lam in lambdas:
                mutant_systems.append([[-x[0]*lam+y[0]] for x, y in zip(param_diff, wt_parameters)])
        else:
            param_diff = [[x[0]-y[0], x[1]-y[1], x[2]-y[2]] for x, y in zip(wt_parameters, mutant_parameters)]
            for lam in lambdas:
                mutant_systems.append([[-x[0]*lam+y[0], -x[1]*lam+y[1], -x[2]*lam+y[2]]
                                       for x, y in zip(param_diff, wt_parameters)])

        chunk = math.ceil(len(mutant_systems) / self.num_gpu)
        groups = grouper(mutant_systems, chunk)
        pool = Pool(processes=self.num_gpu)

        system = copy.deepcopy(self.wt_system)
        box_vectors = self.pdb.topology.getPeriodicBoxVectors()
        system.setDefaultPeriodicBoxVectors(*box_vectors)
        system.addForce(mm.MonteCarloBarostat(1 * unit.atmospheres, 300 * unit.kelvin, 25))###

        fep = partial(run_fep, sim=self, system=system, pdb=self.pdb,
                      n_steps=n_steps, n_iterations=n_iterations, chunk=chunk, all_mutants=mutant_systems)
        u_kln = pool.map(fep, groups)
        pool.close()
        pool.join()
        pool.terminate()

        u_kln = np.vstack(u_kln)
        # Subsample data to extract uncorrelated equilibrium timeseries
        N_k = np.zeros([nstates], np.int32)  # number of uncorrelated samples
        for k in range(nstates):
            [_, g, __] = timeseries.detectEquilibration(u_kln[k, k, :])
            indices = timeseries.subsampleCorrelatedData(u_kln[k, k, :], g=g)
            N_k[k] = len(indices)
            u_kln[k, :, 0:N_k[k]] = u_kln[k, :, indices].T
        # Compute free energy differences and statistical uncertainties
        mbar = MBAR(u_kln, N_k)
        [DeltaF_ij, dDeltaF_ij, _] = mbar.getFreeEnergyDifferences()
        logger.debug("Number of uncorrelated samples per state: {}".format(N_k))
        logger.debug("Relative free energy change for {0} =: {1} +- {2} kT"
              .format(self.name, DeltaF_ij[0, nstates - 1], dDeltaF_ij[0, nstates - 1]))
        return DeltaF_ij[0, nstates - 1]

    def run_parallel_dynamics(self, output_folder, name, n_steps, mutant_parameters):
        system = copy.deepcopy(self.wt_system)
        if mutant_parameters is not None:
            non_bonded_force = system.getForce(self.nonbonded_index)
            self.apply_parameters(non_bonded_force, mutant_parameters)

        box_vectors = self.pdb.topology.getPeriodicBoxVectors()
        system.setDefaultPeriodicBoxVectors(*box_vectors)
        system.addForce(mm.MonteCarloBarostat(1 * unit.atmospheres, 300 * unit.kelvin, 25))###

        dcd_names = [output_folder+name+'_gpu'+str(i)+'.dcd' for i in range(self.num_gpu)]
        groups = grouper(dcd_names, 1)
        pool = Pool(processes=self.num_gpu)
        run = partial(run_dynamics, system=system, pdb=self.pdb, n_steps=math.ceil(n_steps/self.num_gpu),
                      temperature=self.temperature, friction=self.friction, timestep=self.timestep)
        pool.map(run, groups)
        pool.close()
        pool.join()
        pool.terminate()
        return dcd_names

    def build_context(self, system, device):
        integrator = mm.LangevinIntegrator(self.temperature, self.friction, self.timestep)
        integrator.setConstraintTolerance(0.00001)
        platform = mm.Platform.getPlatformByName('CUDA')
        properties = {'CudaPrecision': 'mixed', 'CudaDeviceIndex': device}
        context = mm.Context(system, integrator, platform, properties)
        return context, integrator

    def treat_phase(self, wt_parameters, mutant_parameters, dcd, top, num_frames, wildtype_energy=None):
        if wildtype_energy is None:
            wildtype_energy = FSim.get_mutant_energy(self, [wt_parameters], dcd, top, num_frames, True)
        mutant_energy = FSim.get_mutant_energy(self, mutant_parameters, dcd, top, num_frames)
        phase_free_energy = get_free_energy(mutant_energy, wildtype_energy[0])
        return phase_free_energy

    def apply_parameters(self, force, mutant_parameters, write_charges=False):
        f = open('charges.out', 'w')
        for i, atom_idx in enumerate(self.ligand_atoms):
            index = int(atom_idx)
            charge, sigma, epsilon = force.getParticleParameters(index)
            if self.charge_only:
                force.setParticleParameters(index, mutant_parameters[i][0], sigma, epsilon)
            else:
                force.setParticleParameters(index, mutant_parameters[i][0],
                                            mutant_parameters[i][1], mutant_parameters[i][2])
            if write_charges:
                f.write('{0}    {1}\n'.format(charge/e, mutant_parameters[i][0]/e))
        f.close()

    def get_mutant_energy(self, parameters, dcd, top, num_frames, wt=False):
        chunk = math.ceil(len(parameters)/self.num_gpu)
        if wt:
            groups = [[parameters, '0']]
        else:
            groups = grouper(parameters, chunk)
        pool = Pool(processes=self.num_gpu)
        mutant_eng = partial(mutant_energy, sim=self, dcd=dcd, top=top, num_frames=num_frames,
                            chunk=chunk, total_mut=len(parameters), wt=wt)
        mutants_systems_energies = pool.map(mutant_eng, groups)
        pool.close()
        pool.join()
        pool.terminate()
        mutants_systems_energies = [x for y in mutants_systems_energies for x in y]
        return mutants_systems_energies


def mutant_energy(parameters, sim, dcd, top, num_frames, chunk, total_mut, wt):
    mutants_systems_energies = []
    kj_m = unit.kilojoule_per_mole
    top = md.load(top).topology
    device = parameters[1]
    parameters = parameters[0]
    nonbonded_index = sim.nonbonded_index
    system = copy.deepcopy(sim.wt_system)
    context, integrator = sim.build_context(system, device=device)
    del integrator
    for index, mutant_parameters in enumerate(parameters):
        mutant_num = ((index+1)+(chunk*int(device)))
        if wt:
           logger.debug('Computing potential for wild type ligand')
        else:
            logger.debug('Computing potential for mutant {0}/{1} on GPU {2}'.format(mutant_num, total_mut, device))
        force = system.getForce(nonbonded_index)
        sim.apply_parameters(force, mutant_parameters)
        force.updateParametersInContext(context)

        mutant_energies = []
        append = mutant_energies.append
        for frame in frames(dcd, top, maxframes=num_frames):
            context.setPositions(frame.xyz[0])
            context.setPeriodicBoxVectors(frame.unitcell_vectors[0][0],
                                          frame.unitcell_vectors[0][1], frame.unitcell_vectors[0][2])
            energy = context.getState(getEnergy=True, groups={nonbonded_index}).getPotentialEnergy()
            append(energy / kj_m)
        mutants_systems_energies.append(mutant_energies)
    return mutants_systems_energies


def get_ligand_atoms(ligand_name, snapshot):
    ligand_atoms = snapshot.topology.select('resname {}'.format(ligand_name))
    if len(ligand_atoms) == 0:
        raise ValueError('Did not find ligand in supplied topology by name {}'.format(ligand_name))
    return ligand_atoms


def grouper(list_to_distribute, chunk):
    groups = []
    for i in range(int(math.ceil(len(list_to_distribute)/chunk))):
        group = []
        for j in range(chunk):
            try:
                group.append(list_to_distribute[j+(i*chunk)])
            except IndexError:
                pass
        groups.append(group)
    groups = [[groups[i], str(i)] for i in range(len(groups))]
    return groups


def frames(dcd, top, maxframes):
    maxframes = math.ceil(maxframes/len(dcd))
    for name in dcd:
        for i in range(maxframes):
            frame = md.load_dcd(name, top=top, stride=None, atom_indices=None, frame=i)
            yield frame


def run_fep(parameters, sim, system, pdb, n_steps, n_iterations, chunk, all_mutants):
    device = parameters[1]
    parameters = parameters[0]
    nonbonded_index = sim.nonbonded_index
    context, integrator = sim.build_context(system, device)
    context.setPositions(pdb.positions)
    logger.debug('Minimizing..')
    mm.LocalEnergyMinimizer.minimize(context)
    temperature = 300 * unit.kelvin
    context.setVelocitiesToTemperature(temperature)
    total_states = len(all_mutants)
    nstates = len(parameters)
    u_kln = np.zeros([nstates, total_states, n_iterations], np.float64)
    test = unit.AVOGADRO_CONSTANT_NA * unit.BOLTZMANN_CONSTANT_kB * temperature
    force = system.getForce(nonbonded_index)
    for k, local_mutant in enumerate(parameters):
        window = ((k+1)+(chunk*int(device)))
        logger.debug('Computing potentials for FEP window {0}/{1} on GPU {2}'.format(window, total_states, device))
        for iteration in range(n_iterations):
            sim.apply_parameters(force, local_mutant)
            force.updateParametersInContext(context)
            # Run some dynamics
            integrator.step(n_steps)
            # Compute energies at all alchemical states
            for l, global_mutant in enumerate(all_mutants):
                sim.apply_parameters(force, global_mutant)
                force.updateParametersInContext(context)
                u_kln[k, l, iteration] = context.getState(getEnergy=True, groups={nonbonded_index}).getPotentialEnergy() / test
    return u_kln

def run_dynamics(dcd_name, system, pdb, n_steps, temperature, friction, timestep):
    """
    Given an OpenMM Context object and options, perform molecular dynamics
    calculations.

    Parameters
    ----------
    context : an OpenMM context instance
    n_steps : the number of iterations for the sim

    Returns
    -------

    """
    integrator = mm.LangevinIntegrator(temperature, friction, timestep)
    integrator.setConstraintTolerance(0.00001)

    platform = mm.Platform.getPlatformByName('CUDA')
    device = dcd_name[1]
    dcd_name = dcd_name[0][0]
    properties = {'CudaPrecision': 'mixed', 'CudaDeviceIndex': device} #dcd_name
    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setPositions(pdb.positions)

    logger.debug('Minimizing..')
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature)
    logger.debug('Equilibrating...')
    equi = 250000
    simulation.step(equi)

    simulation.reporters.append(app.DCDReporter(dcd_name, 2500))
    simulation.reporters.append(app.StateDataReporter(stdout, 2500, step=True,
    potentialEnergy=True, temperature=True, progress=True, remainingTime=True,
    speed=True, totalSteps=equi+n_steps, separator='\t'))

    logger.debug('Running Production...')
    simulation.step(n_steps)
    logger.debug('Done!')


def get_free_energy(mutant_energy, wildtype_energy):
    ans = []
    free_energy = []
    for ligand in mutant_energy:
        tmp = 0.0
        for i in range(len(ligand)):
            tmp += (np.exp(-(ligand[i] - wildtype_energy[i]) / kT))
        ans.append(tmp / len(mutant_energy))

    for ligand in ans:
        free_energy.append(-kT * np.log(ligand) * 0.239) # Unit: kcal/mol
    return free_energy

