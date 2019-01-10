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

# CONSTANTS #
kB = 0.008314472471220214 * unit.kilojoules_per_mole/unit.kelvin

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
        self.kT = kB * self.temperature
        kcal = 4.1868 * unit.kilojoules_per_mole
        self.kTtokcal = self.kT/kcal * unit.kilocalories_per_mole
        #Create system from input files
        self.sim_dir = input_folder + sim_name + '/'
        self.sim_name = sim_name
        snapshot = md.load(self.sim_dir + self.sim_name + '.pdb')
        self.pdb = mm.app.pdbfile.PDBFile(self.sim_dir + sim_name + '.pdb')
        parameters_file_path = self.sim_dir + self.sim_name + '.prmtop'
        self.parameters_file = mm.app.AmberPrmtopFile(parameters_file_path)
        system = self.parameters_file.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometers,
                                              constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=0.0005)


        for force_index, force in enumerate(system.getForces()):
            if isinstance(force, mm.NonbondedForce):
                self.nonbonded_index = force_index
            if isinstance(force, mm.HarmonicBondForce):
                bond_force = force
                self.harmonic_index = force_index
            force.setForceGroup(force_index)


        self.add_all_virtuals(system, snapshot, ligand_name)
        self.wt_system = system
        self.ligand_atoms, self.bond_list, self.constraint_list = get_ligand_info(ligand_name,
                                                                                  snapshot, bond_force, system)

    def add_all_virtuals(self, system, snapshot, ligand_name):
        import time
        bond_list = []
        carbons = list(snapshot.topology.select('element C and resname {}'.format(ligand_name)))
        hydrogens = list(snapshot.topology.select('element H and resname {}'.format(ligand_name)))
        for index in range(system.getNumConstraints()):
            i, j, r = system.getConstraintParameters(index)
            if i in carbons and j in hydrogens:
                bond_list.append([i, j])
            if j in carbons and i in hydrogens:
                bond_list.append([j, i])

        names = ['C', 'H']
        for bond in bond_list:
            for atom, name in zip(bond, names):
                print(name)
                print('x: %s\ty: %s\tz: %s' % tuple(snapshot.xyz[0, atom,:]))

    def run_parallel_fep(self, wt_parameters, mutant_parameters, offset, n_steps, n_iterations, lambdas):
        wt_nonbonded = wt_parameters[0]
        mutant_nonbonded = mutant_parameters[0]
        wt_bonded = wt_parameters[1]
        mutant_bonded = mutant_parameters[1]

        wt_bonded, mutant_bonded = self.reorder_mutant_bonds(wt_bonded, mutant_bonded, offset)

        logger.debug('Computing FEP for {}...'.format(self.name))
        nonbonded_mutant_systems = []
        bonded_mutant_systems = []


        if self.charge_only:
            #nonbonded
            param_diff = [[x[0]-y[0]] for x, y in zip(wt_nonbonded, mutant_nonbonded)]
            for lam in lambdas:
                nonbonded_mutant_systems.append([[-x[0]*lam+y[0]] for x, y in zip(param_diff, wt_nonbonded)])
        else:
            #nonbonded
            param_diff = [[x[0]-y[0], x[1]-y[1], x[2]-y[2]] for x, y in zip(wt_nonbonded, mutant_nonbonded)]
            for lam in lambdas:
                nonbonded_mutant_systems.append([[-x[0]*lam+y[0], -x[1]*lam+y[1], -x[2]*lam+y[2]]
                                       for x, y in zip(param_diff, wt_nonbonded)])
        #bonds
        param_diff = [[x[3]-y[3], x[4]-y[4]] for x, y in zip(wt_bonded, mutant_bonded)]
        for lam in lambdas:
            bonded_mutant_systems.append([[y[0], y[1], y[2], -x[0]*lam+y[3], -x[1]*lam+y[4]]
                                          for x, y in zip(param_diff, wt_bonded)])

        mutant_systems = [[x, y] for x, y in zip(nonbonded_mutant_systems, bonded_mutant_systems)]
        nstates = len(mutant_systems)
        chunk = math.ceil(len(mutant_systems) / self.num_gpu)
        groups = grouper(mutant_systems, chunk)
        pool = Pool(processes=self.num_gpu)

        system = copy.deepcopy(self.wt_system)
        box_vectors = self.pdb.topology.getPeriodicBoxVectors()
        system.setDefaultPeriodicBoxVectors(*box_vectors)
        system.addForce(mm.MonteCarloBarostat(1 * unit.atmospheres, self.temperature * unit.kelvin, 25))###

        fep = partial(run_fep, sim=self, system=system, pdb=self.pdb,
                      n_steps=n_steps, n_iterations=n_iterations, chunk=chunk, all_mutants=mutant_systems)
        u_kln = pool.map(fep, groups)
        pool.close()
        pool.join()
        pool.terminate()
        ddg = FSim.gather_dg(self, u_kln, nstates)

        return ddg

    def reorder_mutant_bonds(self, wt_bond, mutant_bond, offset):
        """
        reorders harmonic bonds so they match system and removes harmonic bonds associated with any transformations
        from hydrogen to another atom as these bonds are represented as constraints in the system and applying harmonic
        parameters to these will not work.
        :param wt_bond: harmonic parameters of wt system
        :param mutant_bond: harmonic parameters of mutant system
        :param offset: offset in atom indexing between input systems
        :return: updated bond lists
        """
        new_bonds = []
        for bonds in [wt_bond, mutant_bond]:
            tmp = []
            for bond_i in self.bond_list:
                for bond_j in bonds:
                    if bond_i[1]-offset == bond_j[1] and bond_i[2]-offset == bond_j[2]:
                        tmp.append([bond_i[0], bond_i[1], bond_i[2], bond_j[3], bond_j[4]])
            new_bonds.append(tmp)
        return tuple(new_bonds)

    def gather_dg(self, u_kln, nstates):
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
        logger.debug("Relative free energy change for {0} = {1} +- {2}"
              .format(self.name, DeltaF_ij[0, nstates - 1]*self.kTtokcal, dDeltaF_ij[0, nstates - 1]*self.kTtokcal))

        return DeltaF_ij[0, nstates - 1]*self.kTtokcal

    def run_parallel_dynamics(self, output_folder, name, n_steps, equi, mutant_parameters):
        system = copy.deepcopy(self.wt_system)

        if mutant_parameters is not None:
            non_bonded_force = system.getForce(self.nonbonded_index)
            self.apply_nonbonded_parameters(non_bonded_force, mutant_parameters)

        box_vectors = self.pdb.topology.getPeriodicBoxVectors()
        system.setDefaultPeriodicBoxVectors(*box_vectors)
        system.addForce(mm.MonteCarloBarostat(1 * unit.atmospheres, self.temperature * unit.kelvin, 25))###

        dcd_names = [output_folder+name+'_gpu'+str(i)+'.dcd' for i in range(self.num_gpu)]
        groups = grouper(dcd_names, 1)
        pool = Pool(processes=self.num_gpu)
        run = partial(run_dynamics, system=system, pdb=self.pdb, n_steps=math.ceil(n_steps/self.num_gpu),
                      equi=equi, temperature=self.temperature, friction=self.friction, timestep=self.timestep)
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
        phase_free_energy = FSim.get_free_energy(self, mutant_energy, wildtype_energy[0])
        return phase_free_energy

    def apply_constraint_parameters(self, system, mutant_parameters):
        for constraint in mutant_parameters:
            system.setConstraintParameters(*constraint)

    def apply_bonded_parameters(self, force, mutant_parameters):
        for bond in mutant_parameters:
            force.setBondParameters(*bond)

    def apply_nonbonded_parameters(self, force, mutant_parameters):
        for i, atom_idx in enumerate(self.ligand_atoms):
            index = int(atom_idx)
            charge, sigma, epsilon = force.getParticleParameters(index)
            if self.charge_only:
                force.setParticleParameters(index, mutant_parameters[i][0], sigma, epsilon)
            else:
                force.setParticleParameters(index, mutant_parameters[i][0],
                                            mutant_parameters[i][1], mutant_parameters[i][2])

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

    def get_free_energy(self, mutant_energy, wildtype_energy):
        ans = []
        free_energy = []
        for ligand in mutant_energy:
            tmp = 0.0
            for i in range(len(ligand)):
                tmp += (np.exp(-(ligand[i] - wildtype_energy[i]) / self.kT))
            ans.append(tmp / len(mutant_energy))

        for ligand in ans:
            free_energy.append(-np.log(ligand) * self.kTtokcal)
        return free_energy


def mutant_energy(parameters, sim, dcd, top, num_frames, chunk, total_mut, wt):
    mutants_systems_energies = []
    top = md.load(top).topology
    device = parameters[1]
    parameters = parameters[0]
    nonbonded_index = sim.nonbonded_index
    system = copy.deepcopy(sim.wt_system)
    context, integrator = sim.build_context(system, device=device)
    del integrator
    force = system.getForce(nonbonded_index)
    for index, mutant_parameters in enumerate(parameters):
        mutant_num = ((index+1)+(chunk*int(device)))
        if wt:
           logger.debug('Computing potential for wild type ligand')
        else:
            logger.debug('Computing potential for mutant {0}/{1} on GPU {2}'.format(mutant_num, total_mut, device))
        sim.apply_nonbonded_parameters(force, mutant_parameters)
        force.updateParametersInContext(context)
        mutant_energies = []
        append = mutant_energies.append
        for frame in frames(dcd, top, maxframes=num_frames):
            context.setPositions(frame.xyz[0])
            context.setPeriodicBoxVectors(frame.unitcell_vectors[0][0],
                                          frame.unitcell_vectors[0][1], frame.unitcell_vectors[0][2])
            energy = context.getState(getEnergy=True, groups={nonbonded_index}).getPotentialEnergy()
            append(energy)
        mutants_systems_energies.append(mutant_energies)
    return mutants_systems_energies


def run_fep(parameters, sim, system, pdb, n_steps, n_iterations, chunk, all_mutants):
    device = parameters[1]
    parameters = parameters[0]
    context, integrator = sim.build_context(system, device)
    context.setPositions(pdb.positions)
    logger.debug('Minimizing...')
    mm.LocalEnergyMinimizer.minimize(context)
    temperature = sim.temperature
    context.setVelocitiesToTemperature(temperature)
    total_states = len(all_mutants)
    nstates = len(parameters)
    u_kln = np.zeros([nstates, total_states, n_iterations], np.float64)
    nonbonded_force = system.getForce(sim.nonbonded_index)
    bonded_force = system.getForce(sim.harmonic_index)
    for k, local_mutant in enumerate(parameters):
        window = ((k+1)+(chunk*int(device)))
        logger.debug('Computing potentials for FEP window {0}/{1} on GPU {2}'.format(window, total_states, device))
        for iteration in range(n_iterations):
            sim.apply_nonbonded_parameters(nonbonded_force, local_mutant[0])
            nonbonded_force.updateParametersInContext(context)
            sim.apply_bonded_parameters(bonded_force, local_mutant[1])
            bonded_force.updateParametersInContext(context)
            # Run some dynamics
            integrator.step(n_steps)
            # Compute energies at all alchemical states
            for l, global_mutant in enumerate(all_mutants):
                sim.apply_nonbonded_parameters(nonbonded_force, global_mutant[0])
                nonbonded_force.updateParametersInContext(context)
                sim.apply_bonded_parameters(bonded_force, global_mutant[1])
                bonded_force.updateParametersInContext(context)
                u_kln[k, l, iteration] = context.getState(getEnergy=True,
                        groups={sim.nonbonded_index, sim.harmonic_index}).getPotentialEnergy() / sim.kT
    return u_kln


def run_dynamics(dcd_name, system, pdb, n_steps, equi, temperature, friction, timestep):
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

    logger.debug('Minimizing...')
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature)
    logger.debug('Equilibrating...')
    simulation.step(equi)

    simulation.reporters.append(app.DCDReporter(dcd_name, 2500))
    simulation.reporters.append(app.StateDataReporter(stdout, 2500, step=True,
    potentialEnergy=True, temperature=True, progress=True, remainingTime=True,
    speed=True, totalSteps=equi+n_steps, separator='\t'))

    logger.debug('Running Production...')
    simulation.step(n_steps)
    logger.debug('Done!')


def get_ligand_info(ligand_name, snapshot, force, system):
    ligand_atoms = snapshot.topology.select('resname {}'.format(ligand_name))
    if len(ligand_atoms) == 0:
        raise ValueError('Did not find ligand in supplied topology by name {}'.format(ligand_name))
    bond_list = list()
    for bond_index in range(force.getNumBonds()):
        particle1, particle2, r, k = force.getBondParameters(bond_index)
        if set([particle1, particle2]).intersection(ligand_atoms):
            bond_list.append([bond_index, particle1, particle2, r, k])
    constraint_list = list()
    for constraint_index in range(system.getNumConstraints()):
        particle1, particle2, r = system.getConstraintParameters(constraint_index)
        if set([particle1, particle2]).intersection(ligand_atoms):
            constraint_list.append([constraint_index, particle1, particle2, r])
    return ligand_atoms, bond_list, constraint_list


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

