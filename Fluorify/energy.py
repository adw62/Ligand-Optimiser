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

from .mutants import Mutants

from pymbar import MBAR, timeseries

logger = logging.getLogger(__name__)

# CONSTANTS #
kB = 0.008314472471220214 * unit.kilojoules_per_mole/unit.kelvin

class FSim(object):
    def __init__(self, ligand_name, sim_name, input_folder, charge_only, num_gpu, offset, opt, temperature=300*unit.kelvin,
                 friction=0.3/unit.picosecond, timestep=2.0*unit.femtosecond):
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
        self.offset = int(offset)
        self.opt = opt
        self.charge_only = charge_only
        self.name = sim_name
        self.kT = kB * self.temperature
        kcal = 4.1868 * unit.kilojoules_per_mole
        self.kTtokcal = self.kT/kcal * unit.kilocalories_per_mole
        #Create system from input files
        sim_dir = input_folder + sim_name + '/'
        sim_name = sim_name
        snapshot = md.load(sim_dir + sim_name + '.pdb')
        self.input_pdb = mm.app.pdbfile.PDBFile(sim_dir + sim_name + '.pdb')
        parameters_file_path = sim_dir + sim_name + '.prmtop'
        self.parameters_file = mm.app.AmberPrmtopFile(parameters_file_path)
        system = self.parameters_file.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometers,
                                              constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=0.0005)

        #seperate forces into sperate groups
        for force_index, force in enumerate(system.getForces()):
            if isinstance(force, mm.NonbondedForce):
                nonbonded_force = force
                self.nonbonded_index = force_index
            if isinstance(force, mm.HarmonicBondForce):
                bond_force = force
                self.harmonic_index = force_index
            if isinstance(force, mm.PeriodicTorsionForce):
                torsion_force = force
                self.torsion_index = force_index
            force.setForceGroup(force_index)

        self.ligand_info, self.exceptions_list, self.bond_list,\
        self.torsion_list = get_ligand_info(ligand_name, snapshot, nonbonded_force, bond_force, torsion_force, system)

        self.virt_atom_shift = nonbonded_force.getNumParticles()
        #Add dual topology for fluorination
        if not self.opt:
            self.extended_pos, self.extended_top, self.virt_atom_order, self.h_virt_excep,\
            self.virt_excep_shift, self.zero_exceptions, self.ghost_ligand_info = self.add_all_virtuals(system, nonbonded_force, snapshot, ligand_name)
            f = open(sim_dir + sim_name + '.pdb', 'w')
            mm.app.pdbfile.PDBFile.writeFile(self.extended_top, self.extended_pos, f)
            f.close()

        self.extended_pdb = mm.app.pdbfile.PDBFile(sim_dir + sim_name + '.pdb')
        self.wt_system = system

    def add_all_virtuals(self, system, nonbonded_force, snapshot, ligand_name):
        pos = list(snapshot.xyz[0]*10)
        top = self.input_pdb.topology

        bond_list = []
        hydrogen_order = []
        carbons = list(snapshot.topology.select('element C and resname {}'.format(ligand_name)))
        hydrogens = list(snapshot.topology.select('element H and resname {}'.format(ligand_name)))
        for index in range(system.getNumConstraints()):
            i, j, r = system.getConstraintParameters(index)
            if i in carbons and j in hydrogens:
                bond_list.append([j, i])
                hydrogen_order.append(j-self.offset)
            if j in carbons and i in hydrogens:
                bond_list.append([i, j])
                hydrogen_order.append(i-self.offset)

        hydrogen_order = sorted(hydrogen_order)
        bond_list = sorted(bond_list)

        element = app.element.fluorine
        chain = top.addChain()
        res = top.addResidue('FLU', chain)
        f_weight = 0.242 #1.363/1.097 Ang
        f_charge = -0.2463
        f_sig = 0.3034222854639816
        f_eps = 0.3481087995050717

        ligand_ghost_atoms = []
        ligand_ghost_exceptions = []

        h_exceptions = []
        f_exceptions = []
        for new_atom in bond_list:
            exceptions = []
            system.addParticle(0.00)
            x, y, z = tuple(snapshot.xyz[0, new_atom[0], :]*10)
            pos.extend([[x, y, z]])
            atom_added = nonbonded_force.addParticle(0.0, 1.0, 0.0)
            ligand_ghost_atoms.append(atom_added)
            vs = mm.TwoParticleAverageSite(new_atom[0], new_atom[1], 1+f_weight, -f_weight)
            system.setVirtualSite(atom_added, vs)
            #If ligand is over 1000 atoms there will be repeated names
            top.addAtom('F{}'.format(abs(new_atom[0]) % 1000), element, res)
            #here the fluorine will inherited the exception of its parent hydrogen
            for exception_index in range(nonbonded_force.getNumExceptions()):
                [iatom, jatom, chargeprod, sigma, epsilon] = nonbonded_force.getExceptionParameters(exception_index)
                if jatom == new_atom[0]:
                    if iatom in self.ligand_info[0]:
                        h_exceptions.append([iatom, jatom])
                        exceptions.append([iatom, atom_added, 0.1, 0.1, 0.1])
                if iatom == new_atom[0]:
                    if jatom in self.ligand_info[0]:
                        h_exceptions.append([jatom, iatom])
                        exceptions.append([jatom, atom_added, 0.1, 0.1, 0.1])
            for i, exception in enumerate(exceptions):
                idx = nonbonded_force.addException(*exception)
                f_exceptions.append([idx, exception[0], exception[1], 0.0, 0.1, 0.0])
                ligand_ghost_exceptions.append(idx)

            nonbonded_force.addException(atom_added, new_atom[0], 0.0, 0.1, 0.0, False)

        virt_excep_shift = [[x[0], y[2]-x[1]] for x, y in zip(h_exceptions, f_exceptions)]
        h_virt_excep = [frozenset((x[0], x[1])) for x in h_exceptions]

        return pos, top, hydrogen_order, h_virt_excep, virt_excep_shift, f_exceptions, [ligand_ghost_atoms, ligand_ghost_exceptions]

    def run_parallel_fep(self, mutant_params, system_idx, mutant_idx, n_steps, n_iterations, windows):
        logger.debug('Computing FEP for {}...'.format(self.name))
        mutant_systems = mutant_params.build_fep_systems(system_idx, mutant_idx, windows,
                                                       opt=self.opt, charge_only=self.charge_only)

        nstates = len(mutant_systems)
        chunk = math.ceil(nstates / self.num_gpu)
        groups = grouper(range(nstates), chunk)
        pool = Pool(processes=self.num_gpu)

        system = copy.deepcopy(self.wt_system)
        box_vectors = self.input_pdb.topology.getPeriodicBoxVectors()
        system.setDefaultPeriodicBoxVectors(*box_vectors)
        system.addForce(mm.MonteCarloBarostat(1 * unit.atmospheres, self.temperature * unit.kelvin, 25))###

        fep = partial(run_fep, sim=self, system=system, pdb=self.extended_pdb,
                      n_steps=n_steps, n_iterations=n_iterations, all_mutants=mutant_systems)
        u_kln = pool.map(fep, groups)
        pool.close()
        pool.join()
        pool.terminate()
        ddg = FSim.gather_dg(self, u_kln, nstates)

        return ddg

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

        return DeltaF_ij[0, nstates - 1]*self.kTtokcal, dDeltaF_ij[0, nstates - 1]*self.kTtokcal

    def run_parallel_dynamics(self, output_folder, name, n_steps, equi, mutant_parameters):
        system = copy.deepcopy(self.wt_system)

        if mutant_parameters is not None:
            non_bonded_force = system.getForce(self.nonbonded_index)
            self.apply_nonbonded_parameters(non_bonded_force, mutant_parameters)

        box_vectors = self.input_pdb.topology.getPeriodicBoxVectors()
        system.setDefaultPeriodicBoxVectors(*box_vectors)
        system.addForce(mm.MonteCarloBarostat(1 * unit.atmospheres, self.temperature * unit.kelvin, 25))###

        dcd_names = [output_folder+name+'_gpu'+str(i)+'.dcd' for i in range(self.num_gpu)]
        groups = grouper(dcd_names, 1)
        pool = Pool(processes=self.num_gpu)
        run = partial(run_dynamics, system=system, sim=self, equi=equi, n_steps=math.ceil(n_steps/self.num_gpu))
        pool.map(run, groups)
        pool.close()
        pool.join()
        pool.terminate()
        return dcd_names

    def apply_bonded_parameters(self, force, mutant_parameters):
        for bond_idx, particle_idxs, bond in zip(self.ligand_info[2], self.bond_list, mutant_parameters):
            if frozenset((particle_idxs[0]-self.offset, particle_idxs[1]-self.offset)) != bond['id']:
                raise (ValueError('Fluorify has failed to generate bonds correctly please raise '
                                  'this as and issue at https://github.com/adw62/Fluorify'))
            data = bond['data']
            force.setBondParameters(bond_idx, particle_idxs[0], particle_idxs[1], data[0], data[1])

    def apply_torsion_parameters(self, force, mutant_parameters):
        for torsion_idx, particle_idxs, torsion in zip(self.ligand_info[3], self.torsion_list, mutant_parameters):
            particle_idxs = [x-self.offset for x in particle_idxs]
            if frozenset((particle_idxs[0], particle_idxs[1], particle_idxs[2], particle_idxs[3])) != torsion['id']:
                raise (ValueError('Fluorify has failed to generate torsions correctly please raise '
                                  'this as and issue at https://github.com/adw62/Fluorify'))
            data = torsion['data']
            force.setTorsionParameters(torsion_idx, particle_idxs[0], particle_idxs[1], particle_idxs[2], particle_idxs[3],
                                       data[0], data[1], data[2])

    def apply_nonbonded_parameters(self, force, params, ghost_params, excep, ghost_excep):
        if self.opt:
            ghost_params = []
            ghost_excep = []
        #nonbonded
        for i, index in enumerate(self.ligand_info[0]):
            atom = int(index)
            nonbonded_params = params[i]['data']
            if atom != params[i]['id']+self.offset:
                raise (ValueError('Fluorify has failed to generate nonbonded parameters(0) correctly please raise '
                                  'this as and issue at https://github.com/adw62/Fluorify'))
            if self.charge_only:
                charge, sigma, epsilon = force.getParticleParameters(atom)
                force.setParticleParameters(atom, nonbonded_params[0], sigma, epsilon)
            else:
                if self.opt:
                    raise ValueError('VDW currently not supported for optimization')
                force.setParticleParameters(atom, nonbonded_params[0], nonbonded_params[1], nonbonded_params[2])

        for i, index in enumerate(self.ghost_ligand_info[0]):
            atom = int(index)
            nonbonded_params = ghost_params[i]['data']
            if atom != int(ghost_params[i]['id']):
                raise (ValueError('Fluorify has failed to generate nonbonded parameters(1) correctly please raise '
                                  'this as and issue at https://github.com/adw62/Fluorify'))
            if self.charge_only:
                charge, sigma, epsilon = force.getParticleParameters(atom)
                force.setParticleParameters(index, nonbonded_params[0], sigma, epsilon)
            else:
                force.setParticleParameters(index, nonbonded_params[0], nonbonded_params[1], nonbonded_params[2])
        #exceptions
        for i, index in enumerate(self.ligand_info[1]):
            excep_idx = int(index)
            excep_params = excep[i]['data']
            [p1, p2, charge_prod, sigma, eps] = force.getExceptionParameters(excep_idx)
            if frozenset((p1-self.offset, p2-self.offset)) != excep[i]['id']:
                raise (ValueError('Fluorify has failed to generate nonbonded parameters(2) correctly please raise '
                                  'this as and issue at https://github.com/adw62/Fluorify'))
            if self.charge_only:
                force.setExceptionParameters(excep_idx, p1, p2, excep_params[0], sigma, eps)
            else:
                force.setExceptionParameters(excep_idx, p1, p2, excep_params[0], excep_params[1], excep_params[2])

        for i, (index, shift) in enumerate(zip(self.ghost_ligand_info[1], self.virt_excep_shift)):
            excep_idx = int(index)
            excep_params = ghost_excep[i]['data']
            [p1, p2, charge_prod, sigma, eps] = force.getExceptionParameters(excep_idx)
            if frozenset((p1-self.offset, p2-shift[1]-self.offset)) != ghost_excep[i]['id']:
                raise (ValueError('Fluorify has failed to generate nonbonded parameters(3) correctly please raise '
                                  'this as and issue at https://github.com/adw62/Fluorify'))
            if self.charge_only:
                force.setExceptionParameters(excep_idx, p1, p2, excep_params[0], sigma, eps)
            else:
                force.setExceptionParameters(excep_idx, p1, p2, excep_params[0], excep_params[1], excep_params[2])

    def get_mutant_energy(self, parameters, dcd, top, num_frames):
        chunk = math.ceil(len(parameters)/self.num_gpu)
        groups = grouper(range(len(parameters)), chunk)
        pool = Pool(processes=self.num_gpu)
        mutant_eng = partial(mutant_energy, sim=self, dcd=dcd, top=top, num_frames=num_frames, all_mutants=parameters)
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

    def zero_ghost_exceptions(self, system):
        """
        By initilizing ghost exceptions as non-zero it allows us to change them later without throwing an error
        however these exceptions must be zeroed after the context is created to simulate the physical system.
        """
        force = system.getForce(self.nonbonded_index)
        for excep in self.zero_exceptions:
            force.setExceptionParameters(*excep)
        return force

    def build_context(self, system, device):
        integrator = mm.LangevinIntegrator(self.temperature, self.friction, self.timestep)
        integrator.setConstraintTolerance(0.00001)
        platform = mm.Platform.getPlatformByName('CUDA')
        properties = {'CudaPrecision': 'mixed', 'CudaDeviceIndex': device}
        context = mm.Context(system, integrator, platform, properties)
        return context, integrator

    def treat_phase(self, mutant_parameters, dcd, top, num_frames):
        mutant_energy = FSim.get_mutant_energy(self, mutant_parameters, dcd, top, num_frames)
        wildtype_energy = mutant_energy.pop(-1)
        phase_free_energy = FSim.get_free_energy(self, mutant_energy, wildtype_energy)
        return phase_free_energy


def mutant_energy(idxs, sim, dcd, top, num_frames, all_mutants):
    mutants_systems_energies = []
    top = md.load(top).topology
    device = idxs[1]
    idxs = idxs[0]
    system = copy.deepcopy(sim.wt_system)
    context, integrator = sim.build_context(system, device=device)
    del integrator
    nonbonded_force = FSim.zero_ghost_exceptions(sim, system)
    nonbonded_force.updateParametersInContext(context)
    harmonic_force = system.getForce(sim.harmonic_index)
    torsion_force = system.getForce(sim.torsion_index)
    num_mutants = len(all_mutants)
    for i in idxs:
        if i == int(num_mutants-1):
           logger.debug('Computing potential for wild type ligand')
        else:
            logger.debug('Computing potential for mutant {0}/{1} on GPU {2}'.format(i+1, (num_mutants-1), device))
        sim.apply_nonbonded_parameters(nonbonded_force, all_mutants[i][0], all_mutants[i][1],
                                       all_mutants[i][2], all_mutants[i][3])
        nonbonded_force.updateParametersInContext(context)
        sim.apply_bonded_parameters(harmonic_force, all_mutants[i][4])
        harmonic_force.updateParametersInContext(context)
        sim.apply_torsion_parameters(torsion_force, all_mutants[i][5])
        torsion_force.updateParametersInContext(context)
        mutant_energies = []
        append = mutant_energies.append
        for frame in frames(dcd, top, maxframes=num_frames):
            context.setPositions(frame.xyz[0])
            context.setPeriodicBoxVectors(frame.unitcell_vectors[0][0],
                                          frame.unitcell_vectors[0][1], frame.unitcell_vectors[0][2])
            energy = context.getState(getEnergy=True, groups={sim.nonbonded_index, sim.harmonic_index, sim.torsion_index}).getPotentialEnergy()
            append(energy)
        mutants_systems_energies.append(mutant_energies)
    return mutants_systems_energies


def run_fep(idxs, sim, system, pdb, n_steps, n_iterations, all_mutants):
    device = idxs[1]
    idxs = idxs[0]
    context, integrator = sim.build_context(system, device)
    context.setPositions(pdb.positions)
    nonbonded_force = FSim.zero_ghost_exceptions(sim, system)
    nonbonded_force.updateParametersInContext(context)
    logger.debug('Minimizing...')
    mm.LocalEnergyMinimizer.minimize(context)
    temperature = sim.temperature
    context.setVelocitiesToTemperature(temperature)
    total_states = len(all_mutants)
    nstates = len(idxs)
    u_kln = np.zeros([nstates, total_states, n_iterations], np.float64)
    harmonic_force = system.getForce(sim.harmonic_index)
    torsion_force = system.getForce(sim.torsion_index)
    for k, m_id in enumerate(idxs):
        #m_id, id for mutant
        logger.debug('Computing potentials for FEP window {0}/{1} on GPU {2}'.format(k+1, total_states, device))
        for iteration in range(n_iterations):
            sim.apply_nonbonded_parameters(nonbonded_force, all_mutants[m_id][0], all_mutants[m_id][1],
                                           all_mutants[m_id][2], all_mutants[m_id][3])
            nonbonded_force.updateParametersInContext(context)
            sim.apply_bonded_parameters(harmonic_force, all_mutants[m_id][4])
            harmonic_force.updateParametersInContext(context)
            sim.apply_torsion_parameters(torsion_force, all_mutants[m_id][5])
            torsion_force.updateParametersInContext(context)
            # Run some dynamics
            integrator.step(n_steps)
            # Compute energies at all alchemical states
            for l, global_mutant in enumerate(all_mutants):
                sim.apply_nonbonded_parameters(nonbonded_force, global_mutant[0], global_mutant[1],
                                               global_mutant[2], global_mutant[3])
                nonbonded_force.updateParametersInContext(context)
                sim.apply_bonded_parameters(harmonic_force, global_mutant[4])
                harmonic_force.updateParametersInContext(context)
                sim.apply_torsion_parameters(torsion_force, global_mutant[5])
                torsion_force.updateParametersInContext(context)
                u_kln[k, l, iteration] = context.getState(getEnergy=True,
                        groups={sim.nonbonded_index, sim.harmonic_index, sim.torsion_index}).getPotentialEnergy() / sim.kT
    return u_kln


def run_dynamics(dcd_name, system, sim, equi, n_steps):
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
    pdb = sim.extended_pdb
    temperature = sim.temperature
    friction = sim.friction
    timestep = sim.timestep
    integrator = mm.LangevinIntegrator(temperature, friction, timestep)
    integrator.setConstraintTolerance(0.00001)

    platform = mm.Platform.getPlatformByName('CUDA')
    device = dcd_name[1]
    dcd_name = dcd_name[0][0]
    properties = {'CudaPrecision': 'mixed', 'CudaDeviceIndex': device}
    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setPositions(pdb.positions)

    #zero ghost exceptions
    nonbonded_force = FSim.zero_ghost_exceptions(sim, system)
    nonbonded_force.updateParametersInContext(simulation.context)

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


def get_ligand_info(ligand_name, snapshot, nonbonded_force, harmonic_force, torsion_force, system):
    ligand_atoms = snapshot.topology.select('resname {}'.format(ligand_name))
    if len(ligand_atoms) == 0:
        raise ValueError('Did not find ligand in supplied topology by name {}'.format(ligand_name))
    exception_list = list()
    ligand_exceptions = []
    for exception_index in range(nonbonded_force.getNumExceptions()):
        particle1, particle2, chargeProd, sigma, epsilon = nonbonded_force.getExceptionParameters(exception_index)
        if set([particle1, particle2]).intersection(ligand_atoms):
            exception_list.append((particle1, particle2))
            ligand_exceptions.append(exception_index)
    bond_list = list()
    ligand_bonds = []
    for bond_index in range(harmonic_force.getNumBonds()):
        particle1, particle2, r, k = harmonic_force.getBondParameters(bond_index)
        if set([particle1, particle2]).intersection(ligand_atoms):
            bond_list.append((particle1, particle2))
            ligand_bonds.append(bond_index)
    torsion_list = list()
    ligand_torsions = []
    for torsion_index in range(torsion_force.getNumTorsions()):
        particle1, particle2, particle3, particle4, periodicity, phase, k = torsion_force.getTorsionParameters(torsion_index)
        if set([particle1, particle2, particle3, particle4]).intersection(ligand_atoms):
            torsion_list.append((particle1, particle2, particle3, particle4))
            ligand_torsions.append(torsion_index)
    return [ligand_atoms, ligand_exceptions, ligand_bonds, ligand_torsions], exception_list, bond_list, torsion_list,


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

