#!/usr/bin/env python

import simtk.openmm as mm
from simtk.openmm import app
from simtk import unit
import mdtraj as md
import numpy as np
import copy
from sys import stdout

from math import ceil
from functools import partial
from multiprocess import Pool

#CONSTANTS
e = unit.elementary_charges
kB = 0.008314472471220214
T = 300
kT = kB * T # Unit: kJ/mol

class FSim(object):
    def __init__(self, ligand_name, sim_name, input_folder, charge_only, num_gpu):
        """ A class for creating OpenMM context from input files and calculating free energy
        change when modifying the parameters of the system in the context.

        ligand_name : resname of ligand in input files
        sim_name: complex or solvent
        input_folder: input directory
        """
        self.num_gpu = num_gpu
        self.charge_only = charge_only
        #Create system from input files
        sim_dir = input_folder + sim_name + '/'
        self.snapshot = md.load(sim_dir + sim_name + '.pdb')
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
        self.ligand_atoms = FSim.get_ligand_atoms(self, ligand_name)

    def run_parallel_dynamics(self, output_folder, name, n_steps, mutant_parameters):
        mutant_system = copy.deepcopy(self.wt_system)
        if mutant_parameters is not None:
            non_bonded_force = mutant_system.getForce(self.nonbonded_index)
            self.apply_parameters(non_bonded_force, mutant_parameters)
            context = FSim.build_context(self, mutant_system)
            system = context.getSystem()
        else:
            system = mutant_system

        box_vectors = self.pdb.topology.getPeriodicBoxVectors()
        system.setDefaultPeriodicBoxVectors(*box_vectors)
        system.addForce(mm.MonteCarloBarostat(1 * unit.atmospheres, 300 * unit.kelvin, 25))

        dcd_names = [output_folder+name+'_gpu'+str(i) for i in range(self.num_gpu)]
        pool = Pool(processes=self.num_gpu)
        run = partial(run_dynamics, system=system, pdb=self.pdb, n_steps=ceil(n_steps/self.num_gpu))
        pool.map(run, dcd_names)
        return dcd_names

    def build_context(self, system):
        integrator = mm.VerletIntegrator(2.0*unit.femtoseconds)
        try:
            platform = mm.Platform.getPlatformByName('CUDA')
            properties = {'CudaPrecision': 'mixed'}
            context = mm.Context(system, integrator, platform, properties)
        except:
            print('Not using CUDA')
            platform = mm.Platform.getPlatformByName('Reference')
            context = mm.Context(system, integrator, platform)
        return context

    def get_ligand_atoms(self, ligand_name):
        ligand_atoms = self.snapshot.topology.select('resname {}'.format(ligand_name))
        if len(ligand_atoms) == 0:
            raise ValueError('Did not find ligand in supplied topology by name {}'.format(ligand_name))
        return ligand_atoms

    def treat_phase(self, wt_parameters, mutant_parameters, dcd, top, num_frames, wildtype_energy=None):
        if wildtype_energy is None:
            wildtype_energy = FSim.get_mutant_energy(self, [wt_parameters], dcd, top, num_frames, True)
        mutant_energy = FSim.get_mutant_energy(self, mutant_parameters, dcd, top, num_frames)
        phase_free_energy = get_free_energy(mutant_energy, wildtype_energy[0])
        return phase_free_energy

    def frames(self, dcd, top, maxframes):
        maxframes = ceil(maxframes/len(dcd))
        for name in dcd:
            for i in range(maxframes):
                frame = md.load_dcd(name, top=top, stride=None, atom_indices=None, frame=i)
                yield frame

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
        mutants_frame_energies = []
        KJ_M = unit.kilojoule_per_mole
        for index, mutant_parameters in enumerate(parameters):
            if wt:
                print('Computing potential for wild type ligand')
            else:
                print('Computing potential for mutant {0}/{1}'.format(index+1, len(parameters)))
            mutant_energies = []
            append = mutant_energies.append
            mutant_system = copy.deepcopy(self.wt_system)
            FSim.apply_parameters(self, mutant_system.getForce(self.nonbonded_index), mutant_parameters)
            #mutant_systems.append(mutant_system), could pass systems to gpu then build context
            context = FSim.build_context(self, mutant_system)
            for frame in FSim.frames(self, dcd, top, maxframes=num_frames):
                context.setPositions(frame.xyz[0])
                context.setPeriodicBoxVectors(frame.unitcell_vectors[0][0],
                                              frame.unitcell_vectors[0][1], frame.unitcell_vectors[0][2])
                energy = context.getState(getEnergy=True, groups={self.nonbonded_index}).getPotentialEnergy()
                append(energy / KJ_M)
            mutants_frame_energies.append(mutant_energies)
        return mutants_frame_energies

def run_dynamics(dcd_name, system, pdb, n_steps):
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
    temperature = 300 * unit.kelvin
    friction = 0.3 / unit.picosecond
    timestep = 2.0 * unit.femtosecond
    integrator = mm.LangevinIntegrator(temperature, friction, timestep)
    integrator.setConstraintTolerance(0.00001)

    platform = mm.Platform.getPlatformByName('CUDA')
    device = dcd_name[-1]
    properties = {'CudaPrecision': 'mixed', 'CudaDeviceIndex': device} #dcd_name
    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setPositions(pdb.positions)

    print('Minimizing..')
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
    print('Equilibrating...')
    equi = 100  # 250000
    simulation.step(equi)

    simulation.reporters.append(app.DCDReporter(dcd_name, 2500))
    simulation.reporters.append(app.StateDataReporter(stdout, 2500, step=True,
    potentialEnergy=True, temperature=True, progress=True, remainingTime=True,
    speed=True, totalSteps=n_steps, separator='\t'))

    print('Running Production...')
    simulation.step(n_steps)
    print('Done!')


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

