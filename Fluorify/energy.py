#!/usr/bin/env python

import simtk.openmm as mm
from simtk.openmm import app
from simtk import unit
import mdtraj as md
import numpy as np
import copy

#CONSTANTS
e = unit.elementary_charges
kB = 0.008314472471220214
T = 300
kT = kB * T # Unit: kJ/mol


class FSim(object):
    def __init__(self, ligand_name, sim_name, input_folder, charge_only):
        """ A class for creating OpenMM context from input files and calculating free energy
        change when modifying the parameters of the system in the context.

        ligand_name : resname of ligand in input files
        sim_name: complex or solvent
        input_folder: input directory
        """
        self.charge_only = charge_only
        #Create system from input files
        sim_dir = input_folder + sim_name + '/'
        self.snapshot = md.load(sim_dir + sim_name + '.pdb')
        parameters_file_path = sim_dir + sim_name + '.prmtop'
        parameters_file = mm.app.AmberPrmtopFile(parameters_file_path)
        system = parameters_file.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometers,
                                              constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=0.0005)

        for force_index, force in enumerate(system.getForces()):
            if isinstance(force, mm.NonbondedForce):
                self.nonbonded_index = force_index
            force.setForceGroup(force_index)

        self.wt_system = system
        self.ligand_atoms = FSim.get_ligand_atoms(self, ligand_name)

    def build_context(self, system):
        integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        try:
            platform = mm.Platform.getPlatformByName('CUDA')
            properties = {'CudaPrecision': 'mixed'}
            context = mm.Context(system, integrator, platform, properties)
        except:
            platform = mm.Platform.getPlatformByName('Reference')
            context = mm.Context(system, integrator, platform)
        return context

    def get_ligand_atoms(self, ligand_name):
        ligand_atoms = self.snapshot.topology.select('resname {}'.format(ligand_name))
        if len(ligand_atoms) == 0:
            raise ValueError('Did not find ligand in supplied topology by name {}'.format(ligand_name))
        return ligand_atoms



    def treat_phase(self, ligand_parameters, dcd, top, num_frames):
        wildtype_energy = FSim.get_wildtype_energy(self, dcd, top, num_frames)
        mutant_energy = FSim.get_mutant_energy(self, ligand_parameters, dcd, top, num_frames)
        phase_free_energy = get_free_energy(mutant_energy, wildtype_energy)
        return phase_free_energy

    def frames(self, dcd, top, maxframes):
        for i in range(maxframes):
            frame = md.load_dcd(dcd, top=top, stride=None, atom_indices=None, frame=i)
            yield frame

    def get_wildtype_energy(self, dcd, top, num_frames):
        wildtype_frame_energies = []
        append = wildtype_frame_energies.append
        KJ_M = unit.kilojoule_per_mole
        context = FSim.build_context(self, self.wt_system)
        #need to catch if traj shorter than requested read
        for frame in FSim.frames(self, dcd, top, maxframes=num_frames):
            context.setPositions(frame.xyz[0])
            context.setPeriodicBoxVectors(frame.unitcell_vectors[0][0],
                                          frame.unitcell_vectors[0][1], frame.unitcell_vectors[0][2])
            energy = context.getState(getEnergy=True, groups={self.nonbonded_index}).getPotentialEnergy()
            append(energy/KJ_M)
        return wildtype_frame_energies

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

    def get_mutant_energy(self, parameters, dcd, top, num_frames):
        mutants_frame_energies = []
        KJ_M = unit.kilojoule_per_mole
        for index, mutant_parameters in enumerate(parameters):
            print('Computing potential for mutant {0}/{1}'.format(index+1, len(parameters)))
            mutant_energies = []
            append = mutant_energies.append
            mutant_system = copy.deepcopy(self.wt_system)
            FSim.apply_parameters(self, mutant_system.getForce(self.nonbonded_index), mutant_parameters)
            context = FSim.build_context(self, mutant_system)
            for frame in FSim.frames(self, dcd, top, maxframes=num_frames):
                context.setPositions(frame.xyz[0])
                context.setPeriodicBoxVectors(frame.unitcell_vectors[0][0],
                                              frame.unitcell_vectors[0][1], frame.unitcell_vectors[0][2])
                energy = context.getState(getEnergy=True, groups={self.nonbonded_index}).getPotentialEnergy()
                append(energy / KJ_M)
            mutants_frame_energies.append(mutant_energies)
        return mutants_frame_energies


def get_free_energy(mutant_energy, wildtype_energy, write_convergence=False):
    ans = []
    free_energy = []
    lig_con = []
    for ligand in mutant_energy:
        tmp = 0.0
        convergence = []
        for i in range(len(wildtype_energy)):
            tmp += (np.exp(-(ligand[i] - wildtype_energy[i]) / kT))
            convergence.append(tmp/(float(i)+1.0))
        lig_con.append(convergence)
        ans.append(tmp / len(wildtype_energy))

    if write_convergence:
        f = open('convergence.out', 'w')
        for i, ligand in enumerate(lig_con):
            f.write('{} ///////////////////////////////////////////////\n'.format(i))
            for j, frame in enumerate(ligand):
                f.write('{0},    {1}\n'.format(j, (-kT * np.log(frame) * 0.239)))
        f.close()

    for ligand in ans:
        free_energy.append(-kT * np.log(ligand) * 0.239) # Unit: kcal/mol
    return free_energy

