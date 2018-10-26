#!/usr/bin/env python

from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
import mdtraj as md
import numpy as np
import copy


import time

#CONSTANTS
from openmmtools.constants import ONE_4PI_EPS0
kB = 0.008314472471220214
T = 300
kT = kB * T # Unit: kJ/mol


class FSim(object):
    def __init__(self, ligand_name, sim_name, input_folder):
        """
        Take pdb and dcd for full simulation
        find OG ligand
        apply charge to ligand in sim
        get energy of sim
        make some energy data structure
        """
        sim_dir = input_folder + sim_name + '/'
        self.snapshot = md.load(sim_dir + sim_name + '.pdb')
        self.all_atoms = [int(i) for i in self.snapshot.topology.select('all')]
        parameters_file_path = sim_dir + sim_name + '.prmtop'
        parameters_file = mm.app.AmberPrmtopFile(parameters_file_path)
        system = parameters_file.createSystem()

        forces_to_remove = []
        for force_index, force in enumerate(system.getForces()):
            forces_to_remove.append(force_index)
            if isinstance(force, mm.NonbondedForce):
                nonbonded_force = force
        custom_nonbonded_force = FSim.create_custom_force(self, nonbonded_force)
        system.addForce(custom_nonbonded_force)
        for i in reversed(forces_to_remove):
            system.removeForce(i)

        self.wt_system = system
        self.ligand_atoms = FSim.get_ligand_atoms(self, ligand_name)

    def build_context(self, system):
        integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        platform = mm.Platform.getPlatformByName('CUDA')
        properties = {'CudaPrecision': 'mixed'}
        context = mm.Context(system, integrator, platform, properties)
        return context

    def create_custom_force(self, nonbonded_force):
        [alpha_ewald, nx, ny, nz] = nonbonded_force.getPMEParameters()
        energy_expression = "(4*epsilon*((sigma/r)^12 - (sigma/r)^6) + ONE_4PI_EPS0*chargeprod*erfc(alpha_ewald*r)/r);"
        energy_expression += "epsilon = epsilon1*epsilon2;"
        energy_expression += "sigma = 0.5*(sigma1+sigma2);"
        energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)  # already in OpenMM units
        energy_expression += "chargeprod = charge1*charge2;"
        energy_expression += "alpha_ewald = {:f};".format(alpha_ewald.value_in_unit_system(unit.md_unit_system))
        custom_nonbonded_force = mm.CustomNonbondedForce(energy_expression)
        custom_nonbonded_force.addPerParticleParameter('charge')
        custom_nonbonded_force.addPerParticleParameter('sigma')
        custom_nonbonded_force.addPerParticleParameter('epsilon')
        custom_nonbonded_force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        custom_nonbonded_force.setCutoffDistance(1.1)
        custom_nonbonded_force.setUseLongRangeCorrection(False)

        for particle_index in range(nonbonded_force.getNumParticles()):
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle_index)
            custom_nonbonded_force.addParticle([charge, sigma, epsilon])
        return custom_nonbonded_force

    def get_ligand_atoms(self, ligand_name):
        ligand_atoms = self.snapshot.topology.select('resname {}'.format(ligand_name))
        if len(ligand_atoms) == 0:
            raise ValueError('Did not find ligand in supplied topology by name {}'.format(ligand_name))
        return ligand_atoms

    def treat_phase(self, ligand_charges, dcd, top):
        print('WT...')
        t0 = time.time()
        wildtype_energy = FSim.get_wildtype_energy(self, dcd, top)
        t1 = time.time()
        print('Took {} seconds'.format(t1 - t0))

        mutant_energy = FSim.get_mutant_energy(self, ligand_charges, dcd, top)
        phase_free_energy = get_free_energy(mutant_energy, wildtype_energy)

        return phase_free_energy

    def get_neighbour_list(self, frame, atom=None, cutoff=1.1):
        if atom == None:
            atom = self.ligand_atoms

        neighbour_list = md.compute_neighbors(frame, cutoff, query_indices=atom,
                                              haystack_indices=None, periodic=True)
        neighbour_list = [int(i) for i in neighbour_list[0]]
        return neighbour_list

    def apply_neighbour_list(self, force, neighbours):
        not_neighbours = [x for x in self.all_atoms if x not in neighbours]
        force.addInteractionGroup(neighbours, neighbours)
        force.addInteractionGroup([], not_neighbours)

    def frames(self, dcd, top, maxframes):
        equi = 0
        for i in range(equi, maxframes+equi):
            frame = md.load_dcd(dcd, top=top, stride=None, atom_indices=None, frame=i)
            yield frame

    def get_wildtype_energy(self, dcd, top):

        wildtype_frame_energies = []
        append = wildtype_frame_energies.append
        KJ_M = unit.kilojoule_per_mole
        neighbour_list_interval = 500

        for i, frame in enumerate(FSim.frames(self, dcd, top, maxframes=2000)):
            if i % neighbour_list_interval == 0:
                print(i, 'NEIGH')
                system = copy.deepcopy(self.wt_system)
                neighbours = FSim.get_neighbour_list(self, frame)
                FSim.apply_neighbour_list(self, system.getForce(0), neighbours)
                context = FSim.build_context(self, system)
            context.setPositions(frame.xyz[0])
            energy = context.getState(getEnergy=True).getPotentialEnergy()
            print(energy)
            append(energy/KJ_M)
        return wildtype_frame_energies

    def apply_charges(self, force, charge):
        for i, atom_idx in enumerate(self.ligand_atoms):
            index = int(atom_idx)
            OG_charge, sigma, epsilon = force.getParticleParameters(index)
            force.setParticleParameters(index, [charge[i], sigma, epsilon])

    def get_mutant_energy(self, charges, dcd, top):
        mutants_frame_energies = []
        KJ_M = unit.kilojoule_per_mole
        neighbour_list_interval = 500
        for charge in charges:
            mutant_energies = []
            append = mutant_energies.append

            charged_system = copy.deepcopy(self.wt_system)
            FSim.apply_charges(self, charged_system.getForce(0), charge)
            for i, frame in enumerate(FSim.frames(self, dcd, top, maxframes=2000)):
                if i % neighbour_list_interval == 0:
                    print(i, 'NEIGH')
                    grouped_system = copy.deepcopy(charged_system)
                    neighbours = FSim.get_neighbour_list(self, frame)
                    FSim.apply_neighbour_list(self, grouped_system.getForce(0), neighbours)
                    context = FSim.build_context(self, grouped_system)
                context.setPositions(frame.xyz[0])
                energy = context.getState(getEnergy=True).getPotentialEnergy()
                print(energy)
                append(energy / KJ_M)
            mutants_frame_energies.append(mutant_energies)
        return mutants_frame_energies


def get_energy(frame, context):
    context.setPositions(frame.xyz[0])
    energy = context.getState(getEnergy=True).getPotentialEnergy()
    return energy


def get_free_energy(mutant_energy, wildtype_energy, write_convergence=False):
    ans = []
    free_energy = []
    lig_con = []
    for ligand in mutant_energy:
        tmp = 0.0
        convergence = []
        for i in range(len(wildtype_energy)):
            print(ligand[i] - wildtype_energy[i])
            tmp += (np.exp(-(ligand[i] - wildtype_energy[i]) / kT))
            convergence.append(tmp/(float(i)+1.0))
        lig_con.append(convergence)
        ans.append(tmp / len(wildtype_energy))

    f = open('convergence.out', 'w')
    if write_convergence:
        for i, ligand in enumerate(lig_con):
            f.write('{} ///////////////////////////////////////////////\n'.format(i))
            for j, frame in enumerate(ligand):
                f.write('{0},    {1}\n'.format(j, (-kT * np.log(frame) * 0.239)))
    f.close()

    for ligand in ans:
        free_energy.append(-kT * np.log(ligand) * 0.239) # Unit: kcal/mol
    return free_energy

