#!/usr/bin/env python

from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
import mdtraj as md
import numpy as np

#CONSTANTS
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
        FSim.create_simulation(self, sim_name, sim_dir)
        FSim.get_ligand_atoms(self, ligand_name)

    def create_simulation(self, sim_name, sim_dir):
        pdb = app.PDBFile(sim_dir + sim_name + '.pdb')
        self.snapshot = md.load(sim_dir + sim_name + '.pdb')
        parameters_file_path = sim_dir + sim_name + '.prmtop'
        parameters_file = mm.app.AmberPrmtopFile(parameters_file_path)
        #positions_file_path = sim_dir + sim_file + '.inpcrd'
        #positions_file = mm.app.AmberInpcrdFile(positions_file_path)

        #put every force in different force group
        system = parameters_file.createSystem()
        for force_index, force in enumerate(system.getForces()):
            if isinstance(force, mm.NonbondedForce):
                self.nonbonded_index = force_index
            force.setForceGroup(force_index)
        self.system_save = system

        integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picoseconds,
                                           1.0 * unit.femtoseconds)
        #precision is significant to answers
        platform = mm.Platform.getPlatformByName('CUDA')
        properties = {'CudaPrecision': 'mixed'}
        simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
        simulation.context.setPositions(pdb.positions)
        self.simulation = simulation

    def get_ligand_atoms(self, ligand_name):
        ligand_atoms = self.snapshot.topology.select('resname {}'.format(ligand_name))
        if len(ligand_atoms) == 0:
            raise ValueError('Did not find ligand in supplied topology by name {}'.format(ligand_name))
        self.ligand_atoms = ligand_atoms

    def treat_phase(self, ligand_charges, traj):
        pos = []
        for frame in traj:
            pos.append(frame.xyz[0])
        context = self.simulation.context
        wildtype_energy = get_wildtype_energy(context, pos, self.nonbonded_index)
        mutant_energy = get_mutant_energy(context, ligand_charges, pos,
                                          self.nonbonded_index, self.ligand_atoms)
        phase_free_energy = free_energy(mutant_energy, wildtype_energy)
        return phase_free_energy

    def reset_simulation(self):
        update_context(self.simulation.context, self.system_save)


def apply_charges(context, charge, ligand_atoms):
    system = context.getSystem()
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            nonbonded_force = force
    for i, atom_idx in enumerate(ligand_atoms):
        index = int(atom_idx)
        OG_charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
        nonbonded_force.setParticleParameters(index, charge[i], sigma, epsilon)
    update_context(context, system)


def get_wildtype_energy(context, pos, nonbonded_index):
    wildtype_frame_energies = []
    append = wildtype_frame_energies.append
    KJ_M = unit.kilojoule_per_mole
    State = context.getState(getEnergy=True, groups={nonbonded_index})
    for frame in pos:
        context.setPositions(frame)
        energy = State.getPotentialEnergy()
        append(energy / KJ_M)
    return wildtype_frame_energies


def get_mutant_energy(context, charges, pos, nonbonded_index, ligand_atoms):
    mutants_frame_energies = []
    KJ_M = unit.kilojoule_per_mole
    for charge in charges:
        mutant_energies = []
        apply_charges(context, charge, ligand_atoms)
        append = mutant_energies.append
        State = context.getState(getEnergy=True, groups={nonbonded_index})
        for frame in pos:
            context.setPositions(frame)
            energy = State.getPotentialEnergy()
            append(energy / KJ_M)
        mutants_frame_energies.append(mutant_energies)
    return mutants_frame_energies


def free_energy(mutant_energy, wildtype_energy):
    ans = []
    free_energy = []
    for ligand in mutant_energy:
        tmp = 0.0
        for i in range(len(wildtype_energy)):
            #print(ligand[i] - wildtype_energy[i])
            tmp += (np.exp(-(ligand[i] - wildtype_energy[i]) / kT))
        ans.append(tmp / len(wildtype_energy))
    for ligand in ans:
        free_energy.append(-kT * np.log(ligand) * 0.239) # Unit: kcal/mol
    return free_energy


def update_context(context, src_system):
    ###Credit to Lee-Ping Wang for update context and associated functions
    ### This is only updating nonbonded forces as other forces dont change.
    dest_system = context.getSystem()
    CopySystemParameters(src_system, dest_system)
    for i in range(src_system.getNumForces()):
        if hasattr(dest_system.getForce(i), 'updateParametersInContext'):
            dest_system.getForce(i).updateParametersInContext(context)
        if isinstance(dest_system.getForce(i), mm.CustomNonbondedForce):
            force = src_system.getForce(i)
            for j in range(force.getNumGlobalParameters()):
                pName = force.getGlobalParameterName(j)
                pValue = force.getGlobalParameterDefaultValue(j)
                context.setParameter(pName, pValue)


def CopySystemParameters(src,dest):
    """Copy parameters from one system (i.e. that which is created by a new force field)
    sto another system (i.e. the one stored inside the Target object).
    DANGER: These need to be implemented manually!!!"""
    Copiers = {'NonbondedForce':CopyNonbondedParameters,
               'CustomNonbondedForce':CopyCustomNonbondedParameters}
    for i in range(src.getNumForces()):
        nm = src.getForce(i).__class__.__name__
        if nm in Copiers:
            Copiers[nm](src.getForce(i),dest.getForce(i))
        else:
            pass
            #print('There is no Copier function implemented for the OpenMM force type %s!' % nm)


def CopyNonbondedParameters(src, dest):
    dest.setReactionFieldDielectric(src.getReactionFieldDielectric())
    for i in range(src.getNumParticles()):
        dest.setParticleParameters(i,*src.getParticleParameters(i))
    for i in range(src.getNumExceptions()):
        dest.setExceptionParameters(i,*src.getExceptionParameters(i))


def CopyCustomNonbondedParameters(src, dest):
    '''
    copy whatever updateParametersInContext can update:
        per-particle parameters
    '''
    for i in range(src.getNumParticles()):
        dest.setParticleParameters(i, list(src.getParticleParameters(i)))


def do_nothing(src, dest):
    return