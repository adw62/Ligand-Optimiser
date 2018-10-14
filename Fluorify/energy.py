#!/usr/bin/env python

from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
import mdtraj as md
import copy

class FSim(object):
    def __init__(self, ligand_name, ligand_atom=None, pdb_file='complex', sim_file='complex',
                 sim_dir='../complex_input/'):
        """
        Take pdb and dcd for full simulation
        find OG ligand
        apply charge to ligand in sim
        get energy of sim
        make some energy data structure
        """
        FSim.create_simuation(self, pdb_file, sim_file, sim_dir)
        if ligand_atom == None:
            FSim.get_ligand_atoms(self, ligand_name)

    def create_simuation(self, pdb_file, sim_file, sim_dir):
        pdb = app.PDBFile(sim_dir + pdb_file + '.pdb')
        self.snapshot = md.load(sim_dir + pdb_file + '.pdb')
        parameters_file_path = sim_dir + sim_file + '.prmtop'
        positions_file_path = sim_dir + sim_file + '.inpcrd'
        parameters_file = mm.app.AmberPrmtopFile(parameters_file_path)
        positions_file = mm.app.AmberInpcrdFile(positions_file_path)

        #put every force in different force group
        system = parameters_file.createSystem()
        for force_index, force in enumerate(system.getForces()):
            if isinstance(force, mm.NonbondedForce):
                self.nonbonded_index = force_index
            force.setForceGroup(force_index)
        self.system_save = system

        integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picoseconds,
                                           1.0 * unit.femtoseconds)
        platform = mm.Platform.getPlatformByName('CUDA')
        properties = {'CudaPrecision': 'mixed'}
        simulation = app.Simulation(pdb.topology, self.system_save, integrator, platform,
                                    properties)
        self.simulation = simulation

    def get_ligand_atoms(self, ligand_name):
        ligand_atoms = self.snapshot.topology.select('resname {}'.format(ligand_name))
        self.ligand_atoms = ligand_atoms

    def update_simulation(self, src_system):
        ###Credit to Lee-Ping Wang for update simulation and associated functions
        ### This is only updating nonbonded forces as other forces dont change.
        dest_simulation = self.simulation
        CopySystemParameters(src_system, dest_simulation.system)
        for i in range(src_system.getNumForces()):
            if hasattr(dest_simulation.system.getForce(i), 'updateParametersInContext'):
                dest_simulation.system.getForce(i).updateParametersInContext(dest_simulation.context)
            if isinstance(dest_simulation.system.getForce(i), mm.CustomNonbondedForce):
                force = src_system.getForce(i)
                for j in range(force.getNumGlobalParameters()):
                    pName = force.getGlobalParameterName(j)
                    pValue = force.getGlobalParameterDefaultValue(j)
                    dest_simulation.context.setParameter(pName, pValue)

    def apply_charges(self, charge):
        system = copy.deepcopy(self.system_save)
        for force in system.getForces():
            if isinstance(force, mm.NonbondedForce):
                nonbonded_force = force
        for i, atom_idx in enumerate(self.ligand_atoms):
            index = int(atom_idx)
            OG_charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
            nonbonded_force.setParticleParameters(index, charge[i], sigma, epsilon)
        FSim.update_simulation(self, system)


    def get_wildtype_energy(self, traj):
        """

        :param traj:
        :return: return wildtype ligand as list of frame wise energies
        """
        #reset simulation
        FSim.update_simulation(self, self.system_save)

        wildtype_frame_energies = []
        append = wildtype_frame_energies.append
        for frame in traj:
            self.simulation.context.setPositions(frame.xyz[0])
            State = self.simulation.context.getState(getEnergy=True, groups={self.nonbonded_index})
            append(State.getPotentialEnergy() / unit.kilojoule_per_mole)

        return wildtype_frame_energies

    def get_mutant_energy(self, charges, traj):
        """

        :param charges:
        :param traj:
        :return: list of mutant energies each mutant is list of frame wise energies
        """
        mutants_frame_energies = []
        for charge in charges:
            mutant_energies = []
            FSim.apply_charges(self, charge)
            append = mutant_energies.append
            for frame in traj:
                self.simulation.context.setPositions(frame.xyz[0])
                State = self.simulation.context.getState(getEnergy=True, groups={self.nonbonded_index})
                append(State.getPotentialEnergy() / unit.kilojoule_per_mole)
            mutants_frame_energies.append(mutant_energies)

        return mutants_frame_energies

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