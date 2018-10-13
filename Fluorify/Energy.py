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
        self.system_save = parameters_file.createSystem()
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

    def get_wildtype_energy():
        pass

    def get_mutant_energy(self, charges, traj):
        all_mutant_energies = []
        for charge in charges:
            mutant_energies = []
            FSim.apply_charges(self, charge)
            for frame in traj:
                self.simulation.context.setPositions(frame.xyz[0])
                State = self.simulation.context.getState(getEnergy=True)
                mutant_energies.append(State.getPotentialEnergy() / unit.kilojoule_per_mole)
            all_mutant_energies.append(mutant_energies)
        return all_mutant_energies

def CopyAmoebaBondParameters(src,dest):
    dest.setAmoebaGlobalBondCubic(src.getAmoebaGlobalBondCubic())
    dest.setAmoebaGlobalBondQuartic(src.getAmoebaGlobalBondQuartic())
    for i in range(src.getNumBonds()):
        dest.setBondParameters(i,*src.getBondParameters(i))

def CopyAmoebaOutOfPlaneBendParameters(src,dest):
    dest.setAmoebaGlobalOutOfPlaneBendCubic(src.getAmoebaGlobalOutOfPlaneBendCubic())
    dest.setAmoebaGlobalOutOfPlaneBendQuartic(src.getAmoebaGlobalOutOfPlaneBendQuartic())
    dest.setAmoebaGlobalOutOfPlaneBendPentic(src.getAmoebaGlobalOutOfPlaneBendPentic())
    dest.setAmoebaGlobalOutOfPlaneBendSextic(src.getAmoebaGlobalOutOfPlaneBendSextic())
    for i in range(src.getNumOutOfPlaneBends()):
        dest.setOutOfPlaneBendParameters(i,*src.getOutOfPlaneBendParameters(i))

def CopyAmoebaAngleParameters(src, dest):
    dest.setAmoebaGlobalAngleCubic(src.getAmoebaGlobalAngleCubic())
    dest.setAmoebaGlobalAngleQuartic(src.getAmoebaGlobalAngleQuartic())
    dest.setAmoebaGlobalAnglePentic(src.getAmoebaGlobalAnglePentic())
    dest.setAmoebaGlobalAngleSextic(src.getAmoebaGlobalAngleSextic())
    for i in range(src.getNumAngles()):
        dest.setAngleParameters(i,*src.getAngleParameters(i))
    return

def CopyAmoebaInPlaneAngleParameters(src, dest):
    dest.setAmoebaGlobalInPlaneAngleCubic(src.getAmoebaGlobalInPlaneAngleCubic())
    dest.setAmoebaGlobalInPlaneAngleQuartic(src.getAmoebaGlobalInPlaneAngleQuartic())
    dest.setAmoebaGlobalInPlaneAnglePentic(src.getAmoebaGlobalInPlaneAnglePentic())
    dest.setAmoebaGlobalInPlaneAngleSextic(src.getAmoebaGlobalInPlaneAngleSextic())
    for i in range(src.getNumAngles()):
        dest.setAngleParameters(i,*src.getAngleParameters(i))
    return

def CopyAmoebaVdwParameters(src, dest):
    for i in range(src.getNumParticles()):
        dest.setParticleParameters(i,*src.getParticleParameters(i))

def CopyAmoebaMultipoleParameters(src, dest):
    for i in range(src.getNumMultipoles()):
        dest.setMultipoleParameters(i,*src.getMultipoleParameters(i))

def CopyHarmonicBondParameters(src, dest):
    for i in range(src.getNumBonds()):
        dest.setBondParameters(i,*src.getBondParameters(i))

def CopyHarmonicAngleParameters(src, dest):
    for i in range(src.getNumAngles()):
        dest.setAngleParameters(i,*src.getAngleParameters(i))

def CopyPeriodicTorsionParameters(src, dest):
    for i in range(src.getNumTorsions()):
        dest.setTorsionParameters(i,*src.getTorsionParameters(i))

def CopyNonbondedParameters(src, dest):
    dest.setReactionFieldDielectric(src.getReactionFieldDielectric())
    for i in range(src.getNumParticles()):
        dest.setParticleParameters(i,*src.getParticleParameters(i))
    for i in range(src.getNumExceptions()):
        dest.setExceptionParameters(i,*src.getExceptionParameters(i))

def CopyGBSAOBCParameters(src, dest):
    dest.setSolventDielectric(src.getSolventDielectric())
    dest.setSoluteDielectric(src.getSoluteDielectric())
    for i in range(src.getNumParticles()):
        dest.setParticleParameters(i,*src.getParticleParameters(i))

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
    Copiers = {'AmoebaBondForce':CopyAmoebaBondParameters,
               'AmoebaOutOfPlaneBendForce':CopyAmoebaOutOfPlaneBendParameters,
               'AmoebaAngleForce':CopyAmoebaAngleParameters,
               'AmoebaInPlaneAngleForce':CopyAmoebaInPlaneAngleParameters,
               'AmoebaVdwForce':CopyAmoebaVdwParameters,
               'AmoebaMultipoleForce':CopyAmoebaMultipoleParameters,
               'HarmonicBondForce':CopyHarmonicBondParameters,
               'HarmonicAngleForce':CopyHarmonicAngleParameters,
               'PeriodicTorsionForce':CopyPeriodicTorsionParameters,
               'NonbondedForce':CopyNonbondedParameters,
               'CustomNonbondedForce':CopyCustomNonbondedParameters,
               'GBSAOBCForce':CopyGBSAOBCParameters,
               'CMMotionRemover':do_nothing}
    for i in range(src.getNumForces()):
        nm = src.getForce(i).__class__.__name__
        if nm in Copiers:
            Copiers[nm](src.getForce(i),dest.getForce(i))
        else:
            print('There is no Copier function implemented for the OpenMM force type %s!' % nm)