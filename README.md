# Ligand Charge Optimiser
Optimise the atomistic charges of ligand to maximize receptor binding affinity.

# Installation

- Download from git

  git clone https://github.com/adw62/Ligand_Charge_Optimiser/

- Install via pip

  pip install path/to/directory
  
- Install yank and multiprocess to get all dependencies
 
  conda install -c omnia yank
  
  conda install -c conda-forge multiprocess
  
# Options

[--job_type=STRING] FEP jobs that can be performed,
                    default: 'F'
                    options:
                            F  - Fluorine
                            Cl - Chlorine
                            N  - Nitrogen
                            S  - Sulphur

[--output_folder=STRING] Directory for output,
                         default: './mol_name_job_type'

[--ligand_name=STRING] String ligand is adressed by in input mol2 file,
                       default: 'MOL'

[--mol_name=STRING] Name of input mol2 file containg ligand,
                    default: 'ligand'

[--complex_name=STRING] Name of input pdb file containg ligand,
                        default: 'complex'

[--solvent_name=STRING] Name of input pdb file containg ligand,
                        default: 'solvent'

[--yaml_path=STRING] Path to yaml file containg options for yank exsperiment builder,
                     default: './setup.yaml'

[--o_atom_list=LIST] List of indices of oxygen atoms in ligand.mol2 to be replaced with S
                     default: None

[--c_atom_list=LIST] List of indices of carbon atoms in ligand.mol2 to be replaced with N
                     default: None

[--h_atom_list=LIST] List of indices of hydrogen atoms in ligand.mol2 to be replaced with F or Cl
                     default: None
                     
[--auto_select=STRING] Automatic selction of indicies for mutation based on input,
                       default: None
                       options:
                               1  - for O.1 oxygens, C.1 carbons or their associated H
                               2  - for O.2 oxygens, C.2 carbons or their associated H
                               3  - for O.3 oxygens, C.2 carbons or their associated H
                               ar - for O.ar oxygens, C.ar carbons or their associated H
                       
[--num_frames=INT] Number of frames of trajectory to collect for objective, frames spaced by 5ps,
                   default: 500

[--net_charge=INT] Net charge of ligand to be passed to antechamber for paramterisation, net_charge should also be set in setup.yaml
                    default: 0
            
[--gaff_ver=INT] Gaff version to use in paramterisation,
                 default: 2
                 options: 1, 2
                  
[--equi=INT] Number of steps of equilibriation, each step is 2fs,
             default: 100

[--num_fep=INT] Number of repeates to do when testing the set of optimised charges with full FEP.
                default: 1

[--charge_only=BOOL] Boolean to determine if only charge parameters should be changed.
                     note: should be True for optimisation
                     default: False

[--vdw_only=BOOL] Boolean to determine if only van der Waals parameters should be changed.
                  default: False

[--optimize=BOOL] Boolean to determine if an optimisation is being performed.
                  default: False
       
[--opt_name=STRING] Name of optimisation being performed,
                    default: scipy
                    options: scipy 
                     
[--rmsd=FLOAT] RMSD limit placed on the original and optimised charges,
               default: 0.03 q_e

[--opt_steps=INT] Number of optimisation steps to perform,
                  default: None

[--central_diff=BOOL] Boolean to deterimine if the optmiser will calculate the gradient with a central of forward difference,
                      default: True

[--num_gpu=INT] Number of GPU for the node where the calculation is run,
                note: This software is not configured to use MPI and should only be run on one node, however this node may have multiple GPUs
                default: 1

# Example usage

Optimise atomic charges of a ligand and verify the ddG of this optisation with one full FEP calculation 
LigCharOpt --optimize='1' --charge_only='1' --yaml_path='./setup.yaml'

Run a full FEP calculations on all mutants with 'O.3' oxygens swapped for S  
LigCharOpt --job_type='S' --auto_select='3' --yaml_path='./setup.yaml'
