#!/usr/local/bin/env python

from .fluorify import FluorineScanning
from docopt import docopt

# =============================================================================================
# COMMAND-LINE INTERFACE
# =============================================================================================

usage = """
FLUORIFY
Usage:
  Fluorify [--input=STRING] [--output=STRING] [--mol_file=STRING] [--ligand_name=STRING] [--complex_pdb_file=STRING] 
           [--complex_prmtop_file=STRING] [--traj_file=STRING] [--job_type=STRING]...
"""


def main(argv=None):
    args = docopt(usage, argv=argv, options_first=True)

    msg = 'No {0} specified using default {1}'

    if args['--input']:
        input_folder = args['--input']
    else:
        input_folder = './input/'
        print(msg.format('input folder', input_folder))

    if args['--output']:
        output_folder = args['--output']
    else:
        output_folder = './output/'
        print(msg.format('output folder', output_folder))

    if args['--mol_file']:
        mol_file = args['--mol_file']
    else:
        mol_file ='ligand.mol2'
        print(msg.format('mol file', mol_file))

    if args['--ligand_name']:
        ligand_name = args['--ligand_name']
    else:
        ligand_name ='MOL'
        print(msg.format('ligand name', ligand_name))

    if args['--complex_pdb_file']:
        complex_pdb_file  = args['--complex_pdb_file']
    else:
        complex_pdb_file = 'complex.pdb'
        print(msg.format('complex pdb file', complex_pdb_file))

    if args['--complex_prmtop_file']:
        complex_prmtop_file  = args['--complex_prmtop_file']
    else:
        complex_prmtop_file = 'complex.prmtop'
        print(msg.format('complex prmtop file', complex_prmtop_file))

    if args['--traj_file']:
        traj_file  = args['--traj_file']
    else:
        traj_file = 'input.dcd'
        print(msg.format('trajectory file', traj_file))

    if args['--job_type']:
        job_type = args['--job_type'][0]
        allowed_elements = ['F', 'Cl']
        allowed_carbon_types = ['1', '2', '3', 'ar']
        job_type = job_type.split('_')
        if job_type[0] not in allowed_elements:
            raise ValueError('Allowed elements {}'.format(allowed_elements))
        elif job_type[1] not in allowed_carbon_types:
            raise ValueError('Allowed carbon types {}'.format(allowed_carbon_types))
        else:
                pass
    else:
        job_type = ['F', 'ar']
        print(msg.format('job_type', job_type))

    FluorineScanning(input_folder, output_folder, mol_file, ligand_name,
                        complex_pdb_file, complex_prmtop_file, traj_file, job_type)

