#!/usr/local/bin/env python

from .fluorify import Scanning
from docopt import docopt

# =============================================================================================
# COMMAND-LINE INTERFACE
# =============================================================================================

usage = """
FLUORIFY
Usage:
  Fluorify [--output_folder=STRING] [--mol_file=STRING] [--ligand_name=STRING] [--complex_name=STRING] 
           [--solvent_name=STRING] [--c_atom_list=LIST] [--h_atom_list=LIST] [--num_frames=INT] [--net_charge=INT] 
           [--auto_select=STRING] [--charge_only=BOOL] [--job_type=STRING]...
"""


def main(argv=None):
    args = docopt(usage, argv=argv, options_first=True)

    msg = 'No {0} specified using default {1}'

    if args['--mol_file']:
        mol_file = args['--mol_file']
    else:
        mol_file = 'ligand'
        print(msg.format('mol file', mol_file + '.mol2'))

    if args['--ligand_name']:
        ligand_name = args['--ligand_name']
    else:
        ligand_name = 'MOL'
        print(msg.format('ligand residue name', ligand_name))

    if args['--complex_name']:
        complex_name = args['--complex_name']
    else:
        complex_name = 'complex'
        print(msg.format('complex name', complex_name))

    if args['--solvent_name']:
        solvent_name = args['--solvent_name']
    else:
        solvent_name = 'solvent'
        print(msg.format('solvent name', solvent_name))

    if args['--c_atom_list']:
        c_atom_list = []
        pairs = args['--c_atom_list']
        pairs = pairs.replace(" ", "")
        pairs = pairs.split('and')
        for pair in pairs:
            tmp = []
            pair = pair.split(',')
            for atom in pair:
                tmp.append(atom)
            c_atom_list.append(tmp)
    else:
        c_atom_list = None

    if args['--h_atom_list']:
        h_atom_list = []
        pairs = args['--h_atom_list']
        pairs = pairs.replace(" ", "")
        pairs = pairs.split('and')
        for pair in pairs:
            tmp = []
            pair = pair.split(',')
            for atom in pair:
                tmp.append(atom)
            h_atom_list.append(tmp)
    else:
        h_atom_list = None

    if args['--auto_select']:
        auto_select = args['--auto_select']
        auto = ['1', '2', '3', 'ar']
        if auto_select not in auto:
            raise ValueError('Allowed automatic selections {}'.format(auto))
        if c_atom_list is not None or h_atom_list is not None:
            raise ValueError('Automatic target atom selection will conflict with populated atom lists')
    else:
        if c_atom_list is None and h_atom_list is None:
            raise ValueError('No target atoms specified')
        else:
            auto_select = None

    if args['--job_type']:
        job_type = args['--job_type'][0]
        allowed_jobs = ['F', 'Cl', 'N', 'NxF', 'NxCl']
        if job_type not in allowed_jobs:
            raise ValueError('Allowed elements {}'.format(allowed_jobs))
    else:
        job_type = 'F'
        print(msg.format('job_type', job_type))

    if args['--output_folder']:
        output_folder = args['--output_folder']
    else:
        output_folder = './' + mol_file + '_' + job_type + '/'
        print(msg.format('output folder', output_folder))

    if args['--num_frames']:
        num_frames = int(args['--num_frames'])
    else:
        num_frames = 9500
        print(msg.format('number of frames', num_frames))

    if args['--net_charge']:
        net_charge = int(args['--net_charge'])
    else:
        net_charge = None
        print(msg.format('net charge', net_charge))

    if args['--charge_only']:
        charge_only = int(args['--charge_only'])
    else:
        charge_only = False
    if charge_only == True:
        print('Mutating ligand charges only...')
    else:
        print('Mutating all ligand parameters...')

    Scanning(output_folder, mol_file, ligand_name, net_charge, complex_name,
             solvent_name, job_type, auto_select, c_atom_list, h_atom_list, num_frames, charge_only)

