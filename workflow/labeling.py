from ase.io import read, write, Trajectory
from ase.db import connect
from shutil import copy
import os, subprocess
import numpy as np
import argparse
import json
import toml
from pathlib import Path
from ase.calculators.calculator import CalculationFailed

from perqueue.constants import DYNAMICWIDTHGROUP_KEY,CYCLICALGROUP_KEY, ITER_KW, INDEX_KW


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="DFT labeling of the selected datset", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--label_set",
        type=str,
        help="Path to trajectory to be labeled by DFT",
    )
    parser.add_argument(
        "--train_set",
        type=str,
        help="Path to existing training data set",
    )
    parser.add_argument(
        "--pool_set", 
        type=str, 
        help="Path to MD trajectory obtained from machine learning potential",
    )
    parser.add_argument(
        "--al_info", 
        type=str, 
        help="Path to json file that stores indices selected in pool set",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="arguments.toml",
        help="Path to config file. e.g. 'arguments.toml'"
    )

    return parser.parse_args(arg_list)

def update_namespace(ns, d):
    for k, v in d.items():
        if not isinstance(v, dict):
            ns.__dict__[k] = v

def main(cfg,system_name,**kwargs):
    # Load iteration index
    iter_idx,*_ = kwargs[ITER_KW]

    with open(cfg, 'r') as f:
        main_params = toml.load(f)
    
    # Load local parameters
    task_name = 'labeling'
    params = main_params[task_name]

    # Load the system parameters
    system_params = params['runs'][system_name]
        
    # Update the main parameters with the system parameters
    params.update(system_params)

    # Set the data set selected by the active learning 
    run_path = main_params['global']['run_path']
    al_path = os.path.join(run_path,'select', f'iter_{iter_idx}',system_name)
    params['al_info'] = os.path.join(al_path,'selected.json')

    # Get the pool set and the selected indices
    with open(params['al_info']) as f:
        al_data = json.load(f)
    
    params['pool_set'] = al_data['dataset']  
    al_indices = al_data["selected"]  
    
    # Get namepsace and update it with the parameters
    args = get_arguments()
    update_namespace(args, params)

    # System directory
    run_path = main_params['global']['run_path']
    system_dir = os.path.join(run_path,task_name, f'iter_{iter_idx}',system_name)
    # Move to the system directory and run the simulation
    if not os.path.exists(system_dir):
        os.makedirs(system_dir)
    os.chdir(system_dir)

    # Save parsed arguments
    with open(os.path.join( "arguments.json"), "w") as f:
        json.dump(vars(args), f)

    # Set up dataframe and load possible converged data id's
    db = connect('dft_structures.db')
    db_al_ind = [row.al_ind for row in db.select([('converged','=','True')])]
    
    # Get images and set parameters
    if args.label_set:
        images = read(args.label_set, index = ':')
    elif args.pool_set:
        if isinstance(args.pool_set, list):
            pool_traj = []
            for pool_path  in args.pool_set:
                if Path(pool_path).stat().st_size > 0:
                    pool_traj += read(pool_path, index=':')
        else:
            pool_traj = read(args.pool_set,index=':')
        
        if db_al_ind:
            _,rm,_ = np.intersect1d(al_indices, db_al_ind,return_indices=True)
            al_indices = np.delete(al_indices,rm)
        images = [pool_traj[i] for i in al_indices]        
    else:
        raise RuntimeError('Valid configarations for DFT calculation should be provided!')
    
    vasp_params = params['VASP']
    gpaw_params = params['GPAW']
    check_result = False

    if params['method'] =='VASP':
        from ase.calculators.vasp import Vasp
        # set environment variables
        os.putenv('ASE_VASP_VDW', '/home/energy/modules/software/VASP/vasp-potpaw-5.4')
        os.putenv('VASP_PP_PATH', '/home/energy/modules/software/VASP/vasp-potpaw-5.4')
        os.putenv('ASE_VASP_COMMAND', 'mpirun vasp_std')

        # Update the parameters
        vasp_params.update(system_params)

        calc = Vasp(**vasp_params)
        unconverged_idx = []
        for i, atoms in enumerate(images):
            al_ind = al_indices[i]
            atoms.set_pbc([True,True,True])
            atoms.set_calculator(calc)
            try:
                atoms.get_potential_energy()
            except CalculationFailed:
                check_result = True
                db.write(atoms,al_ind=al_ind,converged=False)
                unconverged_idx.append(i)
                copy('OSZICAR', f'OSZICAR_{i}_{al_ind}')
                copy('CHGCAR', f'CHGCAR_{i}_{al_ind}')
                os.remove('CHGCAR')
                continue

            steps = int(subprocess.getoutput('grep LOOP OUTCAR | wc -l'))
            if steps <= vasp_params['nelm']:
                db.write(atoms,al_ind=al_ind,converged=True)
            else:
                check_result = True
                db.write(atoms,al_ind=al_ind,converged=False)
                unconverged_idx.append(i)
            copy('OSZICAR', f'OSZICAR_{i}_{al_ind}')
            copy('CHGCAR', 'CHGCAR_{i}_{al_ind}')

            os.remove('WAVECAR')
            os.remove('CHGCAR')

    elif params['method'] =='GPAW':
        from gpaw import GPAW, KohnShamConvergenceError
        gpaw_params.update(system_params)
        calc = GPAW(**gpaw_params)
        calc.set(txt='GPAW.txt')
        unconverged_idx = []
        for i, atoms in enumerate(images):
            al_ind = al_indices[i]
            atoms.set_pbc([True,True,True])
            atoms.set_calculator(calc)
            try:
                atoms.get_potential_energy()
            except KohnShamConvergenceError:
                check_result = True
                db.write(atoms,al_ind=al_ind,converged=False)
                unconverged_idx.append(i)
                copy('GPAW.txt', f'GPAW_{i}_{al_ind}.txt')
                continue

            db.write(atoms,al_ind=al_ind,converged=True)
            copy('GPAW.txt', f'GPAW_{i}_{al_ind}.txt')
    else:
        raise RuntimeError('Valid configarations for DFT calculation should be provided!')

    
    if check_result:
        raise RuntimeError(f"DFT calculations of {unconverged_idx} are not converged!")
    # write to training set
    if args.train_set:
        train_traj = Trajectory(args.train_set, mode = 'a')
        database = connect('dft_structures.db')#read('dft_structures.traj', ':')
        for row in database.select([('converged','=','True')]):
            atom = row.toatoms()
            atom.info['system'] = args.system
            atom.info['path'] = str(Path('dft_structures.db').resolve())
            train_traj.write(atom)
    
    # Find how many models to train and return to the run directory
    # Load them to the main parameters again to give the opertunity to change the number of MLP doing the labeling step
    with open(cfg, 'r') as f:
        main_params = toml.load(f)
    params_train = main_params['train']
    if 'ensemble' in params_train:
        dmkey = len(list(params_train['ensemble'].keys()))
    else:
        dmkey = 1

    return True, {DYNAMICWIDTHGROUP_KEY: dmkey}
if __name__ == "__main__":
    main()
