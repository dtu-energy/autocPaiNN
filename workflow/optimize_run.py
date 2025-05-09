from ase.io import read, write
from ase import units

import pandas as pd
import numpy as np
import torch
import os, json
import sys
import argparse
import logging
from cPaiNN.relax import ML_Relaxer
from cPaiNN.utils import setup_seed

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Run NEB simulation", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Where to find the models",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model. Default is cpainn",
        default='cpainn',
    )
    parser.add_argument(
        "--structure_path",
        type=str,
        help="Path to all to be optimized structures",
    )
    parser.add_argument(
        "--opt_steps",
        type=int,
        default=500,
        help="Number of optimization steps",
    )
    parser.add_argument(
        "--opt_fmax",
        type=int,
        #default=0.03,
        help="The maximum force for the optimization",
    )
    parser.add_argument(
        "--opt_algo",
        type=str,
        #default='LBFGSLineSearch',
        help="Algorithm used for optimization",
    )
    parser.add_argument(
        "--opt_cell",
        type=bool,
        default=True,
        help="Optimize the cell or not",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed for this run",
    )

    return parser.parse_args(arg_list)

def update_namespace(ns:argparse.Namespace, d:dict) -> None:
    """

    Update the namespace with the dictionary.

    Args:
        ns: The namespace to update
        d: The dictionary to update the namespace with
    
    """
    for k, v in d.items():
        
        ns.__dict__[k] = v

def optimize_run(params:dict,run_dir:str='.') -> None: 
    # Create device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Put a tensor on the device before loading data
    # This way the GPU appears to be in use when other users run gpustat
    torch.tensor([0], device=device)
    
    # Get arguments
    args = get_arguments()

    # Update namespace with the parameters from the config file
    update_namespace(args, params)

    # Save parsed arguments
    with open(os.path.join(run_dir, "arguments.json"), "w") as f:
        json.dump(vars(args), f)
    
    setup_seed(args.random_seed)
    
    # set logger
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)

    runHandler = logging.FileHandler(os.path.join(run_dir,'optimize.log'), mode='w')
    runHandler.setLevel(logging.DEBUG)
    runHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    logger.addHandler(runHandler)
    logger.addHandler(logging.StreamHandler())
    logging.info(f"Using device: {device}")
    
    # Load MLP model
    # Set up the MLP calculator
    relaxer = ML_Relaxer(calc_name=args.model_name,calc_paths=args.model_path,device=device,relax_cell=args.opt_cell)

    # Load all structures
    structures = read(args.structure_path, index=':')
    logger.info(f"Found {len(structures)} structures in {args.structure_path}")

    pool_set = []
    # Loop over all structures
    for i, atoms in enumerate(structures):
        # Set the calculator
        
        traj_path = os.path.join(run_dir, f"opt_{i}.xyz")
        log_path = os.path.join(run_dir, f"opt_{i}.log")
        logger.info(f"Optimizing structure {i} with {args.opt_algo} algorithm")

        relax_result = relaxer.relax(
            atoms,
            fmax=args.opt_fmax,
            steps=args.opt_steps,
            traj_file=traj_path,
            log_file=log_path,
        )
        pool_set.append(traj_path)
        logger.info(f"Optimized structure {i} saved to {traj_path}")
        logger.info(f"Final energy: {relax_result['final_structure'].get_potential_energy()}")
    
    # Save the pool set
    pool_set_path = os.path.join(run_dir, "pool_set.json")
    with open(pool_set_path, "w") as f:
        json.dump(pool_set, f)
    logger.info(f"Pool set saved to {pool_set_path}")
