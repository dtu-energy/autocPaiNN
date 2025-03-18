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
import re

from perqueue.constants import DYNAMICWIDTHGROUP_KEY,CYCLICALGROUP_KEY, ITER_KW, INDEX_KW

def get_converged_structures(INCAR_file,OSZICAR_file) -> list:
    """
    Get the converged structures from OUTCAR file
    Parameters
    ----------
    root_dir : str
        The root directory of the calculation

    Returns
    -------
    list of ase.Atoms objects 
    """
    # Get nelm from INCAR
    with open(INCAR_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'NELM = ' in line:
                nelm =int(line.split('=')[-1].strip())
                break
    # Get index of converged calculations
    scf_list = []
    with open(OSZICAR_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'F=' in line:
                header = lines[i-1].split(':')[0]
                scf = int(re.search(header+r':\s+(\d+)', lines[i-1]).group(1))
                scf_list.append(scf)
    # Single point calculation
    if len(scf_list) == 1:          
        if scf != nelm:
            return True
        else:
            return False
    # Relaxation calculation
    else:
        if scf_list[-1] != nelm:
            return True
        else:
            return False

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

def update_namespace(ns:argparse.Namespace, d:dict) -> None:
    """

    Update the namespace with the dictionary.

    Args:
        ns: The namespace to update
        d: The dictionary to update the namespace with
    
    """
    for k, v in d.items():
        if not ns.__dict__.get(k):
            ns.__dict__[k] = v

def main(cfg,system_name,**kwargs):

    # Load iteration index
    iter_idx,*_ = kwargs[ITER_KW]

    # Find how many models to train and return to the run directory
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
