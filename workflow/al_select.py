import numpy as np
import os 
import json
import argparse, toml
from pathlib import Path
import logging
from ase.io import read
import torch

from perqueue.constants import DYNAMICWIDTHGROUP_KEY,CYCLICALGROUP_KEY, ITER_KW, INDEX_KW

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="General Active Learning", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--method",
        type=str,
        help="Set if you do NEB or MD",
    )
    parser.add_argument(
        "--neb_init",
        type=bool,
        default= True,
        help="if True: Choose all the inital NEB images(not MD) to be labeled including initial and final image. Do not work if you have set dataset as a param",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        help="How to get features",
    )
    parser.add_argument(
        "--selection",
        type=str,
        help="Selection method, one of `max_dist_greedy`, `deterministic_CUR`, `lcmd_greedy`, `max_det_greedy` or `max_diag`",
    )
    parser.add_argument(
        "--n_random_features",
        type=int,
        help="If `n_random_features = 0`, do not use random projections.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="How many data points should be selected",
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
        "--dataset", type=str, help="Path to ASE trajectory",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        help="Train/test/validation split file json",
    )
    parser.add_argument(
        "--pool_set", type=str, help="Path to MD trajectory obtained from machine learning potential",
    )
    parser.add_argument(
        "--train_set", type=str, help="Path to training set. Useful for pool/train based selection method",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed for this run",
    )
    return parser.parse_args(arg_list)

def update_namespace(ns, d):
    for k, v in d.items():
        if not ns.__dict__.get(k):
            ns.__dict__[k] = v

def main(cfg,system_name,**kwargs):
    from cPaiNN.active_learning import GeneralActiveLearning
    from cPaiNN.data import AseDataset
    from cPaiNN.relax import ML_Relaxer
    from workflow.train_functions import setup_seed

    # Create device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # Put a tensor on the device before loading data
    # This way the GPU appears to be in use when other users run gpustat
    torch.tensor([0], device=device)

    # Load iteration index
    iter_idx,*_ = kwargs[ITER_KW]
    
    # Load all parameters from config file
    with open(cfg, 'r') as f:
        main_params = toml.load(f)

    # Load local parameters
    task_name = 'select'
    params = main_params[task_name]

    # Add random seed and model path to the trained MLPs
    params['random_seed'] = main_params['global']['random_seed']
    run_path = main_params['global']['run_path']
    params['model_path'] = os.path.join(run_path,'train', f'iter_{iter_idx}')
    
    # Load the system parameters
    system_params = params['runs'][system_name]
    method = system_params['method']
    params['method'] = method

    # Find the system dataset and update the parameters
    sytem_path = os.path.join(run_path,'simulate', f'iter_{iter_idx}',system_name)
    if method == 'MD':
        pool_set = [os.path.join(sytem_path,'MD.traj'),os.path.join(sytem_path,'warning_struct.traj')]
    elif method == 'NEB':
        pool_set = [os.path.join(sytem_path,'NEB_MD.traj')]
        args_system_simulate = json.load(open(os.path.join(sytem_path,'arguments.json')))
        neb_img = args_system_simulate['num_img'] + 2 # Including initial and final image
    
    params['pool_set'] = pool_set 
    # Update the main parameters with the system parameters
    params.update(system_params)

    # Get the Namespace and update the parameters
    args = get_arguments()
    update_namespace(args, params)

    setup_seed(args.random_seed)

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

    # Load models
    ML_class = ML_Relaxer(calc_name=args.model_name,calc_paths=args.model_path,device=device)
    if not ML_class.ensemble:
        raise NotImplementedError("Only ensemble training is supported at the moment")
    models = ML_class.models

    # Test if models is a list and there is a pool set and train set
    assert isinstance(models, list)
    assert args.pool_set and args.train_set 

    # set logger
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                "log.txt", mode="w"
            ),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Selected system: {system_name}")
    logging.info(f"Selected method: {args.method}")

    logging.info(f"Loding the dataset from {args.pool_set} and {args.train_set}")
    # Load pool set and train set
    if isinstance(args.pool_set, list):
        dataset = []
        for traj in args.pool_set:
            if Path(traj).stat().st_size > 0:
                dataset += read(traj, index=':') 
    data_dict = {
        'pool': AseDataset(dataset, cutoff=models[0].cutoff),
        'train': AseDataset(args.train_set, cutoff=models[0].cutoff),
        #'train': AseDataset(read(args.train_set, index=':1000'), cutoff=models[0].cutoff),
    }

    logging.info(f"Train set size: {len(data_dict['train'])}")
    logging.info(f"Pool set size: {len(data_dict['pool'])}")

    # raise error if the pool dataset is not large enough
    if args.method =='MD':
        if len(data_dict['pool']) < args.batch_size*10 :
            raise RuntimeError(f"""The pool data set ({len(data_dict['pool'])}) is not large enough for selection!
            It should be larger than 10 times batch size ({args.batch_size*10}).
            Check your MD simulation!""")

    # Select structures
    al = GeneralActiveLearning(
        kernel=args.kernel, 
        selection=args.selection, 
        n_random_features=args.n_random_features,
    )
    logging.info(f"Selecting {args.batch_size} structures")
    # Manually choose N NEB images to be labeled.
    if args.method == 'NEB':
        indices_neb = np.arange(0,neb_img)
        if neb_img > args.batch_size:
            raise RuntimeError(f"""The pool data set is not large enough for selection! 
            Choose a batch size ({args.batch_size}) larger then NEB selection ({neb_img})!""")
        if neb_img == args.batch_size:
            indices = indices_neb.tolist()
        else:
            data_dict['pool'] = AseDataset(dataset[neb_img:], cutoff=models[0].cutoff)
            indices_al = al.select(models, data_dict, al_batch_size=args.batch_size-neb_img)
            indices = np.concatenate((indices_neb,np.array(indices_al)+neb_img),dtype=int).tolist()
    
    # Choose N MD images to be labeled.
    elif args.method == 'MD':
        indices = al.select(models, data_dict, al_batch_size=args.batch_size)
    else:
        raise RuntimeError("Please give valid method for selection!")
    
    al_idx = indices
    al_info = {
        'kernel': args.kernel,
        'selection': args.selection,
        'dataset': args.pool_set,
        'selected': al_idx,
    }

    with open('selected.json', 'w') as f:
        json.dump(al_info, f)

    return True, {'system_name':system_name}

if __name__ == "__main__":
    main()
