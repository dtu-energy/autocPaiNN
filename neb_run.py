from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read, write, Trajectory
from ase import units
from ase.db import connect
from ase.build import make_supercell
from ase.neb import NEB, NEBTools
from ase.optimize import BFGS
from ase.constraints import FixAtoms

import pandas as pd
import numpy as np
import torch
import os
import sys
from glob import glob
from pathlib import Path
import toml
import argparse
from pathlib import Path
import logging
from PaiNN.data import AseDataset, collate_atomsdata
from PaiNN.model import PainnModel
from PaiNN.calculator import MLCalculator, EnsembleCalculator

def setup_seed(seed):
     torch.manual_seed(seed)
     if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Run NEB simulation", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--load_model",
        type=str,
        help="Where to find the models",
    )
    parser.add_argument(
        "--initial_image",
        type=str,
        help="Path to initial structure directory",
    )
    parser.add_argument(
        "--final_image",
        type=str,
        help="Path to final structure directory",
    )
    parser.add_argument(
        "--fmax",
        type=int,
        #default=0.03,
        help="The maximum force along the NEB path",
    )
    parser.add_argument(
        "--num_img",
        type=int,
        #default=50,
        help="Number of images for NEB pathway",
    )
    parser.add_argument(
        "--num_MD",
        type=int,
        #default=5,
        help="Number of NEB images (sorted by max energies) to do small MD sim. upon. If None a small MD is done for all images",
    )
    parser.add_argument(
        "--time_MD",
        type=int,
        #default=5, #fs
        help="The time step of the small MD sim.",
    )
    parser.add_argument(
        "--time_step_MD",
        type=float,
        default=1,
        help="Time step of small MD simulation",
    )
    parser.add_argument(
        "--temperature_MD",
        type=float,
        #default=400,
        help="Temperture of small MD simulation",
    )
    parser.add_argument(
        "--friction_MD",
        type=float,
        #default=5,
        help="Setting the friction term in the Langevin dynamics",
    )
    parser.add_argument(
        "--print_step",
        type=int,
        default=1,
        help="Fix atoms under the specified value",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed for this run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Set which device to use for running MD e.g. 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="arguments.toml",
        help="Path to config file. e.g. 'arguments.toml'"
    )

    return parser.parse_args(arg_list)

def load_images(img_IS, img_FS):
    initial = read(img_IS)
    final = read(img_FS)
    return initial, final

def update_namespace(ns, d):
    for k, v in d.items():
        ns.__dict__[k] = v

class CallsCounter:
    def __init__(self, func):
        self.calls = 0
        self.func = func
    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.func(*args, **kwargs)

def main():
    args = get_arguments()
    if args.cfg:
        with open(args.cfg, 'r') as f:
            params = toml.load(f)
        update_namespace(args, params)

    setup_seed(args.random_seed)
    # set logger
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)

    runHandler = logging.FileHandler('neb.log', mode='w')
    runHandler.setLevel(logging.DEBUG)
    runHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    errorHandler = logging.FileHandler('neb_error.log', mode='w')
    errorHandler.setLevel(logging.WARNING)
    errorHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))

    logger.addHandler(runHandler)
    logger.addHandler(errorHandler)
    logger.addHandler(logging.StreamHandler())
    logger.warning = CallsCounter(logger.warning)
    logger.info = CallsCounter(logger.info)
    # load model
    model_pth = Path(args.load_model).rglob('*best_model.pth')
    models = []
    for each in model_pth:
        state_dict = torch.load(each) 
        model = PainnModel(
            num_interactions=state_dict["num_layer"], 
            hidden_state_size=state_dict["node_size"], 
            cutoff=state_dict["cutoff"],
        )
        model.to(args.device)
        model.load_state_dict(state_dict["model"])    
        models.append(model)

    encalc = EnsembleCalculator(models)
    # Read initial and final images from dft
    initial_image = args.initial_image
    final_image = args.final_image
    initial_dft, final_dft = load_images(initial_image, final_image)

    # Optimize intial and final images from dft using ML-FF
    logger.info('Optimizing images with ML')
    initial_dft.calc, final_dft.calc = encalc, encalc
    opt_initial = BFGS(initial_dft, trajectory='NEB_init_img.traj',logfile='initial_bfgs_img.log')
    opt_final = BFGS(final_dft, trajectory='NEB_final_img.traj',logfile='final_bfgs_img.log')
    opt_steps = 200
    opt_initial.run(fmax=args.fmax, steps=opt_steps)
    opt_final.run(fmax=args.fmax, steps=opt_steps)
    # Loading the optimized structures for initial and final image
    traj_inital = Trajectory('NEB_init_img.traj')
    traj_final = Trajectory('NEB_final_img.traj')
    initial, final = traj_inital[-1], traj_final[-1]
    # Check if the systems converged:
    fmax_init = pd.read_csv('initial_bfgs_img.log',delimiter="\s+")['fmax'].values.astype(float)[-1]
    fmax_final = pd.read_csv('final_bfgs_img.log',delimiter="\s+")['fmax'].values.astype(float)[-1]
    if fmax_init >args.fmax or fmax_final > args.fmax:
        sys.exit(f'Maximum optimization step reached. System did not converged. Check your system or increase the number of optimization steps from {opt_steps} steps')

    
    # Set calculator for each image
    images = [initial]
    for i in range(args.num_img):
        image = initial.copy()
        image.calc = encalc
        images.append(image)
    images.append(final)
    print(images)
    # Remove the vaccancies
    

    # Load NEB
    neb = NEB(images,allow_shared_calculator=True,climb=True) # 
    # Interpolate linearly the positions of all middle images
    neb.interpolate(mic=True)
    write('neb_initial.traj', images)
    
    # Set up the calculator for the intermediate images
    #for im in images[1:args.num_img-1]:
    #    im.calc = encalc

    logger.info('Optimize')
    # Optimize:
    optimizer = BFGS(neb, trajectory='NEB.traj',logfile='neb_bfgs.log')

    def printenergy(a=images):  # store a reference to atoms in the definition.
        """Function to print the potential, kinetic and total energy."""
        for i in range(1,args.num_img+1):
            ensemble = a[i].calc.results['ensemble']
            epot = a[i].get_potential_energy()
            ekin = a[i].get_kinetic_energy()
            temp = ekin / (1.5 * units.kB) / a[i].get_global_number_of_atoms()
            energy_var = ensemble['energy_var']
            forces_var = np.mean(ensemble['forces_var'])
            forces_sd = np.mean(np.sqrt(ensemble['forces_var']))
            forces_l2_var = np.mean(ensemble['forces_l2_var'])
            if forces_sd > 0.1: # was 0.05
                logger.warning(f"Image={i} Epot={epot} energy_var={energy_var} max_var={forces_var} forces_sd={forces_sd} forces_l2_var={forces_l2_var}")
            else:
                logger.info(f"Image={i} Epot={epot} energy_var={energy_var} max_var={forces_var} forces_sd={forces_sd} forces_l2_var={forces_l2_var}")
    
    optimizer.attach(printenergy, interval=args.print_step)
    logger.info('run')
    optimizer.run(fmax=args.fmax)

    

    ### Do small MD sim on a few NEB images
    # Finding the last structures of NEB.
    traj_old = Trajectory('NEB.traj')
    ind = args.num_img+2 #last x images, including initial and final image
    traj_NEB = traj_old[-ind:]
    write('NEB_MD.traj',traj_NEB)
    
    #Finding the largest energies
    if args.num_MD:
        nebtools = NEBTools(traj_NEB)
        E = nebtools.get_fit().energies
        sort_max = np.argsort(np.abs(E))[::-1]
        image_ind = sort_max[:args.num_MD] #np.random.choice(np.arange(0,len(traj_NEB)),args.num_MD,replace=False)
    else:
        image_ind = np.arange(0,ind)
    

    logger.info('random indices:')
    logger.info(image_ind)
    for i in image_ind:
        atoms = traj_NEB[i]
        atoms.calc = encalc
        atoms.get_potential_energy()
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_MD)
        dyn = Langevin(atoms, args.time_step_MD * units.fs, temperature_K=args.temperature_MD, friction=args.friction_MD)

        def printenergy_MD(a=atoms):  # store a reference to atoms in the definition.
            """Function to print the potential, kinetic and total energy."""
            ensemble = a.calc.results['ensemble']
            epot = a.get_potential_energy()
            ekin = a.get_kinetic_energy()
            temp = ekin / (1.5 * units.kB) / a.get_global_number_of_atoms()
            energy_var = ensemble['energy_var']
            forces_var = np.mean(ensemble['forces_var'])
            forces_sd = np.mean(np.sqrt(ensemble['forces_var']))
            forces_l2_var = np.mean(ensemble['forces_l2_var'])
            if forces_sd > 0.1: # was 0.05
                logger.warning(f"Image={i} Epot={epot} Ekin={ekin} temperature={temp} energy_var={energy_var} max_var={forces_var} forces_sd={forces_sd} forces_l2_var={forces_l2_var}")
            else:
                logger.info(f"Image={i} Epot={epot} Ekin={ekin} temperature={temp} energy_var={energy_var} max_var={forces_var} forces_sd={forces_sd} forces_l2_var={forces_l2_var}")
        
        dyn.attach(printenergy_MD, interval=1)
        
        logger.debug(f'MD starts for Image: {i} for {args.time_step_MD} fs')
        traj = Trajectory('NEB_MD.traj', 'a',atoms)
        dyn.attach(traj.write, interval=args.time_step_MD)
        dyn.run(args.time_MD)
    # # Write args to textfile:
    # with open(path_fix_traj + "arguments.txt", "w") as f:
    #     f.write("\n".join(['Removed magnesium', str(args.remove), 'Steps',
    #      str(steps),'T',str(T),'friction',str(friction)]))

    # # We also want to save the positions of all atoms after every 100th time step.
    # traj = Trajectory(path_fix_traj + traj_name + ".traj", "w", atoms)
    # dyn.attach(traj.write, interval=1)

    # dyn.run(steps)

if __name__ == "__main__":
    main()