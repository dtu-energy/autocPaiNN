import os 
import ast
import toml
from workflow.md_run import MD
from workflow.neb_run import NEB_run

from perqueue.constants import DYNAMICWIDTHGROUP_KEY,CYCLICALGROUP_KEY, ITER_KW, INDEX_KW

def main(cfg,run_list,**kwargs):
    # Load perqueue index
    idx, *_ =kwargs[INDEX_KW]
    # Load iteration index
    iter_idx,*_ = kwargs[ITER_KW]
    
    # Load all parameters from config file
    with open(cfg, 'r') as f:
        main_params = toml.load(f)

    # Load local parameters
    task_name = 'simulate'
    params_simulate = main_params[task_name]

    # Get system name 
    run_list = ast.literal_eval(run_list)
    system_name = run_list[idx]

    # Get simulation method and parameters
    system_params = params_simulate['runs'][system_name]
    method = system_params['method']

    # Get the parameters for the simulation method
    params = params_simulate[method]

    # Update the simulation method parameters with the system name
    params.update(system_params)
    
    # Make folder for the system
    # Get run path
    run_path = main_params['global']['run_path']
    system_dir = os.path.join(run_path,task_name, f'iter_{iter_idx}',system_name)
    # Create the iteration directory
    try:
        os.makedirs(system_dir)
    except FileExistsError:
        pass

    # Add random seed and model path to the trained MLPs
    params['random_seed'] = main_params['global']['random_seed']
    params['model_path'] = os.path.join(run_path,'train', f'iter_{iter_idx}')

    # Move to the system directory and run the simulation
    os.chdir(system_dir)

    # Run the simulation methods
    if method == 'MD':
        # If iteration >0 then load the previous iteration MD trajcetory 
        if iter_idx > 0:
            # Load the previous iteration MD trajectory 
            MD_path = os.path.join(run_path,task_name, f'iter_{iter_idx-1}',system_name,'MD.traj')
            params['init_traj'] = MD_path

        # Run the MD simulation
        MD(params)

    elif method == 'NEB':
        NEB_run(params)

    else:
        raise NotImplementedError(f"Method {method} not implemented")
    
    # Return to the run directory
    return True, {'system_name':system_name}

if __name__ == "__main__":
    main()
