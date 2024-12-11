from pathlib import Path
import toml, os
import numpy as np
from perqueue import PersistentQueue
from perqueue.task_classes.task import Task
from perqueue.task_classes.task_groups import CyclicalGroup, Workflow, DynamicWidthGroup, StaticWidthGroup
from perqueue.constants import DYNAMICWIDTHGROUP_KEY

### Paths ###
main_path = Path('./') # path to the auto-cPaiNN code

### Load config file ###
config = toml.load(main_path/'config.toml')

print(config)

### Task resources ###
Train_resources = "8:sm3090el8:12h"
Simulation_resources = "8:sm3090el8:12h"
Activelearning_resources = "8:sm3090el8:12h"
Labeling_resources = "24:xeon24el8_test:30m"

### Set Tasks ###
main_argument = ['--config', main_path/'config.toml']

Train = Task(main_path/'workflow/train.py', name='Train MLP model', args=main_argument, resources=Train_resources)
Simulation = Task(main_path/'workflow/simulation.py', name='Run ML simulation', args=main_argument, resources=Simulation_resources)
Activelearning = Task(main_path/'workflow/al_select.py', name='Run active learning', args=main_argument, resources=Activelearning_resources)
Labeling = Task(main_path/'workflow/labeling.py', name='Run DFT labeling', args=main_argument, resources=Labeling_resources)

### Set Task Groups ###
# If you perform ensemble training more models are train or else one main model is trained 
if 'ensemble' in config['train']:
    swg_width = len(config['train']['ensemble'])
else:
    swg_width = 1

# Initial training of the MLP model
swg = StaticWidthGroup([Train], width=swg_width)

# The MLP labeling step where each structure follows each other independently
dwg_labeling = DynamicWidthGroup([Simulation,Activelearning,Labeling])

# The MLP training step where the model is trained after labeling. Here all structure needs to be finsihsed before the next training step
dwg_train = DynamicWidthGroup([Train])

# Combine the steps into a cyclical group
cg = CyclicalGroup([dwg_labeling, dwg_train], max_tries=config['global']['max_iteration'])

### Set Workflow ###

wf = Workflow({swg:[], cg:[swg]})

### Submit Workflow ###
#with PersistentQueue() as pq:
#    pq.submit(wf)