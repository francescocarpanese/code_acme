import tensorflow as tf
from acme.agents.tf import actors
from acme.tf import utils as tf2_utils
from acme import specs
import numpy as np
from acme import wrappers
import matplotlib.pyplot as plt
import toml
from environments.dm_control import tank, MovingCoil0D

import pandas as pd
from dm_control.rl.control import Environment

store_path = '/root/acme/ab7bc9d0-714d-11ec-bd41-0242ac110002/'
store_path = '/root/acme/b2baf376-7150-11ec-ba54-0242ac110002/'
store_path = '/root/acme/5f76dae4-7151-11ec-941b-0242ac110002/'

# Read configuration file from folder
def get_dict(filename):
    with open(store_path + 'parameters/' + filename) as file:
        return  toml.load(file)

physics_par = get_dict('phys_par.toml')
task_par = get_dict('task_par.toml')

# Generate physics env and task with parameters as during training
if physics_par['phys_name'] == 'tank':
    physics = tank.Physics.physics(**physics_par)
    task = tank.Tasks.Step(**task_par)
elif physics_par['phys_name'] == 'MovingCoil':
    physics = MovingCoil0D.Physics.physics(**physics_par)
    task = MovingCoil0D.Tasks.Step(**task_par)
else:
    raise NameError(f'Physics env {physics_par["phys_name"]} not available')

# Set debug to True to store episode data
task.par_dict['debug'] = True

# Instantiate env
env = Environment(physics, task, time_limit=2. )  
env = wrappers.CanonicalSpecWrapper(env, clip= True) # Clip actions by bounds
env = wrappers.SinglePrecisionWrapper(env) 

# Load policy snapshot data
folder_path =  store_path +'snapshots/policy'
env_logs_path = store_path + '/logs/environment_loop/logs.csv'
env_logs = pd.read_csv(env_logs_path)

# Load policy 
policy_net = tf.saved_model.load(folder_path)

# Create the actor which defines how we take actions.
actor = actors.FeedForwardActor(
    policy_network=policy_net)

# Replay policy
timestep = env.reset()
while not timestep.last():
    action = actor.select_action(timestep.observation) 
    timestep = env.step(action)

# Fetch sim data
data = MovingCoil0D.Tasks.pack_datadict(env.task.datadict)

# Plot all time traces in episode dictionary
for key in [k for k in data.keys() if k != 'time']:
    plt.plot(data['time'],data[key])
    plt.ylabel(key)
    plt.xlabel('time')
    plt.show()


plt.plot(env_logs.episodes, env_logs.episode_return)
plt.xlabel('episodes')
plt.ylabel('epsisode return')
plt.show()


plt.plot(env_logs.episodes, env_logs.steps_per_second)
plt.xlabel('episodes')
plt.ylabel('sps')
plt.show()
