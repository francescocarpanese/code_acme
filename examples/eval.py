"""Simple script to visualize trained policy"""
import tensorflow as tf
from acme.agents.tf import actors
from acme import wrappers
import matplotlib.pyplot as plt
import toml
import pandas as pd
from environments.dm_control.utils import episode_data
from dm_control.rl.control import Environment
from environments.dm_control import tank, moving_coil

STORE_PATH = '/root/acme/842e55d4-74b1-11ec-a60d-0242ac110002/'

def get_dict(filename):
    """"Read configuration file from folder"""
    with open(STORE_PATH + 'parameters/' + filename) as file:
        return  toml.load(file)

physics_par = get_dict('phys_par.toml')
task_par = get_dict('task_par.toml')

# Generate physics env and task with parameters as during training
if physics_par['phys_name'] == 'tank':
    physics = tank.physics.Physics(**physics_par)
    task = tank.tasks.Step(**task_par)
elif physics_par['phys_name'] == 'MovingCoil':
    physics = moving_coil.physics.Physics(**physics_par)
    task = moving_coil.tasks.Step(**task_par)
else:
    raise NameError(f'Physics env {physics_par["phys_name"]} not available')

# Set debug to True to store episode data
task.par_dict['debug'] = True

# Instantiate env
env = Environment(physics, task, time_limit=2.)  
env = wrappers.CanonicalSpecWrapper(env, clip=True) # Clip actions by bounds
env = wrappers.SinglePrecisionWrapper(env)

# Load policy snapshot data
FOLDER_PATH =  STORE_PATH +'snapshots/policy'
ENV_LOGS_PATH = STORE_PATH + '/logs/environment_loop/logs.csv'
env_logs = pd.read_csv(ENV_LOGS_PATH)

# Load policy
policy_net = tf.saved_model.load(FOLDER_PATH)

# Create the actor which defines how we take actions.
actor = actors.FeedForwardActor(
    policy_network=policy_net)

# Replay policy
timestep = env.reset()
while not timestep.last():
    action = actor.select_action(timestep.observation)
    timestep = env.step(action)

# Fetch sim data
data = episode_data.pack_datadict(env.task.datadict)

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
