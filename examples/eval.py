from itertools import count
from tensorflow._api.v2 import saved_model
import tensorflow as tf
from acme.tf.savers import Snapshotter
from acme.agents.tf import actors
from acme.tf.networks.distributional import MultivariateNormalDiagHead
import sonnet as snt
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme import specs
import numpy as np
from acme import wrappers
import matplotlib.pyplot as plt
from environments.tank_dm_control import tank
from environments.tank_dm_control import Tasks
import pandas as pd
from dm_control.rl.control import Environment

# dmpo
store_path = '/root/acme/f3bb1052-6d3d-11ec-9e9e-0242ac110002/'
store_path = '/root/acme/f3bb1052-6d3d-11ec-9e9e-0242ac110002/'

# d4pg
#store_path = '/root/acme/3f9beb98-6edf-11ec-aab6-0242ac110002/'
#store_path = '/root/acme/1293cf06-6ee1-11ec-b046-0242ac110002/'
store_path = '/root/acme/1293cf06-6ee1-11ec-b046-0242ac110002/'

store_path = '/root/acme/f3bb1052-6d3d-11ec-9e9e-0242ac110002/'
store_path = '/root/acme/de08e7e2-6f0a-11ec-a830-0242ac110002/'
store_path = '/root/acme/4b7c5704-6f11-11ec-b118-0242ac110002/'

# Instantiate env
env = Environment(tank.physics(), Tasks.Step(debug = True, t_step=1.), time_limit=2. )  
env = wrappers.CanonicalSpecWrapper(env, clip= True) # Clip actions by bounds
# Env was store in single precision hence need 
# to use single precision to use saved net
env = wrappers.SinglePrecisionWrapper(env) 

# Specification for network IO
spec = specs.Array([2], dtype=np.float32)
inputs = tf2_utils.add_batch_dim(tf2_utils.zeros_like(spec))

# Load policy snapshot
folder_path =  store_path +'snapshots/policy'
env_logs_path = store_path + '/logs/environment_loop/logs.csv'
env_logs = pd.read_csv(env_logs_path)

# Extract solution for debugging
policy_net = tf.saved_model.load(folder_path)

# Check __call__ function return the desired solution
policy_net(inputs)

# Create the actor which defines how we take actions.
actor = actors.FeedForwardActor(
    policy_network=policy_net)

# Replay policy
timestep = env.reset()
while not timestep.last():
    action = actor.select_action(timestep.observation) 
    timestep = env.step(action)

# Fetch sim data
data = Tasks.pack_datadict(env.task.datadict)

# Plot all time traces
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
