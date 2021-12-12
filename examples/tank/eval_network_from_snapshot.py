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
from environments.tank import tank
from environments.tank import Tasks
import pandas as pd


# Instantiate env
env = tank.env(tend=2., task = Tasks.Step(), debug = True)
# Env was store in single precisio hence need 
# to use single precision to use saved net
env = wrappers.CanonicalSpecWrapper(env, clip= True) # Clip actions by bounds
env = wrappers.SinglePrecisionWrapper(env) 

# Specification for network IO
spec = specs.Array([2], dtype=np.float32)
inputs = tf2_utils.add_batch_dim(tf2_utils.zeros_like(spec))

store_path = '/root/acme/a7873ce0-5b38-11ec-ac5b-0242ac110002/'
store_path = '/root/acme/d867b0be-5b39-11ec-ad8a-0242ac110002/'
store_path = '/root/acme/f5a6e038-5b3d-11ec-a09d-0242ac110002/'
store_path = '/root/acme/dae95774-5b3f-11ec-a7c0-0242ac110002/'
store_path = '/root/acme/dae95774-5b3f-11ec-a7c0-0242ac110002/'

# Load snapshot
folder_path =  store_path +'snapshots/policy'
env_logs_path = store_path + '/logs/environment_loop/logs.csv'
env_logs = pd.read_csv(env_logs_path)

# Extract solution for debugging
policy_net = tf.saved_model.load(folder_path)

# Check __call__ function return the desired solution
policy_net(inputs)

# Create the behavior policy.
behavior_network = snt.Sequential([
        policy_net,
        networks.StochasticSamplingHead(),
    ])

# Create the actor which defines how we take actions.
actor = actors.FeedForwardActor(
    policy_network=behavior_network)

# Storing action and observation for debugging
#a_policy = np.zeros([1,N_max_steps])

timestep = env.reset()
while not timestep.last():
    # These actions are intended as 
    action = actor.select_action(timestep.observation) 
    timestep = env.step(action)

data = env.get_packed_datadict()

# Plot all time traces
for key in [k for k in data.keys() if k != 'time']:
    plt.plot(data['time'],data[key])
    plt.xlabel(key)
    plt.ylabel('time')
    plt.show()


plt.plot(env_logs.episodes, env_logs.episode_return)
plt.xlabel('episodes')
plt.ylabel('epsisode return')
plt.show()


plt.plot(env_logs.episodes, env_logs.steps_per_second)
plt.xlabel('episodes')
plt.ylabel('sps')
plt.show()

plt.plot(env_logs.episodes, env_logs.steps_per_second)
plt.xlabel('episodes')
plt.ylabel('sps')
plt.show()

plt.plot(env_logs.steps, env_logs.episode_return)
plt.xlabel('steps')
plt.ylabel('epsisode return')
plt.show()

