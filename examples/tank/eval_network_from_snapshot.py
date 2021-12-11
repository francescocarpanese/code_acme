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

N_max_steps = 21

# Instantiate env
env = tank.env(tend=2., task = Tasks.HoldTarget())
# Env was store in single precisio hence need 
# to use single precision to use saved net
env = wrappers.SinglePrecisionWrapper(env) 
#environment = wrappers.StepLimitWrapper(env, N_max_steps) # Set max number of simulation time steps


# Specification for network IO
spec = specs.Array([2], dtype=np.float32)
inputs = tf2_utils.add_batch_dim(tf2_utils.zeros_like(spec))

# Load snapshot
folder_path =  '/root/acme/c1b5646e-5aae-11ec-89d6-0242ac110002/snapshots/policy'

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
a = np.zeros([1,N_max_steps])
o = np.zeros([2,N_max_steps])
r = np.zeros([1,N_max_steps])

counter = 0

timestep = env.reset()
while not timestep.last():
    action = actor.select_action(timestep.observation)
    timestep = env.step(action)
    a[:,counter] = action
    o[:,counter] = timestep.observation
    r[:,counter] = timestep.reward
    counter += 1

plt.title('Observation')
plt.plot(np.transpose(o))
plt.show()


plt.title('Reward')
plt.plot(np.transpose(r))
plt.show()


plt.title('Actions')
plt.plot(np.transpose(a))
plt.show()
