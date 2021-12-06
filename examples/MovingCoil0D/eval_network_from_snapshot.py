from itertools import count
from tensorflow._api.v2 import saved_model
from environments.MovingCoil0D.Moving_Coil import Moving_Coil    
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

N_max_steps = 31

# Instantiate env
env = Moving_Coil(tend=3.)
# Env was store in single precisio hence need 
# to use single precision to use saved net
env = wrappers.SinglePrecisionWrapper(env) 
#environment = wrappers.StepLimitWrapper(env, N_max_steps) # Set max number of simulation time steps


# Specification for network IO
spec = specs.Array([2], dtype=np.float32)
inputs = tf2_utils.add_batch_dim(tf2_utils.zeros_like(spec))

# Load snapshot
folder_path = '/root/acme/0b0625b4-54ff-11ec-8aa5-0242ac110003/snapshots/policy/assets'

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
a = np.zeros([2,N_max_steps])
o = np.zeros([2,N_max_steps])

counter = 0

timestep = env.reset()
while not timestep.last():
    action = actor.select_action(timestep.observation)
    timestep = env.step(action)
    a[:,counter] = action
    o[:,counter] = timestep.observation
    counter += 1

plt.plot(np.transpose(o))
plt.show()

