# python3
"""Example training agents on dm_control environments"""

from typing import Dict, Sequence

from absl import app
from absl import flags
from acme import agents, specs
from acme import types
from acme.environment_loop import EnvironmentLoop
from acme.wrappers.single_precision import SinglePrecisionWrapper
from acme.wrappers.canonical_spec import CanonicalSpecWrapper
from acme.agents.tf import dmpo, d4pg, mpo
from acme.tf import networks
from acme.tf import utils as tf2_utils
import dm_env
import numpy as np
import sonnet as snt
from acme.utils import paths
import json
import tensorflow as tf 

from environments.dm_control import tank, MovingCoil0D

from dm_control.rl.control import Environment

flags.DEFINE_integer('num_episodes', 120, 'Number of episodes to run for.')
flags.DEFINE_float('time_limit', 2., 'End simulation time [s]')
flags.DEFINE_string('agent', 'dmpo', 'Choose agent ["dmpo", "d4pg", "mpo"] ')
flags.DEFINE_string('environment', 'tank', 'Choose environment ["tank","moving_coil"]')
FLAGS = flags.FLAGS

# Set random seed for example reproducibility
tf.random.set_seed(
  1500
)

def make_physics_task():
  # Select environment and task
  if FLAGS.environment == 'tank':
    return tank.Physics.physics(), tank.Tasks.Step(t_step = 1.)
  elif FLAGS.environment == 'moving_coil':
    return MovingCoil0D.Physics.physics(), MovingCoil0D.Tasks.Step(t_step = 1.)
  else:
    raise ValueError(f'Environment {FLAGS.environment} not available')

def make_environment() -> dm_env.Environment:
  """Creates environment."""
  physics, task = make_physics_task()

  environment = Environment(physics, task, time_limit=FLAGS.time_limit)  
  # Clip actions by bounds
  environment = CanonicalSpecWrapper(
    environment= environment,
    clip= True, 
    ) 
  # Wrap to single precision
  environment = SinglePrecisionWrapper(environment) 
  return environment

def make_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (8,8),
    critic_layer_sizes: Sequence[int] = (8,8),
    vmin: float = 0.,
    vmax: float = 40.,
    num_atoms: int = 300,
    ) -> Dict[str, types.TensorTransformation]:
  """Creates networks used by the agent."""

  # Get total number of action dimensions from action spec.
  num_dimensions = np.prod(action_spec.shape, dtype=int)

  # Create the shared observation network; here simply a state-less operation.
  observation_network = tf2_utils.batch_concat

  if FLAGS.agent == 'dmpo' or FLAGS.agent == 'mpo':
    # Specify default dimension
    policy_layer_sizes = [64,64]
    critic_layer_sizes = [64,64]
    # Create the policy network.
    policy_network = snt.Sequential([
      networks.LayerNormMLP(policy_layer_sizes),
      networks.MultivariateNormalDiagHead(num_dimensions),
    ])

  elif FLAGS.agent == 'd4pg':
    # Specify default dimension
    policy_layer_sizes = [8,8]
    critic_layer_sizes = [8,8]      
    # Create the policy network.
    policy_network = snt.Sequential([
      networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
      networks.NearZeroInitializedLinear(num_dimensions),
      networks.TanhToSpec(action_spec),
    ])

  # The multiplexer transforms concatenates the observations/actions.
  multiplexer = networks.CriticMultiplexer(
      critic_network=networks.LayerNormMLP(critic_layer_sizes),
      action_network=networks.ClipToSpec(action_spec))

  if FLAGS.agent == 'dmpo' or FLAGS.agent == 'd4pg':
    # Create the critic network.
    critic_network = snt.Sequential([
        multiplexer,
        networks.DiscreteValuedHead(vmin, vmax, num_atoms),
    ])
  
  elif FLAGS.agent == 'mpo':
    critic_layer_sizes = list(critic_layer_sizes) + [1] # Hack to conform to mpo implementation
    critic_network = networks.CriticMultiplexer(
      critic_network=networks.LayerNormMLP(critic_layer_sizes))

  return {
      'policy': policy_network,
      'critic': critic_network,
      'observation': observation_network,
  }

def make_agent(environment_spec, agent_networks):
  if FLAGS.agent == 'dmpo':
    # Construct the agent.
    agent = dmpo.DistributionalMPO(
      environment_spec=environment_spec,
      policy_network=agent_networks['policy'],
      critic_network=agent_networks['critic'],
      observation_network=agent_networks['observation'], 
      batch_size = 40,
      target_policy_update_period = 20,
      target_critic_update_period = 20,
      min_replay_size = 10,
    )
  elif FLAGS.agent == 'd4pg':
    # Construct the agent.
    agent = d4pg.D4PG(
      environment_spec=environment_spec,
      policy_network=agent_networks['policy'],
      critic_network=agent_networks['critic'],
      observation_network=agent_networks['observation'],  # pytype: disable=wrong-arg-types
      batch_size = 5,
      target_update_period = 10,
      min_replay_size = 50,
      max_replay_size = 10000, 
      n_step=10,
      sigma=0.6,
    )

  elif FLAGS.agent == 'mpo':
    agent = mpo.MPO(
      environment_spec=environment_spec,
      policy_network=agent_networks['policy'],
      critic_network=agent_networks['critic'],
      observation_network=agent_networks['observation'], 
      batch_size = 40,
      target_policy_update_period = 20,
      target_critic_update_period = 20,
      min_replay_size = 10,
    )
  
  return agent

def store_parameters(env):
  """ 
  A new folder is generated for every new process with unique id. 
  Path available with path.get_unique_id() 
  """
  out_path = paths.process_path('~/acme', 'parameters')
  env._physics.write_config_file( out_path  , '/phys_par')
  env._task.write_config_file(out_path, '/task_par')

def main(_):
  # Create an environment and grab the spec.
  environment = make_environment()
  environment_spec = specs.make_environment_spec(environment)

  # Create the networks to optimize (online) and target networks.
  agent_networks = make_networks(environment_spec.actions)

  # Create agent
  agent = make_agent(environment_spec, agent_networks)

  # Store the running parameters in process related folder
  store_parameters(environment)

  # Run the environment loop.
  loop = EnvironmentLoop(environment, agent)
  loop.run(num_episodes=FLAGS.num_episodes)

  # Force snapshot storing of latest behavior policy
  agent._learner._snapshotter.save(force=True)
  
if __name__ == '__main__':
  app.run(main)