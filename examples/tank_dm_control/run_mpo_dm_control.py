# python3
"""Example running MPO on the tank env locally."""

from threading import active_count
from typing import Dict, Sequence

from absl import app
from absl import flags
from acme import specs
from acme import types
from acme.environment_loop import EnvironmentLoop
from acme.wrappers.single_precision import SinglePrecisionWrapper
from acme.wrappers.canonical_spec import CanonicalSpecWrapper
from acme.agents.tf import dmpo
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils.loggers.tf_summary import TFSummaryLogger
from acme.utils import loggers
import dm_env
import numpy as np
import sonnet as snt
from acme.utils import paths
import json

from environments.tank_dm_control import tank
from environments.tank_dm_control import Tasks
from dm_control.rl.control import Environment

flags.DEFINE_integer('num_episodes', 150, 'Number of episodes to run for.')
flags.DEFINE_float('time_limit', 2., 'End simulation time [s]')
flags.DEFINE_integer('batch_size', 40, 'batch size replay buffer')
flags.DEFINE_list('policy_layer_sizes', ['64','64'], 'MLP layer size policy net')
flags.DEFINE_list('critic_layer_sizes', ['64','64'], 'MLP layer size critic net')
FLAGS = flags.FLAGS

def make_environment() -> dm_env.Environment:
  """Creates environment."""
  environment = Environment(
    tank.physics(),
    Tasks.Step(t_step= 1.),
    time_limit=FLAGS.time_limit,
  )  
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
    policy_layer_sizes: Sequence[int] = (64,64),
    critic_layer_sizes: Sequence[int] = (64,64),
    vmin: float = 0.,
    vmax: float = 40.,
    num_atoms: int = 300,
    ) -> Dict[str, types.TensorTransformation]:
  """Creates networks used by the agent."""

  # Get total number of action dimensions from action spec.
  num_dimensions = np.prod(action_spec.shape, dtype=int)

  # Create the shared observation network; here simply a state-less operation.
  observation_network = tf2_utils.batch_concat

  # Create the policy network.
  policy_network = snt.Sequential([
      networks.LayerNormMLP(policy_layer_sizes),
      networks.MultivariateNormalDiagHead(num_dimensions),
  ])

  # The multiplexer transforms concatenates the observations/actions.
  multiplexer = networks.CriticMultiplexer(
      critic_network=networks.LayerNormMLP(critic_layer_sizes),
      action_network=networks.ClipToSpec(action_spec))

  # Create the critic network.
  critic_network = snt.Sequential([
      multiplexer,
      networks.DiscreteValuedHead(vmin, vmax, num_atoms),
  ])

  return {
      'policy': policy_network,
      'critic': critic_network,
      'observation': observation_network,
  }

def store_parameters(env):
  """ 
  A new folder is generated for every new process with unique id. 
  Path available with path.get_unique_id() 
  """
  out_path = paths.process_path('~/acme', 'parameters')
  write_f_json(out_path, 'phys_par', env._physics.get_par_dict())
  write_f_json(out_path, 'task_par', env._task.get_par_dict())

def write_f_json(path,fname,dictionary):
  # Serializing json 
  json_object = json.dumps(dictionary, indent = 4)
  
  # Write output file
  with open( path + "/" + fname +".json", "w") as outfile:
    outfile.write(json_object)


def main(_):
  # Create an environment and grab the spec.
  environment = make_environment()
  environment_spec = specs.make_environment_spec(environment)

  # Create the networks to optimize (online) and target networks.
  agent_networks = make_networks(
      environment_spec.actions,
      policy_layer_sizes= tuple(map(int,FLAGS.policy_layer_sizes)),
      critic_layer_sizes= tuple(map(int,FLAGS.critic_layer_sizes))
  )

  # Construct the agent.
  agent = dmpo.DistributionalMPO(
      environment_spec=environment_spec,
      policy_network=agent_networks['policy'],
      critic_network=agent_networks['critic'],
      observation_network=agent_networks['observation'], 
      batch_size = FLAGS.batch_size,
      target_policy_update_period = 20,
      target_critic_update_period = 20,
      min_replay_size = 10,
  )

  # Store the running parameters in process related folder
  store_parameters(environment)

  # Run the environment loop.
  loop = EnvironmentLoop(environment, agent)
  loop.run(num_episodes=FLAGS.num_episodes)

  # Force snapshot storing of latest behavior policy
  agent._learner._snapshotter.save(force=True)
  
if __name__ == '__main__':
  app.run(main)