# python3
"""Example running MPO on the Moving Coil env locally."""

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

from environments.tank_dm_control import tank
from environments.tank_dm_control import Tasks
import os
from dm_control.rl.control import Environment


flags.DEFINE_integer('num_episodes', 100, 'Number of episodes to run for.')
flags.DEFINE_float('tend', 2., 'Final simulation time [s]')
flags.DEFINE_string('task', 'Step', 'Defins task: ["Dummy","HoldTarget"]')
flags.DEFINE_string('out_path', './.tmp_train', 'Output path to store checkpoint and results' )
FLAGS = flags.FLAGS

# Create storing folder if not existing already
def create_out_folder(path):
  pass
#  if os.path.isdir(path):
#    raise RuntimeError('Failed to create output folder. Folder already exists')
#  os.mkdir(path)

# Task selector depending on flag
def make_task(task_name: str):
  if task_name == 'HoldTarget':
    return Tasks.Step()
  elif task_name == 'Dummy':
    return Tasks.Dummy()
  elif task_name == 'Step':
    return Tasks.Step(t_step= 1.)


def make_environment() -> dm_env.Environment:
  """Creates environment."""
  task = make_task(FLAGS.task)
  environment = Environment(tank.physics(), task, time_limit=2. )  
  environment = CanonicalSpecWrapper(environment= environment, clip= True) # Clip actions by bounds
  environment = SinglePrecisionWrapper(environment) 
  return environment

def make_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (64,64),
    critic_layer_sizes: Sequence[int] = (64,64),
    vmin: float = 0.,
    vmax: float = 20.,
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


def main(_):
  # Create out folder 
  create_out_folder(FLAGS.out_path)

  # Create an environment and grab the spec.
  environment = make_environment()
  environment_spec = specs.make_environment_spec(environment)

  # Create the networks to optimize (online) and target networks.
  agent_networks = make_networks(environment_spec.actions)

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

  # Run the environment loop.
  loop = EnvironmentLoop(environment, agent)
  loop.run(num_episodes=FLAGS.num_episodes)

  # Force snapshot storing of latest behavior policy
  agent._learner._snapshotter.save(force=True)
  
if __name__ == '__main__':
  app.run(main)