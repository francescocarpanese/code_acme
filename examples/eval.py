"""Simple script to visualize trained policy"""
from absl import app
from absl import flags
import tensorflow as tf
from acme.agents.tf import actors
from acme import wrappers
import matplotlib.pyplot as plt
import toml
import pandas as pd
from environments.dm_control.utils import episode_data
from dm_control.rl.control import Environment
from environments.dm_control import tank, moving_coil, moving_coil2D
import os

# FLAGS definition
flags.DEFINE_string(
    'train_out_folder_path',
    './trainig_outputs',
    'Define output folder path',
     )
flags.DEFINE_string(
    'plot_path',
    './plots',
    'Define output folder path',
     )   
flags.DEFINE_float('time_limit', 2., 'End simulation time [s]')  
FLAGS = flags.FLAGS


def get_dict(filename):
    """"Read configuration file from folder"""
    with open(FLAGS.train_out_folder_path + '/' + 'parameters/' + filename) as file:
        return  toml.load(file)

def gen_phys_task(physics_par, task_par):
    # Generate physics env and task with parameters as during training
    if physics_par['phys_name'] == 'tank':
        physics = tank.physics.Physics(**physics_par)
        task = tank.tasks.Step(**task_par)
    elif physics_par['phys_name'] == 'MovingCoil':
        physics = moving_coil.physics.Physics(**physics_par)
        task = moving_coil.tasks.Step(**task_par)
    elif physics_par['phys_name'] == 'MovingCoil2D':
        physics = moving_coil2D.physics.Physics(**physics_par)
        task = moving_coil2D.tasks.Step(**task_par)
    else:
        raise NameError(f'Physics env {physics_par["phys_name"]} not available')
    return physics,task

def make_plot(data, logs):
    # Create folder if not existing 
    if not os.path.exists(FLAGS.plot_path):
        os.makedirs(FLAGS.plot_path)

    # Plot all time traces in episode dictionary
    for key in [k for k in data.keys() if k != 'time']:
        plt.figure()
        plt.plot(data['time'],data[key])
        plt.ylabel(key)
        plt.xlabel('time')
        plt.show()
        plt.savefig(FLAGS.plot_path + '/' + key + '.png')

    # Plot episode return over episodes
    plt.figure()  
    plt.plot(logs.episodes, logs.episode_return)
    plt.xlabel('episodes')
    plt.ylabel('epsisode return')
    plt.show()
    plt.savefig(FLAGS.plot_path + '/' + 'episode_return' + '.png')

    # Plot sps over episodes
    plt.figure()  
    plt.plot(logs.episodes, logs.steps_per_second)
    plt.xlabel('episodes')
    plt.ylabel('sps')
    plt.show()
    plt.savefig(FLAGS.plot_path + '/' + 'sps' + '.png')

def main(_):
    # Load parameters 
    physics_par = get_dict('phys_par.toml')
    task_par = get_dict('task_par.toml')

    # Generate env 
    physics, task = gen_phys_task(physics_par, task_par)

    # Set debug to True to store episode data during running
    task.par_dict['debug'] = True

    # Instantiate env
    env = Environment(physics, task, time_limit=FLAGS.time_limit)  
    env = wrappers.CanonicalSpecWrapper(env, clip=True) # Clip actions by bounds
    env = wrappers.SinglePrecisionWrapper(env)

    # Load policy snapshot data
    policy_path =  FLAGS.train_out_folder_path + '/' + 'snapshots/policy'

    # Load policy
    policy_net = tf.saved_model.load(policy_path)

    # Create the actor.
    actor = actors.FeedForwardActor(
        policy_network=policy_net)

    # Replay policy
    timestep = env.reset()
    while not timestep.last():
        action = actor.select_action(timestep.observation)
        timestep = env.step(action)

    # Fetch sim data
    data = episode_data.pack_datadict(env.task.datadict)
    
    ENV_LOGS_PATH = FLAGS.train_out_folder_path + '/' + '/logs/environment_loop/logs.csv'
    env_logs = pd.read_csv(ENV_LOGS_PATH)

    # Generate plots
    make_plot(data, env_logs)

if __name__ == '__main__':
    app.run(main)