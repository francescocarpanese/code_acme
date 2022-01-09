""" Container for tasks"""

import numpy as np
from dm_control.rl.control import Task
from dm_env import specs
from collections import namedtuple
from environments.dm_control.utils import param

class Step(Task):

    def __init__(self, **kwargs):
        super().__init__()
        # Define default parameters
        par = namedtuple('par', 'name value description')
        self.default_par_list = [
            par('task_name','Step','Name of task'),
            par('maxinflow',5.,'max control inflow'),
            par('h_goal1',1.,'[m] target height 1st time interval'),
            par('h_goal2',0.8,'[m] target height 2nd time interval'),
            par('t_step',float('inf'),'[s] switching instant 1st->2nd target'),
            par('debug',False,'if True store data during episode')
        ]

        # Generate parameter dictionary
        self.par_dict = {x.name: x.value for x in self.default_par_list}
        
        # Overload parameters from inputs
        param.overload_par_dict(self.par_dict, **kwargs)

    def initialize_episode(self, physics):
        # Eventually pass some parameters
        self.datadict = []
        physics.__init__()

    def get_reference(self, physics) -> float:
        return self.par_dict['h_goal1'] if physics.time() < self.par_dict['t_step'] else self.par_dict['h_goal2']

    def get_observation(self, physics):
        # Let the actor observe the reference and the state
        return np.concatenate(([self.get_reference(physics)], physics._state))

    def get_reward(self, physics):
        sigma = 0.1
        mu = self.get_reference(physics)
        # Gaussian like rewards on target
        return np.exp(-np.power(physics._state[0] - mu, 2.) / (2 * np.power(sigma, 2.)))
    
    def before_step(self, action, physics):
        physics.set_control(action)
        # Store data dictionary for debugging
        if self.par_dict['debug']: extend_debug_datadict(self, physics, action)

    def observation_spec(self, physics):
        """Returns the observation spec."""
        return specs.Array(
            shape=(2,),
            dtype=np.float32,
            name='observation')

    def action_spec(self, physics):
        """Returns the action spec."""
        return specs.BoundedArray(
            shape=(1,),
            dtype=np.float32,
            minimum=0.,
            maximum=self.par_dict['maxinflow'],
            name='action')

    def get_par_dict(self):
        return self.par_dict

    def write_config_file(self, path, filename):
        param.write_config_file(self.default_par_list, self.par_dict, path,filename)

    def set_par_from_config_file(self, path):
        param.set_par_from_config_file(self.par_dict, path)


# ------------- Utils  ------------ 
def extend_debug_datadict(task, physics, action):
    # Data are stored before taking the step
    task.datadict.append(
        {
            'state': physics._state,
            'action': action,
            'time': physics.time(),
            'observation': task.get_observation(physics),
            'reward': task.get_reward(physics), 
            }
         )

def pack_datadict(datadict):
    # Pack data dictionary into numpy array
    # Assume all dictionaries have the same length
    return {key: np.asarray([ts[key] for ts in datadict])
            for key in datadict[0] }


