#MIT License

#Copyright (c) 2022 Francesco Carpanese

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

""" Container for tasks"""
from collections import namedtuple
import numpy as np
from dm_control.rl.control import Task
from dm_env import specs

from environments.dm_control.utils import param

class Step(Task):
    """ Step task:
    Keep constant value and step to different constant value at t_step
    """

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

        # Init variable to store results when debug == True
        self.datadict = []

    def initialize_episode(self, physics):
        # Eventually pass some parameters
        self.datadict = []
        physics.reset()

    def get_reference(self, physics):
        """Returns target reference"""
        if physics.time() < self.par_dict['t_step']:
            target = self.par_dict['h_goal1']
        else:
            target = self.par_dict['h_goal2']
        return target

    def get_observation(self, physics):
        # Let the actor observe the reference and the state
        return np.concatenate((
            [self.get_reference(physics)],
            physics.get_state()
            ))

    def get_reward(self, physics):
        sigma = 0.1
        mean = self.get_reference(physics)
        # Gaussian like rewards on target
        return np.exp(
            -np.power(physics.get_state()[0] - mean, 2.)/(2*np.power(sigma, 2.))
        )

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
        """Return dictionary with parameters"""
        return self.par_dict

    def write_config_file(self, path, filename):
        """Writes toml configuration file from parameters"""
        param.write_config_file(
            self.default_par_list,
            self.par_dict,
            path,
            filename
        )

    def set_par_from_config_file(self, path):
        """Read config file and set parameters"""
        param.set_par_from_config_file(self.par_dict, path)


# TODO(fc) move following utiles elsewhere
def extend_debug_datadict(task, physics, action):
    """Append episode data to datadictionary

    Data are stored before taking the step
    """
    task.datadict.append(
        {
            'state': physics.get_state(),
            'action': action,
            'time': physics.time(),
            'observation': task.get_observation(physics),
            'reward': task.get_reward(physics),
            }
         )

def pack_datadict(datadict):
    """
    Pack data dictionary into numpy array.
    Assume all value in dictionary have the same length time lenght.
    """
    return {key: np.asarray([ts[key] for ts in datadict])
            for key in datadict[0] }
