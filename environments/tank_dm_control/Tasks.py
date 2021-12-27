""" Container for tasks"""

import numpy as np
from dm_control.rl.control import Task
from dm_env import specs

class HoldTarget(Task):

    def __init__(self,
                 h_goal = 1.,
                 maxinflow = 5.,
                 debug = False,
                 ) -> None:
        super().__init__()
        # set parameters for the task
        self._h_goal = h_goal
        self._maxinflow = maxinflow
        self._debug = debug
        self.datadict = []


    def initialize_episode(self, physics):
        # Eventually pass some parameters
        self.datadict = []
        physics.__init__()

    def get_reference(self):
        return self._h_goal

    def get_observation(self, physics):
        # Let the actor observe the reference and the state
        return np.concatenate(([self.get_reference()], physics._state))

    def get_reward(self, physics):
        sigma = 0.5
        mu = self.get_reference()
        # Gaussian like rewards on target
        return np.exp(-np.power(physics._state[0] - mu, 2.) / (2 * np.power(sigma, 2.)))
    
    def before_step(self, action, physics):
        physics.set_control(action)
        # Store data dictionary for debugging
        if self._debug: extend_debug_datadict(self, physics, action)

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
            maximum=self._maxinflow,
            name='action')


# ------------- Utils  ------------ 
def extend_debug_datadict(task, physics, action):
    # Data are stored before taking the step
    task.datadict.append(
        {
            'state': physics._state,
            'action': action,
            'time': physics.time(),
            'observation': task.get_observation(physics),
            'reward': task.get_reward(physics)
            }
         )

def pack_datadict(datadict):
    # Pack data dictionary into numpy array
    # Assume all dictionaries have the same length
    return {key: np.asarray([ts[key] for ts in datadict])
            for key in datadict[0] }

