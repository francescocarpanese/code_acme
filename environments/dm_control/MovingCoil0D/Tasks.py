""" Container for tasks"""

import numpy as np
from dm_control.rl.control import Task
from dm_env import specs

class Step(Task):

    def __init__(self,
                 maxIp= 10.,
                 minIp= -10,
                 x_goal1= 0.,
                 x_goal2= 0.1,
                 t_step= float('inf'),
                 debug= False,
                 ) -> None:
        super().__init__()

        self._x_goal1 = x_goal1
        self._x_goal2 = x_goal2
        self._t_step = t_step
        self._maxIp = maxIp
        self._minIp = minIp
        self._debug = debug
        self.datadict = []    

    def initialize_episode(self, physics):
        # Eventually pass some parameters
        self.datadict = []
        physics.__init__()

    def get_reference(self, physics) -> float:
        return self._x_goal1 if physics.time() < self._t_step else self._x_goal2

    def get_observation(self, physics):
        # Let the actor observe the reference and the state
        return np.concatenate(([self.get_reference(physics)], physics._state))

    def get_reward(self, physics):
        sigma = 0.05
        mu = self.get_reference(physics)
        # Gaussian like rewards on target
        return np.exp(-np.power(physics._state[0] - mu, 2.) / (2 * np.power(sigma, 2.)))
    
    def before_step(self, action, physics):
        physics.set_control(action)
        # Store data dictionary for debugging
        if self._debug: extend_debug_datadict(self, physics, action)

    def observation_spec(self, physics):
        """Returns the observation spec."""
        return specs.Array(
            shape=(3,),
            dtype=np.float32,
            name='observation')

    def action_spec(self, physics):
        """Returns the action spec."""
        return specs.BoundedArray(
            shape=(2,),
            dtype=np.float32,
            minimum=self._minIp,
            maximum=self._maxIp,
            name='action')

    def get_par_dict(self):
        return {
            'type': 'Step',
            'x_goal1': self._x_goal1,
            'x_goal2': self._x_goal2,
            't_step': self._t_step,
            'maxIp': self._maxIp,
            'minIp': self._minIp,
            't_step': self._t_step
        }

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


