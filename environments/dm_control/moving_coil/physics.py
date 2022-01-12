"""Moving coil environment as dm_env format """

from __future__ import annotations
from collections import namedtuple
import numpy as np

import dm_env
from dm_control.rl import control

from environments.dm_control.utils import param

class Physics(control.Physics):
    """Environment built on the `dm_control.Environment` class."""

    def __init__(self, **kwargs):

        # Define default parameters
        par = namedtuple('par', 'name value description')
        self.default_par_list = [
            par('phys_name','MovingCoil','Name of physics module'),
            par('x1',-1.,"x[m] location fixed coil 1"),
            par('x2',1.,"x[m] location fixed coil 2"),
            par('Ip',1.,"Ip[A] moving coil"),
            par('m',1.,"m[Kg] moving coil"),
            par('dt_sim',1e-1,"[s] Discretization time interval for sim "),
            par('init_state',[0.,0.], "Initial state [x, dxdt]"),
        ]

        # Generate parameter dictionary
        self.par_dict = {x.name: x.value for x in self.default_par_list}

        # Overload parameters from inputs
        param.overload_par_dict(self.par_dict, **kwargs)

        # Reset physics
        self.reset()

    def reset(self):
        """Reset Physical values"""
        self._state = self.par_dict['init_state']
        self._time = 0.
        self._action =  np.asarray([0., 0.])

    def after_reset(self):
        pass

    def step(self) -> dm_env.TimeStep[float, float, np.ndarray]:
        """Updates the environment according to the action."""
                         
        # Advance env in time
        self._state = self.par_dict['dt_sim']*self._F() + self._state

        # Update sim time 
        self._time += self.par_dict['dt_sim']

        # Stick coil to the boundary if reached, force to infinity
        if self._state[0] <= self.par_dict['x1']:
            self._state[0] = self.par_dict['x1']
            self._state[1] = 0.
        elif self._state[0] >= self.par_dict['x2']:
            self._state[0] = self.par_dict['x2']
            self._state[1] = 0.


    # Attraction force definition between coils
    def _Fa(self, d, I_coil):
        distance = d - self._state[0]
        return self.par_dict['Ip']*I_coil/distance if distance != 0 else 0.

    def _F(self):
        """ Physical RHS for ODE d state / dt = F(state, action) """
        return np.array([self._state[1], self._a(self._action[0], self._action[1])])

    # Acceleration definition d _state[1] / dt_sim
    def _a(self, I1, I2):
        return 1/self.par_dict['m']*(self._Fa(self.par_dict['x1'], I1) + self._Fa(self.par_dict['x2'], I2))


    def time(self):
        """Total elapsed simulation time"""
        return self._time

    def timestep(self):
        """return dt simulation step"""
        return self.par_dict['dt_sim']


    def check_truncation(self):
        """ Terminate if one coil reaches boundary or physical states not finite """
        return  self._state[0] == self.par_dict['x1'] or \
                self._state[0] == self.par_dict['x2'] or \
                not all(np.isfinite(self._state))

    def check_divergence(self):
        """ Terminate if one coil reaches boundary or physical states not finite """

        if  self._state[0] <= self.par_dict['x1']:
            raise control.PhysicsError('Moving coil reached fixed coil 1')

        if  self._state[0] >= self.par_dict['x2']:
            raise control.PhysicsError('Moving coil reached fixed coil 2')
        
        if not all(np.isfinite(self._state)):
            raise PhysicsError('System state not finite')

    def set_control(self, action):
        self._action = action

    def get_par_dict(self):
        return self.par_dict

    def write_config_file(self, path, filename):
        param.write_config_file(self.default_par_list, self.par_dict, path,filename)

    def set_par_from_config_file(self, path):
        param.set_par_from_config_file(self.par_dict, path)