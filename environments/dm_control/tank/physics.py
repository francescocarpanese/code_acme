"""Moving coil environment as dm_env format """

from __future__ import annotations
from collections import namedtuple
import dm_env
import numpy as np
from dm_control.rl import control

from environments.dm_control.utils import param

class Physics(control.Physics):
    """Environment built on the `dm_env.Environment` class."""
    def __init__(self, **kwargs):

        # Define default parameters
        par = namedtuple('par', 'name value description')
        self.default_par_list = [
            par('phys_name','tank','Name of physics module'),
            par('alpha',1.,'outflow coefficient'),
            par('dt_sim',0.5e-1,'# [s] Discretization time interval for sim'),
            par('hmax',5,'[m] max water height in tank'),
            par('init_state',[1.],'[m] initial water height')
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
        self._action =  np.asarray([0.])
    
    def after_reset(self):
        pass

    def step(self, n_sub_steps = 1) -> dm_env.TimeStep[float, float, np.ndarray]:
        """Updates the environment according to the action."""
    
        # Euler explicit time step
        self._state = self.par_dict['dt_sim']*self._F() + self._state

        # Update sim time 
        self._time += self.par_dict['dt_sim']

        # Keep h min at 0
        if self._state[0] <= 0.: self._state[0] = 0.
            
    def _F(self):
        """ Physical RHS for ODE d state / dt = F(state, action) """
        return -self.par_dict['alpha']*np.sqrt(self._state) + self._action

    def time(self):
        """Total elapsed simulation time"""
        return self._time

    def timestep(self):
        """dt simulation step"""
        return self.par_dict['dt_sim']

    def check_divergence(self):
        """ Terminate if one coil reaches boundary or physical states not finite """
        if  self._state[0] >= self.par_dict['hmax']:
            raise control.PhysicsError(f'h > max value = {self.par_dict["hmax"]} [m]') 

        if not all(np.isfinite(self._state)):
            raise control.PhysicsError('System state not finite')

    def set_control(self, action):
        self._action = action

    def get_par_dict(self):
        return self.par_dict

    def write_config_file(self, path, filename):
        param.write_config_file(self.default_par_list, self.par_dict, path,filename)

    def set_par_from_config_file(self, path):
        param.set_par_from_config_file(self.par_dict, path)