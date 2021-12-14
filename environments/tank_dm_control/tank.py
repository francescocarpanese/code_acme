"""Moving coil environment as dm_env format """

from __future__ import annotations

import dm_env
from dm_env import specs
import numpy as np
from environments.tank import Tasks
from dm_control.rl.control import Physics
from dm_control.rl.control import PhysicsError

class physics(Physics):
    """Environment built on the `dm_env.Environment` class."""
    def __init__(self,
                 alpha: float = 1,
                 dt_sim: float = 0.5e-1,  # [s] Discretization time interval for sim
                 hmax = 5,
                 init_state: np.ndarray = np.array([1.], dtype=np.float32)  # Initial state [x, dxdt]
                 ): 
        
        # Fetch parameters
        self._state = init_state
        self._init_state = init_state
        self._alpha = alpha
        self._dt_sim = dt_sim
        self._hmax = hmax
        self._t = 0.
        self._action = np.asarray([0.])
        
    def reset(self):
        """Reset Physical values"""
        self._state = self._init_state
        self._time = 0.
        self._action =  np.asarray([0.])
    
    def after_reset(self):
        pass

    def step(self, n_sub_steps = 1) -> dm_env.TimeStep[float, float, np.ndarray]:
        """Updates the environment according to the action."""
    
        # Euler explicit time step
        self._state = self._dt_sim*self._F() + self._state

        # Update sim time 
        self._time += self._dt_sim

        # Keep h min at 0
        if self._state[0] <= 0.: self._state[0] = 0.
            
    def _F(self):
        """ Physical RHS for ODE d state / dt = F(state, action) """
        return -self._alpha*np.sqrt(self._state) + self._action

    def time(self):
        """Total elapsed simulation time"""
        return self._time

    def timestep(self):
        """dt simulation step"""
        return self._dt_sim

    def check_divergence(self):
        """ Terminate if one coil reaches boundary or physical states not finite """
        if  self._state[0] >= self._hmax:
            raise PhysicsError(f'h > max value = {self._hmax} [m]') 

        if not all(np.isfinite(self._state)):
            raise PhysicsError('System state not finite')

    def set_control(self, action):
        self._action = action

