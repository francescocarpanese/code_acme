"""Moving coil environment as dm_env format """

from __future__ import annotations

import dm_env
import numpy as np
from dm_control.rl.control import Physics
from dm_control.rl.control import PhysicsError

class physics(Physics):
    """Environment built on the `dm_control.Environment` class."""

    def __init__(self,
                 x1: float = -1.,  # x[m] location fixed coil 1
                 x2: float = 1.,  # x[m] location fixed coil 2
                 Ip: float = 1.,  # Ip[A] moving coil
                 m:  float = 1.,  # m[Kg] moing coil
                 dt_sim: float = 1e-1,  # [s] Discretization time interval for sim              
                 init_state: float= [0.,0.],  # Initial state [x, dxdt]]
                 ): 

        # Fetch parameters
        self._init_state = np.asarray(init_state, dtype=np.float32)
        self._state = init_state
        
        self._x1 = x1
        self._x2 = x2
        self._Ip = Ip
        self._m = m
        self._dt_sim = dt_sim
        
        self._t = 0.
        self._action = np.asarray([0., 0.])
        
    def reset(self):
        """Reset Physical values"""
        self._state = self._init_state
        self._time = 0.
        self._action =  np.asarray([0., 0.])
    
    def after_reset(self):
        pass

    def step(self) -> dm_env.TimeStep[float, float, np.ndarray]:
        """Updates the environment according to the action."""
                         
        # Advance env in time
        self._state = self._dt_sim*self._F() + self._state

        # Update sim time 
        self._time += self._dt_sim

        # Stick coil to the boundary if reached, force to infinity
        if self._state[0] <= self._x1:
           self._state[0] = self._x1
           self._state[1] = 0.
        elif self._state[0] >= self._x2:
           self._state[0] = self._x2
           self._state[1] = 0.


    # Attraction force definition between coils
    def _Fa(self, d, I_coil):
        distance = d - self._state[0]
        return self._Ip*I_coil/distance if distance != 0 else 0.

    def _F(self):
        """ Physical RHS for ODE d state / dt = F(state, action) """
        return np.array([self._state[1], self._a(self._action[0], self._action[1])])

    # Acceleration definition d _state[1] / dt_sim
    def _a(self, I1, I2):
        return 1/self._m*(self._Fa(self._x1, I1) + self._Fa(self._x2, I2))


    def time(self):
        """Total elapsed simulation time"""
        return self._time

    def timestep(self):
        """dt simulation step"""
        return self._dt_sim


    def check_truncation(self):
        # Terminate if one coil reaches boundary or physical states not finite
        return  self._state[0] == self._x1 or \
                self._state[0] == self._x2 or \
                not all(np.isfinite(self._state))

    def check_divergence(self):
        """ Terminate if one coil reaches boundary or physical states not finite """

        if  self._state[0] <= self._x1:
            raise PhysicsError(f'Moving coil reached fixed coil 1') 

        if  self._state[0] >= self._x2:
            raise PhysicsError(f'Moving coil reached fixed coil 2') 
        
        if not all(np.isfinite(self._state)):
            raise PhysicsError('System state not finite')

    def set_control(self, action):
        self._action = action

    def get_par_dict(self):
        return {
            'dt_sim': self._dt_sim,
            'x1': self._x1,
            'x2': self._x2,
            'Ip': self._Ip,
            'm': self._m,
            'init_state': self._init_state.tolist()
            }
