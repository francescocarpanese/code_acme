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

"""water tank physics environment as dm_control.Environment

Physical equations:
The environement simulates the dynamic of the water level in tank,
with an outflow nozzle and an inflow from the top.

The dynamics of the water level h is described by the following ode,
d h /dt = - alpha sqrt(h) + w_in

where:
h = water level
alpha = nozzle coeff
w_in = inflow from the top (action/actuator)
"""

from __future__ import annotations
from collections import namedtuple
import numpy as np
from dm_control.rl import control
from environments.dm_control.utils import param

# Naming for standard physical values doesn't conform to pylint.
# pylint: disable=invalid-name


class Physics(control.Physics):
    """Environment built on the dm_control.Environment class."""
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

    def step(self, n_sub_steps = 1):
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
        """ Physical terminations:
         - water reached maximum level
         - physical states not finite
         """
        if  self._state[0] >= self.par_dict['hmax']:
            raise control.PhysicsError(
                f'h > max value = {self.par_dict["hmax"]} [m]'
            )

        if not all(np.isfinite(self._state)):
            raise control.PhysicsError('System state not finite')

    def set_control(self, action): # pylint: disable=arguments-renamed
        self._action = action # pylint: disable=attribute-defined-outside-init

    def get_par_dict(self):
        """Return dictionary with parameters"""
        return self.par_dict

    def get_state(self):
        """Return physical states"""
        return self._state

    def write_config_file(self, path, filename):
        """Write toml file with parameters"""
        param.write_config_file(
            self.default_par_list,
            self.par_dict,
            path,
            filename,
        )

    def set_par_from_config_file(self, path):
        """Read parameters from file and set them"""
        param.set_par_from_config_file(self.par_dict, path)
