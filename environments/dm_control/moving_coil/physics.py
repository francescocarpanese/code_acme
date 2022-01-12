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

"""Physical environment implemented as dm_control.Environement

Physical equations:
The environment simulates a straight line 1m long moving coil,
carrying Ip current, moving in between two fixed in space straight
line coils. The environment is in 1D.

The attraction force between the moving coil, with current Ip,
and the fixed coil with current I1 is given by
    F = mu_0 Ip * I1 * Ip/ (2 pi r)
where r is the cartesian distance between the two coils.

The coil dynamic follows the 2nd Newton's law
    m d^2 x / d t^2 = F(x, Ip, I1) +  F(x, Ip, I2)

where:
m = mass of the moving coil [Kg]
x = position of the moving coil [m]
Ip = current in the moving coil [A]
I1 = current in the fixed coil 1 [A] (action/actuator)
I2 = current in the fixed coil 2 [A] (action/actuator)
"""

from __future__ import annotations
from collections import namedtuple
import numpy as np
from dm_control.rl import control
from environments.dm_control.utils import param

# Naming for standard physical values doesn't conform to pylint.
# pylint: disable=invalid-name

class Physics(control.Physics):
    """Physics class based on the dm_control.Environment class."""

    def __init__(self, **kwargs):

        # Define default parameters
        par = namedtuple('par', 'name value description')
        self.default_par_list = [
            par('phys_name', 'MovingCoil', 'Name of physics module'),
            par('x1', -1., 'x[m] location fixed coil 1'),
            par('x2', 1., 'x[m] location fixed coil 2'),
            par('Ip', 1., 'Ip[A] moving coil'),
            par('m', 1., 'm[Kg] moving coil'),
            par('dt_sim', 1e-1, '[s] Discretization time interval for sim'),
            par('init_state', [0.,0.], 'Initial state [x, dxdt]'),
        ]

        # Generate parameter dictionary
        self.par_dict = {x.name: x.value for x in self.default_par_list}

        # Overload parameters from inputs
        param.overload_par_dict(self.par_dict, **kwargs)

        # Reset physics
        self.reset()

    def reset(self):
        """Resets Physical values"""
        self._state = self.par_dict['init_state']
        self._time = 0.
        self._action =  np.asarray([0., 0.])

    def after_reset(self):
        pass

    def step(self): # Substeps already handled by dm_env.Environment pylint: disable=arguments-differ
        """Step physics according to the action."""

        # Advance env in time
        self._state = self.par_dict['dt_sim']*self._F() + self._state # pylint: disable=attribute-defined-outside-init

        # Update sim time
        self._time += self.par_dict['dt_sim']

        # Stick coil to the boundary if reached, force to infinity
        if self._state[0] <= self.par_dict['x1']:
            self._state[0] = self.par_dict['x1']
            self._state[1] = 0.
        elif self._state[0] >= self.par_dict['x2']:
            self._state[0] = self.par_dict['x2']
            self._state[1] = 0.


    def _Fa(self, d, I_coil):
        """Returns attraction force between moving coil and fixed coil"""
        distance = d - self._state[0]
        return self.par_dict['Ip']*I_coil/distance if distance != 0 else 0.

    def _F(self):
        """Returns physical RHS for ODE d state / dt = F(state, action) """
        return np.array([self._state[1],
             self._a(self._action[0], self._action[1])])

    def _a(self, I1, I2):
        """Returns moving coil acceleration"""
        return 1/self.par_dict['m']*(
            self._Fa(self.par_dict['x1'], I1) +
            self._Fa(self.par_dict['x2'], I2)
            )

    def time(self):
        """Returns total elapsed simulation time"""
        return self._time

    def timestep(self):
        """Returns dt simulation step"""
        return self.par_dict['dt_sim']

    def check_divergence(self):
        """ Terminates episode if:
         - moving coil reaches domain boundary
         - physical states not finite
         """

        if  self._state[0] <= self.par_dict['x1']:
            raise control.PhysicsError('Moving coil reached fixed coil 1')

        if  self._state[0] >= self.par_dict['x2']:
            raise control.PhysicsError('Moving coil reached fixed coil 2')

        if not all(np.isfinite(self._state)):
            raise control.PhysicsError('System state not finite')

    def set_control(self, action): # pylint: disable=arguments-renamed
        self._action = action # pylint: disable=attribute-defined-outside-init

    def get_par_dict(self): # pylint: disable=missing-function-docstring
        return self.par_dict
    
    def get_state(self): # pylint: disable=missing-function-docstring
        return self._state

    def write_config_file(self, path, filename): # pylint: disable=missing-function-docstring
        param.write_config_file(
            self.default_par_list,
            self.par_dict,
            path,filename
            )

    def set_par_from_config_file(self, path): # pylint: disable=missing-function-docstring
        param.set_par_from_config_file(self.par_dict, path)
