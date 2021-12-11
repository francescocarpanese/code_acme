"""Moving coil environment as dm_env format """

from __future__ import annotations

import dm_env
from dm_env import specs
import numpy as np
from environments.MovingCoil0D import Tasks

class Moving_Coil(dm_env.Environment):
    """Environment built on the `dm_env.Environment` class.
    The agent must change the current in 2 fixed coils, located at x1 and x2,
    on the sides of a moving coil, in order to
    bring the moving coil to the desired x target.

    Physic law for the moving coil:
    ds/dt = F(s), with state s = [x, dx/dt], F(s) = []

    Time discretization 1st order Euler explicit scheme s_t+1 = dt_sim*F(s_t) + s_t

    Observation are assumed to be istantaneous with action
    Actions are kept constant during simulation loop

    """
    def __init__(self,
                 x1:  float = -1.,  # x[m] location fixed coil 1
                 x2: float = 1.,  # x[m] location fixed coil 2
                 Ip: float = 1.,  # Ip[A] moving coil
                 m:  float = 1.,  # m[Kg] moing coil
                 dt_sim: float = 1e-1,  # [s] Discretization time interval for sim
                 dt_ctr: float = 1e-1,   # [s] ctr time interval
                 tend: float = float(np.inf), # Tot simulation interval 
                 x_goal: float = 0.,  # x target for rewards
                 discount: float = 0.9995,  # Discount factor
                 init_state: np.ndarray = np.array([0., 0.], dtype=np.float32),  # Initial state [x, dxdt]
                 task: Tasks.Task = Tasks.Dummy() ): 
        
        # Fetch parameters
        self._state = init_state
        self._init_state = init_state
        self._reset_next_step = False
        self._x1 = x1
        self._x2 = x2
        self._Ip = Ip
        self._m = m
        self._dt_sim = dt_sim
        self._x_goal = x_goal
        self._discount = discount
        # TODO need to check that dt_ctr is multiple of dt_sim
        self._n_sub_step = int(dt_ctr/dt_sim) # This operation is equivalent to floor
        self._tend = tend
        self._t = 0.
        self._n_phys_steps = 0
        self._task = task
        

    def reset(self) -> dm_env.TimeStep[float, float, np.ndarray]:
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        self._state = self._init_state
        self._t = 0.
        self._n_phys_steps = 0 # Number of step for physics simulator
        return dm_env.restart(self._observation())

    def step(self, action: float) -> dm_env.TimeStep[float, float, np.ndarray]:
        """Updates the environment according to the action."""
        # Reset env
        if self._reset_next_step:
            return self.reset()
                                
        for _ in range(self._n_sub_step):
            # Advance env in time
            self._state = self._dt_sim*self.F(action) + self._state
            # Update sim time 
            self._t += self._dt_sim
            self._n_phys_steps += 1

            # Stick coil to the boundary if reached, force to infinity
            if self._state[0] <= self._x1:
                self._state[0] = self._x1
                self._state[1] = 0.
            elif self._state[0] >= self._x2:
                self._state[0] = self._x2
                self._state[1] = 0.

            # Compute reward from task
            reward = self._task.get_reward(self)
            
            # Check for termination criteria and return
            if self.check_truncation():
                return dm_env.truncation(reward=reward,
                                      observation=self._observation(), discount=self._discount)

        # Check for termination criteria and return
        if self._t > self._tend:
            return dm_env.termination(reward=reward,
                                      observation=self._observation())
        else:
            return dm_env.transition(reward=reward,
                                     discount=self._discount,
                                     observation=self._observation())

    def observation_spec(self) -> specs.Array:
        """Returns the observation spec."""
        return specs.Array(
            shape=(2,),
            dtype=np.float32,
            name='observation')

    def action_spec(self) -> specs.BoundedArray:
        """Returns the action spec."""
        return specs.BoundedArray(
            shape=(2,),
            dtype=np.float32,
            minimum=-10,
            maximum=10,
            name='action')

    def _observation(self) -> np.ndarray:
        # For simplicity observations = system state
        return self._state.copy()

    # Attraction force definition between coils
    def Fa(self, d, I_coil):
        distance = d - self._state[0]
        return self._Ip*I_coil/distance if distance != 0 else 0.

    # NL operator, RHS of ODE
    def F(self, action):
        return np.array([self._state[1], self.a(action[0], action[1])])

    # Acceleration definition d _state[1] / dt_sim
    def a(self, I1, I2):
        return 1/self._m*(self.Fa(self._x1, I1) + self.Fa(self._x2, I2))

    def check_truncation(self):
        # Terminate if one coil reaches boundary or physical states not finite
        return  self._state[0] == self._x1 or \
                self._state[0] == self._x2 or \
                not all(np.isfinite(self._state))

