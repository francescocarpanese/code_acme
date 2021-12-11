"""Moving coil environment as dm_env format """

from __future__ import annotations

import dm_env
from dm_env import specs
import numpy as np
from environments.tank import Tasks

class env(dm_env.Environment):
    """Environment built on the `dm_env.Environment` class.
    """
    def __init__(self,
                 alpha: float = 1,
                 dt_sim: float = 1e-1,  # [s] Discretization time interval for sim
                 dt_ctr: float = 1e-1,   # [s] ctr time interval
                 tend: float = float(np.inf), # Tot simulation interval 
                 discount: float = 0.9995,  # Discount factor
                 reference: float = 0., 
                 hmax = 5,
                 maxinflow = 10., # max sink is alpha*sqrt(h_max)
                 init_state: np.ndarray = np.array([1.], dtype=np.float32),  # Initial state [x, dxdt]
                 task: Tasks.Task = Tasks.Dummy() ): 
        
        # Fetch parameters
        self._state = init_state
        self._init_state = init_state
        self._reset_next_step = False
        self._alpha = alpha
        self._dt_sim = dt_sim
        self._reference = reference
        self._discount = discount
        self._hmax = hmax
        # TODO need to check that dt_ctr is multiple of dt_sim
        self._n_sub_step = int(dt_ctr/dt_sim) # This operation is equivalent to floor
        self._tend = tend
        self._t = 0.
        self._n_phys_steps = 0
        self._task = task
        self._maxinflow = maxinflow

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
            self._state = self._dt_sim*self._F(action) + self._state

            # Update sim time 
            self._t += self._dt_sim
            self._n_phys_steps += 1

            # Wather out of tank
            if self._state[0] >= self._hmax:
                self._state[0] = self._hmax
            if self._state[0] <= 0.:
                self._state[0] = 0.

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

    def _F(self, action):
        return -self._alpha*np.sqrt( self._state ) + action

    def observation_spec(self) -> specs.Array:
        """Returns the observation spec."""
        return specs.Array(
            shape=(2,),
            dtype=np.float32,
            name='observation')

    def action_spec(self) -> specs.BoundedArray:
        """Returns the action spec."""
        return specs.BoundedArray(
            shape=(1,),
            dtype=np.float32,
            minimum=0.,
            maximum=self._maxinflow,
            name='action')

    def _observation(self) -> np.ndarray:
        # For simplicity let the agent observe directly the state of the system
        return np.concatenate(([self._task.get_reference(self)], self._state))

    def check_truncation(self):
        # Terminate if one coil reaches boundary or physical states not finite
        return  self._state[0] >= self._hmax or \
                not all(np.isfinite(self._state))

