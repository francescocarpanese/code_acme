""" Container for tasks"""
from abc import ABC, abstractclassmethod
import numpy as np

class Task(ABC):
    @abstractclassmethod
    def get_reward(self, _ ) -> float:
        pass

    @abstractclassmethod
    def get_reference(self, _ ) -> float:
        pass

class Dummy(Task):
    def get_reward(self, _ ):
        return 0.
    def get_reference(self, _) -> float:
        return 0.

class HoldTarget(Task):
    def __init__(self,
                 h_goal: float = 2.
                 ) -> None:
        self._h_goal = h_goal

    def get_reference(self, env) -> float:
        return self._h_goal

    def get_reward(self, env):
        term_rew = 0.  # reward for termination state
        sigma = 0.1
        mu = self.get_reference(env)
        # Gauss like rewards on target
        gauss = np.exp(-np.power(env._state[0] - mu, 2.) / (2 * np.power(sigma, 2.)))
        return term_rew if env.check_truncation() else gauss
    

class Step(Task):
    def __init__(self,
                 h_goal1= 1.,
                 h_goal2= 0.5,
                 t_step= 1.,
                  ) -> None:
        # TODO add description
        self._h_goal1 = h_goal1
        self._h_goal2 = h_goal2
        self._t_step = t_step

    def get_reference(self, env) -> float:
        return self._h_goal1 if  env._t < self._t_step else self._h_goal2

        """ Check if a small ramp is easier to control
        # Step with linear ramp
        if env._t < self._t_step:
            return self._x_goal1
        elif env._t >= self._t_step and env._t < self._t_step + self._dt_ramp:
            return (self._x_goal2 - self._x_goal1)/self._dt_ramp*(env._t - self._t_step) + self._x_goal1
        else:
            return self._x_goal2
        """
        

    def get_reward(self, env) -> float:
        term_rew = 0.  # reward for termination state
        sigma = 0.05   
        mu = self.get_reference(env)

        # Gauss like reward on target
        gauss = np.exp(-np.power(env._state[0] - mu, 2.) / (2 * np.power(sigma, 2.)))
        return term_rew if env.check_truncation() else gauss

