from environments.MovingCoil0D.Tasks.Task import Task
import numpy as np

class HoldTarget(Task):
    def __init__(self,
                 x_goal: float = 0.) -> None:
        self._x_goal = x_goal

    def get_reward(self, env):
        term_rew = 0.  # reward for termination state
        sigma = 0.05
        mu = self._x_goal
        # Gauss like rewards on target
        gauss = np.exp(-np.power(env._state[0] - mu, 2.) / (2 * np.power(sigma, 2.)))
        return term_rew if env.check_truncation() else gauss
