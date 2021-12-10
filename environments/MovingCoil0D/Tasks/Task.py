from abc import ABC, abstractclassmethod

class Task(ABC):
    @abstractclassmethod
    def get_reward(self, env) -> float:
        pass