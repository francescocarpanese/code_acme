from environments.MovingCoil0D.Tasks.Task import Task

class Dummy(Task):
    def get_reward(self, _ ):
        return 0