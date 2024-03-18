import numpy as np
from .trajectory import Trajectory
from .trajectory_element import TrajectoryElement
from .contiguous_sample import ContiguousSample
from .replay_memory import ReplayMemory


class VolatileMemory(ReplayMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, batch_size=0, **kwargs)

    def sample(self):
        res = self._memory
        self._memory = []

        return ContiguousSample(res)
