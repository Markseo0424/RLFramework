import numpy as np
from .trajectory import Trajectory
from .trajectory_element import TrajectoryElement
from .sample import Sample


class ReplayMemory(object):
    def __init__(self, batch_size=32, max_len=1000000):
        self._memory = []
        self.batch_size = batch_size
        self.max_len = max_len

    def __len__(self):
        return len(self._memory)

    def append(self, **kwargs):
        element = TrajectoryElement(**kwargs)
        self._memory.append(element)

        # cut if element is too many
        if len(self) > self.max_len:
            self._memory = self._memory[-self.max_len]

    def append_element(self, element: TrajectoryElement):
        self._memory.append(element)

        # cut if element is too many
        if len(self) > self.max_len:
            self._memory = self._memory[-self.max_len]

    def append_traj(self, traj: Trajectory):
        self._memory = self._memory + traj.get_elements()

        # cut if element is too many
        if len(self) > self.max_len:
            self._memory = self._memory[-self.max_len]

    def sample(self):
        sample = np.random.randint(len(self), size=(self.batch_size,))
        raw_elements = []

        for i in range(self.batch_size):
            element = self._memory[sample[i]]
            raw_elements.append(element)

        return Sample(raw_elements)
