import numpy as np
from .trajectory_element import TrajectoryElement


class Trajectory(object):
    def __init__(self):
        self.__memory = []

    def __len__(self):
        return len(self.__memory)

    def reset(self):
        self.__memory = []

    def append(self, **kwargs):
        element = TrajectoryElement(**kwargs)

        if len(self):
            last = self.__memory[-1]
            last.next = element

        self.__memory.append(element)

    def get_elements(self):
        return self.__memory

    def recent(self):
        return self.__memory[-1]
