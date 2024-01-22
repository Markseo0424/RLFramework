import abc
from abc import ABC


class Agent(ABC):
    def __init__(self):
        self._state = None

    def act(self):
        """
        Get an action about current state.
        It must be followed by set_state().
        :return: Current action about current state.
        """
        return self.policy(self._state)

    def reset(self):
        self._state = None
        self.reset_params()

    def set_state(self, state):
        """
        Setter of current state.
        :param state: Current state of Environment.
        """
        self._state = state

    @abc.abstractmethod
    def policy(self, state):
        """
        :param state: Current state of environment.
        :return: Action about current state.
        Returns action based on current state.
        Corresponds to pi(s,a).
        Action must be index of an action when using Deep RL Trainers.
        """
        pass

    @abc.abstractmethod
    def reset_params(self):
        pass
