import abc
from abc import ABC


class Environment(ABC):
    def __init__(self):
        self.timestep = -1
        self._state = None
        self._action = None
        self._reward = None

    def reset(self):
        self.timestep = -1
        self._state = None
        self._action = None
        self._reward = None
        self.reset_params()

    def act(self, action):
        """
        Set an action of Agent to Environment.
        :param action: Current action of an agent.
        """
        self._action = action

    def step(self):
        """
        Steps environment.
        Gets next state and current reward.
        """
        self.timestep += 1
        old_state = self._state
        self._state = self.update(self._state, self._action)
        self._reward = self.reward(old_state, self._action, self._state)

    # getters
    def get_state(self):
        """
        Getter of current state.
        """
        return self._state

    def get_reward(self):
        """
        Getter of previous reward.
        """
        return self._reward

    @abc.abstractmethod
    def update(self, state, action):
        """
        :param state: Current state of environment.
        :param action: Current action of agent.
        :return: Next state of environment. None if it is termination state.
        Update parameters, return next state.
        Corresponds to P(s,a).
        States must be unbatched numpy array when using Deep RL Trainers.
        """
        pass

    @abc.abstractmethod
    def reward(self, state, action, next_state):
        """
        :param state: Previous state of environment.
        :param action: Previous action of environment.
        :param next_state: Current action of environment.
        :return: Reward of previous state-action set.
        Gives reward based on previous state and action.
        Corresponds to R(s,a,s').
        """
        pass

    @abc.abstractmethod
    def reset_params(self):
        pass
