from ..space import *


class Environment(object):
    def __init__(self, observation_space: Space, action_space: Space):
        self.observation_space = observation_space
        self.action_space = action_space

        self.timestep = 1
        self.__state = None
        self.__action = None
        self.__reward = None
        self.done = False

    def reset(self):
        self.timestep = 1
        self.__state = None
        self.__action = None
        self.__reward = None
        self.done = False
        self.reset_params()

    def init_state(self, state):
        self.__state = state

    def act(self, action):
        """
        Set an action of Agent to Environment.
        :param action: Current action of an agent.
        """
        self.__action = action

    def step(self):
        """
        Steps environment.
        Gets next state and current reward.
        """
        self.timestep += 1
        old_state = self.__state
        self.__state, self.done = self.update(self.__state, self.__action)
        self.__reward = self.reward(old_state, self.__action, self.__state)

    # getters
    def get_state(self):
        """
        Getter of current state.
        """
        return self.__state

    def get_reward(self):
        """
        Getter of previous reward.
        """
        return self.__reward

    def update(self, state, action):
        """
        Update parameters, return next state.
        Corresponds to P(s,a).
        States must be unbatched numpy array when using Deep RL Trainers.

        :param state: Current state of environment.
        :param action: Current action of agent.
        :return: Next state of environment, whether env is done. You can use None if it is termination state.
        """
        return state, False

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

    def reset_params(self):
        pass
