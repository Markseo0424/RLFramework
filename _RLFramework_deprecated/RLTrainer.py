import abc
from abc import ABC
from _RLFramework_deprecated.Environment import Environment
from _RLFramework_deprecated.Agent import Agent


class RLTrainer(ABC):
    def __init__(self, environment: Environment, agent: Agent):
        self.environment = environment
        self.agent = agent
        self.memory_state = []
        self.memory_action = []
        self.memory_reward = []
        self.timestep = -1

    def step(self):
        self.timestep += 1

        self.environment.step()
        self.agent.set_state(self.environment.get_state())
        action = self.agent.act()
        self.environment.act(action)

        self.memory_state.append(self.environment.get_state())
        self.memory_reward.append(self.environment.get_reward())
        self.memory_action.append(action)
        self.memory()

        if self.environment.timestep >= 1:
            if self.check_train():
                self.train(self.memory_state[-2], self.memory_action[-2], self.memory_reward[-1], self.memory_state[-1])
        if self.check_reset():
            self.reset()

    def reset(self):
        self.environment.reset()
        self.agent.reset()
        self.memory_state = []
        self.memory_action = []
        self.memory_reward = []
        self.reset_params()

    @abc.abstractmethod
    def check_reset(self):
        """
        :return: Bool value for whether to reset an Environment.
        """
        pass

    @abc.abstractmethod
    def check_train(self):
        """
        :return: Bool value for whether to train.
        """
        pass

    @abc.abstractmethod
    def memory(self):
        """
        Abstract function about memorizing factors.
        It can be overridden if need to memorize elements after each step.
        """
        pass

    @abc.abstractmethod
    def train(self, state, action, reward, next_state):
        """
        :param state: Current state of an environment.
        :param action: Current action of an agent.
        :param reward: Current reward of current state-action set.
        :param next_state: Next state of an environment.
        Abstract function about Training.
        """
        pass

    @abc.abstractmethod
    def reset_params(self):
        pass
