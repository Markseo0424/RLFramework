import abc
import threading
import copy

from _RLFramework_deprecated.RLTrainer import RLTrainer
from _RLFramework_deprecated.Network import Network
from _RLFramework_deprecated.Environment import Environment
from _RLFramework_deprecated.Agent import Agent
from _RLFramework_deprecated.A2C.A2CTrainer import A2CTrainer


class A3CTrainer(RLTrainer):
    def __init__(self, policy_net: Network, value_net: Network, environment: Environment, agent: Agent, num_heads=8,
                 gamma=1, grad_clip=None, a2c_trainer_class=A2CTrainer, **kwargs):
        """
        :param num_heads: Number of async heads during training.
        :param gamma: Discount factor of Reward
        :param grad_clip: Gradient Clipping norm magnitude.
        :param a2c_trainer_class: User-defined A2C trainer class.
        :param kwargs: Arguments for trainer class.
        """
        super().__init__(environment=environment, agent=agent)

        self.policy_net = policy_net
        self.value_net = value_net

        self.gamma = gamma
        self.grad_clip = grad_clip

        self.heads = []
        for i in range(num_heads):
            new_environment = copy.deepcopy(self.environment)
            new_agent = copy.copy(self.agent)
            self.heads.append(a2c_trainer_class(policy_net=self.policy_net, value_net=self.value_net,
                                                environment=new_environment, agent=new_agent, gamma=self.gamma,
                                                grad_clip=self.grad_clip, **kwargs))
            self.environment = new_environment
            self.agent = new_agent

        self.lock = threading.Lock()

    def async_step(self, trainer: A2CTrainer):
        trainer.timestep += 1

        trainer.environment.step()
        trainer.agent.set_state(trainer.environment.get_state())
        action = trainer.agent.act()
        trainer.environment.act(action)

        trainer.memory_state.append(trainer.environment.get_state())
        trainer.memory_reward.append(trainer.environment.get_reward())
        trainer.memory_action.append(action)
        trainer.memory()

        if trainer.environment.timestep >= 1:
            if trainer.check_train():
                with self.lock:
                    trainer.train(trainer.memory_state[-2], trainer.memory_action[-2], trainer.memory_reward[-1], trainer.memory_state[-1])
        if trainer.check_reset():
            trainer.reset()

    def step(self):
        threads = []
        for head in self.heads:
            threads.append(threading.Thread(
                target=self.async_step, daemon=True, args=[head]
            ))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        self.memory()
        if self.check_reset():
            self.reset()

    def train(self, state, action, reward, next_state):
        pass

    def check_train(self):
        pass

    @abc.abstractmethod
    def memory(self):
        """
        Abstract function about memorizing factors.
        It can be overridden if need to memorize elements after each step.
        """
        pass

    @abc.abstractmethod
    def check_reset(self):
        pass

    @abc.abstractmethod
    def reset_params(self):
        pass
