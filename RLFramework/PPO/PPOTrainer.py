import abc
import torch
import torch.nn as nn
import numpy as np
import copy

from RLFramework.RLTrainer import RLTrainer
from RLFramework.Network import Network
from RLFramework.Environment import Environment
from RLFramework.Agent import Agent


class PPOTrainer(RLTrainer):
    def __init__(self, policy_net: Network, value_net: Network, environment: Environment, agent: Agent,
                 epsilon=0.01, gamma=0.99, train_freq=128, entropy_weight=0.1, verbose="none"):
        super().__init__(environment=environment, agent=agent)

        self.policy_net = policy_net
        self.value_net = value_net

        self.epsilon = epsilon
        self.gamma = gamma
        self.entropy_weight = entropy_weight

        self.steps = 0
        self.batch_memory = []
        self.train_freq = train_freq

        self.verbose = verbose.split(",")

    def check_train(self):
        return self.steps % self.train_freq == 0

    def train(self, state, action, reward, next_state):
        memory = self.batch_memory
        optim = self.policy_net.optimizer

        if self.verbose == "all" or "policy" in self.verbose:
            print(f"start policy optimization. memory size : {len(memory)}")

        loss = 0

        for _state, _action, _reward, _next_state in memory:
            current_policy = self.policy_net(_state)
            old_policy = current_policy.detach()
            current_value = self.value_net.predict(_state)
            pred_next_value = 0 if _next_state is None else self.value_net.predict(_next_state)

            r = current_policy[_action] / old_policy[_action]
            advantage = self.gamma * pred_next_value - current_value

            clip_loss = torch.minimum(r * advantage, torch.clip(r, 1 - self.epsilon, 1 + self.epsilon) * advantage)
            entropy_loss = - current_policy[_action] * torch.log(current_policy[_action] + 1e-7)

            loss = loss - clip_loss - self.entropy_weight * entropy_loss

        loss = loss / len(memory)
        optim.zero_grad()
        loss.backward()
        optim.step()

    def memory(self):
        self.steps += 1
        if len(self.memory_state) >= 2 and self.memory_state[-2] is not None:
            if self.verbose == "all" or "value" in self.verbose:
                print("train value function")

            if self.memory_state[-1] is not None:
                pred_next_value = self.value_net.predict(self.memory_state[-1]).item()
            else:
                pred_next_value = 0

            qvalue = self.gamma * float(pred_next_value) + self.memory_reward[-1]

            if self.verbose == "all" or "value" in self.verbose:
                print(f"pred_next_value : {pred_next_value}\nreward : {self.memory_reward[-1]}\nqvalue : {qvalue}")

            # update value network
            for i in range(3):
                loss = self.value_net.train_batch(np.stack([self.memory_state[-2]]), np.stack([np.array([qvalue])]),
                                                  loss_function=nn.MSELoss())
                if self.verbose == "all" or "value" in self.verbose:
                    print(f"  > loss: {loss}")

            self.batch_memory.append([self.memory_state[-2], self.memory_action[-2],
                                      self.memory_reward[-1], self.memory_state[-1]])

    @abc.abstractmethod
    def check_reset(self):
        pass

    @abc.abstractmethod
    def reset_params(self):
        pass
