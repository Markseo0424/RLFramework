import abc
import torch
import torch.nn as nn
import numpy as np
import copy

from RLFramework.RLTrainer import RLTrainer
from RLFramework.Network import Network
from RLFramework.Environment import Environment
from RLFramework.Agent import Agent
from ReplayBuffer import ReplayBuffer


class DDPGTrainer(RLTrainer):
    def __init__(self, policy_net: Network, q_net: Network, environment: Environment, agent: Agent,
                 batch_size=128, start_train_step=5000, buffer_len=1000000, slot_weights: dict = None,
                 gamma=0.99, tau=0.01, verbose="none"):
        super().__init__(environment=environment, agent=agent)

        self.policy_net = policy_net
        self.target_policy_net = copy.deepcopy(policy_net)
        self.q_net = q_net
        self.target_q_net = copy.deepcopy(q_net)

        self.batch_size = batch_size

        self.start_train_step = start_train_step
        self.step = 0

        self.gamma = gamma
        self.tau = tau

        self.replay_buffer = ReplayBuffer(buffer_len=buffer_len, slot_weights=slot_weights)

        self.verbose = verbose.split(",")

    def check_train(self):
        return self.step >= self.start_train_step

    def train(self, state, action, reward, next_state):
        memory = self.replay_buffer.sample(self.batch_size)

        actor_loss = 0
        critic_loss = 0

        for _state, _action, _reward, _next_state in memory:
            if _next_state is None:
                next_Q = 0
            else:
                pred_next_action = self.target_policy_net.predict(_next_state).cpu().detach().numpy()
                next_Q = self.target_q_net.predict(np.concatenate([_state, pred_next_action]))

            y = _reward + self.gamma * next_Q
            pred_Q = self.q_net(torch.cat([torch.FloatTensor(_state).to(self.q_net.device), _action]))

            critic_loss = critic_loss + (pred_Q - y) ** 2

            pred_action = self.policy_net(torch.FloatTensor(_state).to(self.policy_net.device))
            pred_Q = self.q_net(torch.cat([torch.FloatTensor(_state).to(self.q_net.device), pred_action]))

            actor_loss = actor_loss - pred_Q

        policy_optim = self.policy_net.optimizer
        q_optim = self.q_net.optimizer

        critic_loss = critic_loss / len(memory)
        actor_loss = actor_loss / len(memory)

        q_optim.zero_grad()
        critic_loss.backward()
        q_optim.step()

        policy_optim.zero_grad()
        actor_loss.backward()
        policy_optim.step()

        return actor_loss, critic_loss

    def memory(self):
        """
        Saves data of (state, action, reward, next state) to the replay buffer.
        Can be overridden when need to memorize other values.
        """

        self.step += 1

        if self.environment.timestep >= 1:
            state, action, reward, next_state = self.memory_state[-2], self.memory_action[-2], self.memory_reward[
                -1], self.memory_state[-1]
            self.replay_buffer.append(state, action, reward, next_state,
                                      slot=self.choose_slot(state, action, reward, next_state))

    def choose_slot(self, state, action, reward, next_state):
        """
        :param state: Current state of environment.
        :param action: Current action of agent.
        :param reward: Reward of Current state-action set.
        :param next_state: Next state of environment.
        :return: Slot name where this data would be inserted.
        Check state, action and reward, and returns replay buffer slot where the data should be inserted.
        """
        return "default"

    @abc.abstractmethod
    def check_reset(self):
        pass

    @abc.abstractmethod
    def reset_params(self):
        pass
