import abc
import torch
import torch.nn as nn
import numpy as np
import copy

from _RLFramework_deprecated.RLTrainer import RLTrainer
from _RLFramework_deprecated.Network import Network
from _RLFramework_deprecated.Environment import Environment
from _RLFramework_deprecated.Agent import Agent
from .ReplayBuffer import ReplayBuffer
import numpy as np


class DDPGTrainer(RLTrainer):
    def __init__(self, policy_net: Network, q_net: Network, environment: Environment, agent: Agent,
                 batch_size=128, start_train_step=10000, train_freq=500, buffer_len=1000000, slot_weights: dict = None,
                 gamma=0.99, tau=0.001, verbose="none"):
        super().__init__(environment=environment, agent=agent)

        self.policy_net = policy_net
        self.target_policy_net = copy.deepcopy(policy_net)
        self.q_net = q_net
        self.target_q_net = copy.deepcopy(q_net)

        self.batch_size = batch_size

        self.start_train_step = start_train_step
        self.train_freq = train_freq
        self.timestep = 0

        self.gamma = gamma
        self.tau = tau

        self.replay_buffer = ReplayBuffer(buffer_len=buffer_len, slot_weights=slot_weights)

        self.verbose = verbose.split(",")

    def check_train(self):
        return self.timestep >= self.start_train_step and self.timestep % self.train_freq == 0

    def soft_update(self, target, source):
        target_state_dict = target.state_dict()
        source_state_dict = source.state_dict()

        for param_name in source_state_dict:
            target_param = target_state_dict[param_name]
            source_param = source_state_dict[param_name]
            # print(target_param, source_param)
            target_param.copy_(
                target_param * (1.0 - self.tau) + source_param * self.tau
            )

    def get_batches(self, memory):
        states = []
        actions = []
        target_q = []

        for _state, _action, _reward, _next_state in memory:
            if _next_state is None:
                next_Q = 0
            else:
                pred_next_action = self.target_policy_net.predict(_next_state).cpu().detach().numpy()
                next_Q = self.target_q_net.predict(np.concatenate([_next_state, pred_next_action]))

            y = torch.tensor([float(_reward + self.gamma * next_Q)])

            target_q.append(y)
            states.append(_state)
            actions.append(_action)

        states = np.stack(states)
        actions = np.stack(actions)
        target_q = torch.stack(target_q).to(self.q_net.device)

        return states, actions, target_q

    def train(self, state, action, reward, next_state):
        critic_optim = self.q_net.optimizer
        actor_optim = self.policy_net.optimizer

        # print("========== train ============")
        memory = self.replay_buffer.sample(self.batch_size)
        state_batch, action_batch, target_q = self.get_batches(memory)

        # print("========== critic loss calc ============")

        critic_optim.zero_grad()

        pred_Q = self.q_net(
            torch.cat([torch.FloatTensor(state_batch), torch.FloatTensor(action_batch)], dim=1).to(self.q_net.device))

        critic_loss = nn.MSELoss()(pred_Q, target_q)

        critic_loss.backward()
        critic_optim.step()

        # print(
        #     f"  state : {state_batch[0]}, action: {action_batch[0]}, target_Q : {target_q[0].item()}, pred_Q : {pred_Q[0].item()}")

        # print("========== actor loss calc ============")

        actor_optim.zero_grad()

        pred_action = self.policy_net(torch.FloatTensor(state_batch).to(self.policy_net.device))
        pred_Q = self.q_net(torch.cat([torch.FloatTensor(state_batch).to(self.q_net.device), pred_action], dim=1))

        actor_loss = -torch.mean(pred_Q)

        actor_loss.backward()
        actor_optim.step()

        # print(
        #    f"  state : {state_batch[0]}, action: {action_batch[0]}, pred_action : {pred_action[0]}, pred_Q : {pred_Q[0].item()}")

        self.soft_update(self.target_q_net, self.q_net)
        self.soft_update(self.target_policy_net, self.policy_net)

        # print("======================")
        # print(actor_loss.item(), critic_loss.item())

        return actor_loss.item(), critic_loss.item()

    def memory(self):
        """
        Saves data of (state, action, reward, next state) to the replay buffer.
        Can be overridden when need to memorize other values.
        """

        self.timestep += 1

        if self.environment.timestep >= 1:
            state, action, reward, next_state = self.memory_state[-2], self.memory_action[-2], self.memory_reward[
                -1], self.memory_state[-1]
            self.replay_buffer.append(state, action, reward, next_state,
                                      slot=self.choose_slot(state, action, reward, next_state))

            # print(f"memory: \n  state : {state}\n  action : {action}\n  reward : {reward}\n  next_state : {next_state}")

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
