import abc
import torch
import torch.nn as nn
import numpy as np
import copy

from RLFramework.RLTrainer import RLTrainer
from RLFramework.Network import Network
from RLFramework.Environment import Environment
from RLFramework.Agent import Agent
from .ReplayBuffer import ReplayBuffer


class SACTrainer(RLTrainer):
    def __init__(self, policy_net: Network, value_net: Network, q_nets: tuple[Network], environment: Environment, agent: Agent,
                 batch_size=128, start_train_step=10000, train_freq=500, buffer_len=1000000, slot_weights: dict = None,
                 gamma=0.99, tau=0.001, verbose="none"):
        super().__init__(environment=environment, agent=agent)

        assert len(q_nets) == 2

        self.policy_net = policy_net
        self.q_net_1 = q_nets[0]
        self.q_net_2 = q_nets[1]
        self.value_net = value_net
        self.target_value_net = copy.deepcopy(value_net)

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

    def to_tensor(self, x):
        return torch.FloatTensor(x).to(self.value_net.device)

    def get_batches(self, memory):
        states = []
        actions = []
        log_probs = []
        target_qs = []

        for _state, _action, _reward, _next_state in memory:
            states.append(_state)
            actions.append(_action[0])
            log_probs.append(_action[1])
            if _next_state is None:
                target_qs.append(float(_reward))
            else:
                target_qs.append(float(_reward + self.gamma * self.target_value_net.predict(_next_state)))

        states = np.stack(states)
        actions = np.stack(actions)
        log_probs = np.stack(log_probs)
        target_qs = np.stack(target_qs)

        return states, actions, log_probs, target_qs

    def train(self, state, action, reward, next_state):
        assert getattr(self.policy_net, "evaluate_prob", None) is not None, "Define evaluate_prob(state_batch, action_batch) in policy net."

        memory = self.replay_buffer.sample(self.batch_size)
        state_batch, action_batch, log_prob_batch, target_q_batch = self.get_batches(memory)

        # Value net optimization
        Jv_1 = 0.5 * torch.mean(torch.pow(
            self.value_net(self.to_tensor(state_batch)) -
            self.q_net_1.predict(np.concatenate([state_batch, action_batch], axis=1)) +
            log_prob_batch
        , 2))

        Jv_2 = 0.5 * torch.mean(torch.pow(
            self.value_net(self.to_tensor(state_batch)) -
            self.q_net_2.predict(np.concatenate([state_batch, action_batch], axis=1)) +
            log_prob_batch
        , 2))

        Jv = torch.minimum(Jv_1, Jv_2)

        self.value_net.optimizer.zero_grad()
        Jv.backward()
        self.value_net.optimizer.step()

        # Q net optimization
        Jq = 0.5 * torch.mean(torch.pow(
            self.q_net_1(self.to_tensor(np.concatenate([state_batch, action_batch], axis=1))) -
            target_q_batch
        )) + 0.5 * torch.mean(torch.pow(
            self.q_net_2(self.to_tensor(np.concatenate([state_batch, action_batch], axis=1))) -
            target_q_batch
        ))

        self.q_net_1.optimizer.zero_grad()
        self.q_net_2.optimizer.zero_grad()
        Jq.backward()
        self.q_net_1.optimizer.step()
        self.q_net_2.optimizer.step()

        # Policy optimization

        Jpi_1 = torch.mean(
            torch.log(self.policy_net.evaluate_prob(state_batch, action_batch) + 1e-7) -
            self.q_net_1.predict(np.concatenate([state_batch, action_batch], axis=1))
        )

        Jpi_2 = torch.mean(
            torch.log(self.policy_net.evaluate_prob(state_batch, action_batch) + 1e-7) -
            self.q_net_1.predict(np.concatenate([state_batch, action_batch], axis=1))
        )

        Jpi = torch.minimum(Jpi_1, Jpi_2)

        self.policy_net.optimizer.zero_grad()
        Jpi.backward()
        self.policy_net.optimizer.step()

        # Soft Update
        self.soft_update(self.target_value_net, self.value_net)

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
