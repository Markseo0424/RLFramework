import abc
import torch
import torch.nn as nn
import numpy as np
import copy

from RLFramework.RLTrainer import RLTrainer
from RLFramework.Network import Network
from RLFramework.Environment import Environment
from RLFramework.Agent import Agent


class TRPOTrainer(RLTrainer):
    def __init__(self, policy_net: Network, value_net: Network, environment: Environment, agent: Agent,
                 delta=0.01, alpha=0.9, gamma=0.99, CG_iter=10, search_iter=50, train_freq=128, damping=0):
        super().__init__(environment=environment, agent=agent)

        self.policy_net = policy_net
        self.value_net = value_net

        self.delta = delta
        self.alpha = alpha
        self.gamma = gamma
        self.CG_iter = CG_iter
        self.search_iter = search_iter
        self.damping = damping

        self.update_vectors = None

        self.steps = 0
        self.batch_memory = []
        self.train_freq = train_freq

    def _hessian_vector_product(self, grad, param):
        return lambda v: torch.autograd.grad(grad.T @ v, param, retain_graph=True)[0].reshape((-1, 1)) + self.damping * v

    def _conjugate_gradient(self, A, b, break_bound=1e-7):
        x = torch.zeros(b.shape).to(self.policy_net.device)
        r = b
        v = torch.clone(r)

        for i in range(self.CG_iter):
            Av = A(v)
            alpha = (r.T @ r) / (v.T @ Av + 1e-9)
            prev_r = torch.clone(r)
            x = x + alpha * v
            r = r - alpha * Av

            if r.T @ r < break_bound:
                return x

            v = r + ((r.T @ r) / (prev_r.T @ prev_r + 1e-7)) * v

        return x

    def _get_update_vector(self, param, obj, kld):
        kld_grad = torch.autograd.grad(kld, param, retain_graph=True, create_graph=True)[0]
        kld_grad = kld_grad.reshape((-1, 1))

        grad = torch.autograd.grad(obj, param, retain_graph=True)[0]
        grad = grad.reshape((-1, 1))

        A = self._hessian_vector_product(kld_grad, param)
        s = self._conjugate_gradient(A, grad)

        beta = torch.sqrt(2 * self.delta / (s.T @ grad + 1e-7))

        update = beta * s
        update = update.reshape(param.shape)

        return update

    def _get_update_vectors(self, obj, kld):
        update_vectors = {}

        state_dict = self.policy_net.state_dict(keep_vars=True)

        for name in state_dict:
            param = state_dict[name]
            update = self._get_update_vector(param, obj, kld)
            update_vectors[name] = update

        self.update_vectors = update_vectors

    def _update_params(self, source_network: torch.nn.Module, alpha, i):
        if self.update_vectors is None:
            return

        network = copy.deepcopy(source_network)

        state_dict = network.state_dict()

        for name in state_dict:
            param = state_dict[name]
            new_param = param + (alpha ** i) * self.update_vectors[name]
            param.copy_(new_param)

        return network

    def check_train(self):
        return self.steps % self.train_freq == 0

    def train(self, state, action, reward, next_state):
        memory = self.batch_memory

        states = []
        actions = []
        old_policies = []
        q_values = []
        advantages = []

        for _state, _action, _qvalue, _next_state in memory:
            current_value = self.value_net.predict(_state)

            advantage = _qvalue - current_value

            states.append(_state)
            actions.append(_action)
            old_policies.append(self.policy_net.predict(_state))
            q_values.append(_qvalue)
            advantages.append(advantage)

        # update value network
        self.value_net.train_batch(np.stack(states), np.stack(q_values).reshape((-1, 1)), loss_function=nn.MSELoss())

        obj = 0

        policies = []

        for i in range(len(states)):
            policy = self.policy_net(torch.FloatTensor(states[i]).to(self.policy_net.device))
            policies.append(policy)

            obj = obj + policy[actions[i]] / (old_policies[i][actions[i]] + 1e-7) * advantages[i]

        obj = obj / len(states)
        old_policies = torch.stack(old_policies)
        policies = torch.stack(policies)

        kld = torch.nn.functional.kl_div(torch.log(policies + 1e-7), old_policies, reduction="batchmean")

        self._get_update_vectors(obj, kld)

        updated = False

        for i in range(self.search_iter):
            new_network = self._update_params(self.policy_net, self.alpha, i)
            new_policies = []
            new_obj = 0
            for j in range(len(states)):
                new_pred = new_network.predict(states[j])

                new_policies.append(new_pred)
                new_obj = new_obj + new_pred[actions[j]] / (old_policies[j][actions[j]] + 1e-7) * advantages[j]

            new_obj = new_obj / len(states)
            new_kld = torch.nn.functional.kl_div(torch.log(torch.stack(new_policies) + 1e-7), old_policies, reduction="batchmean")

            if new_obj > obj and new_kld < self.delta:
                self.policy_net.load_state_dict(new_network.state_dict())
                updated = True
                break

        if updated:
            self.batch_memory = []
            return updated, obj, new_obj, new_kld

        else:
            return updated, obj, obj, kld

    def memory(self):
        self.steps += 1
        if len(self.memory_state) >= 2 and self.memory_state[-2] is not None:
            if self.memory_state[-1] is not None:
                pred_next_value = self.value_net.predict(self.memory_state[-1]).item()
            else:
                pred_next_value = 0

            qvalue = self.gamma * pred_next_value + self.memory_reward[-1]

            self.batch_memory.append([self.memory_state[-2], self.memory_action[-2],
                                      qvalue, self.memory_state[-1]])

    @abc.abstractmethod
    def check_reset(self):
        pass

    @abc.abstractmethod
    def reset_params(self):
        pass

