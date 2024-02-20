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
                 delta=0.01, alpha=0.5, gamma=0.99, CG_iter=10, search_iter=10, train_freq=128, damping=0.01,
                 verbose="none"):
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
        self.grads = None

        self.steps = 0
        self.batch_memory = []
        self.train_freq = train_freq

        self.verbose = verbose.split(",")

    def _hessian_vector_product(self, grad, param):
        return lambda v: nn.utils.parameters_to_vector(
            torch.autograd.grad(grad.T @ v, param, retain_graph=True)).reshape((-1, 1)) + self.damping * v

    def _conjugate_gradient(self, A, b, break_bound=1e-4):
        if self.verbose == "all" or "cg" in self.verbose:
            print("conjugate gradient start : ")

        x = torch.zeros(b.shape).to(self.policy_net.device)
        r = b
        v = torch.clone(r)

        for i in range(self.CG_iter):
            Av = A(v)
            alpha = (r.T @ r) / (v.T @ Av + 1e-8)
            prev_r = torch.clone(r)
            x = x + alpha * v
            r = r - alpha * Av

            if self.verbose == "all" or "cg" in self.verbose:
                print(f"  iter {i} : r = {(r.T @ r).item()}")

            if r.T @ r < break_bound:
                return x

            v = r + ((r.T @ r) / (prev_r.T @ prev_r + 1e-7)) * v

        return x

    def _get_update_vectors(self, obj, kld):
        param = nn.ParameterList(self.policy_net.parameters())

        kld_grad = nn.utils.parameters_to_vector(torch.autograd.grad(kld, param, retain_graph=True, create_graph=True))
        kld_grad = kld_grad.reshape((-1, 1))

        grad = nn.utils.parameters_to_vector(torch.autograd.grad(obj, param, retain_graph=True))
        grad = grad.reshape((-1, 1))

        A = self._hessian_vector_product(kld_grad, param)
        s = self._conjugate_gradient(A, grad)

        beta = torch.sqrt(2 * self.delta / abs(s.T @ grad))

        update = beta * s

        self.grads = grad
        self.update_vectors = update

    def _update_params(self, source_network: torch.nn.Module, alpha, i):
        if self.update_vectors is None:
            return

        network = copy.deepcopy(source_network)

        param_vector = nn.utils.parameters_to_vector(network.parameters())

        new_param = param_vector + (alpha ** i) * self.update_vectors.reshape(-1)

        nn.utils.vector_to_parameters(new_param, network.parameters())

        return network

    def check_train(self):
        return self.steps % self.train_freq == 0

    def train(self, state, action, reward, next_state):
        memory = self.batch_memory

        states = []
        actions = []
        old_policies = []
        advantages = []

        if self.verbose == "all" or "policy" in self.verbose:
            print(f"start policy optimization. memory size : {len(memory)}")

        for _state, _action, _reward, _next_state in memory:
            current_value = self.value_net.predict(_state)

            if _next_state is not None:
                pred_next_value = self.value_net.predict(_next_state).item()
            else:
                pred_next_value = 0

            advantage = self.gamma * pred_next_value + _reward - current_value

            states.append(_state)
            actions.append(_action)
            old_policies.append(self.policy_net.predict(_state))
            advantages.append(advantage)

        if self.verbose == "all" or "policy" in self.verbose:
            print("memory append complete.")

        obj = 0

        policies = []

        for i in range(len(states)):
            policy = self.policy_net(torch.FloatTensor(states[i]).to(self.policy_net.device))
            policies.append(policy)

            obj = obj + policy[actions[i]] / (old_policies[i][actions[i]] + 1e-7) * advantages[i]

        if self.verbose == "all" or "policy" in self.verbose:
            print("appended policy.")

        obj = obj / len(states)
        old_policies = torch.stack(old_policies)
        policies = torch.stack(policies)

        kld = torch.nn.functional.kl_div(torch.log(policies + 1e-7), old_policies, reduction="batchmean")

        if self.verbose == "all" or "policy" in self.verbose:
            print("calcultaed obj and kld.")

        self._get_update_vectors(obj, kld)

        if self.verbose == "all" or "policy" in self.verbose:
            print("got full step.")

        v = self.update_vectors
        g = self.grads
        expected_improve = v.T @ g
        if self.verbose == "all" or "policy" in self.verbose:
            print(f"expected improve : {expected_improve.item()}")

        updated = False

        if self.verbose == "all" or "policy" in self.verbose:
            print("line search start.")
        for i in range(self.search_iter):
            new_network = self._update_params(self.policy_net, self.alpha, i)
            new_policies = []
            new_obj = 0
            for j in range(len(states)):
                new_pred = new_network.predict(states[j])

                new_policies.append(new_pred)
                new_obj = new_obj + new_pred[actions[j]] / (old_policies[j][actions[j]] + 1e-7) * advantages[j]

            new_obj = new_obj / len(states)
            new_kld = torch.nn.functional.kl_div(torch.log(torch.stack(new_policies) + 1e-7), old_policies,
                                                 reduction="batchmean")

            if self.verbose == "all" or "policy" in self.verbose:
                print(f"  new obj : {new_obj.item()}, new_kld : {new_kld}")
                print(f"  improve : {new_obj.item() - obj.item()}")

            if new_obj > obj and new_kld < self.delta:
                if self.verbose == "all" or "policy" in self.verbose:
                    print("  > step size passed!")

                self.policy_net.load_state_dict(new_network.state_dict())
                updated = True
                break
            elif new_obj <= obj:
                if self.verbose == "all" or "policy" in self.verbose:
                    print("  > no improve. reduce step size.")
            else:
                if self.verbose == "all" or "policy" in self.verbose:
                    print("  > kld too large. reduce step size.")

        if updated:
            self.batch_memory = []
            return updated, obj, new_obj, new_kld

        else:
            return updated, obj, obj, kld

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
