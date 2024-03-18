import torch
import torch.nn as nn
from torch.optim.adam import Adam
from .. import PolicyOptimizer
from ...traj import Sample


class DKLPolicyOptim(PolicyOptimizer):
    def __init__(self, lr=1e-4, alpha=0.2, gamma=0.99):
        super().__init__(
            required_list=[
                "q_1", "q_2", "pi"
            ]
        )

        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma

        self.pi_optim = None

    def init_optim(self):
        self.pi_optim = Adam(self.pi.parameters(), lr=self.lr)

    def step(self, x: Sample):
        states, actions, logprobs, rewards, next_states, constants = x.get_batch(gamma=self.gamma, level=1)

        policy = self.pi(states)
        pred_actions, pred_logprobs = self.pi.sample_action(policy)

        pred_q = torch.minimum(
            self.q_1(states, pred_actions),
            self.q_2(states, pred_actions)
        )

        loss = torch.mean(
            self.alpha * pred_logprobs - pred_q
        )

        self.pi_optim.zero_grad()
        loss.backward()
        self.pi_optim.step()

        return loss.item()
