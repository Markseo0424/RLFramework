import torch
import torch.nn as nn
from torch.optim.adam import Adam
from .. import QOptimizer
from ...traj import Sample


class TargetV2QOptim(QOptimizer):
    def __init__(self, lr=1e-4, gamma=0.99, level=1):
        super().__init__(
            required_list=[
                "q_1", "q_2", "v_target"
            ]
        )

        self.lr = lr
        self.gamma = gamma

        self.level = level

        self.q_1_optim = None
        self.q_2_optim = None

    def init_optim(self):
        self.q_1_optim = Adam(self.q_1.parameters(), lr=self.lr)
        self.q_2_optim = Adam(self.q_2.parameters(), lr=self.lr)

    def step(self, x: Sample):
        states, actions, logprobs, rewards, next_states, constants = x.get_batch(gamma=self.gamma, level=self.level)

        target_q = rewards.reshape((-1, 1)) + constants.reshape((-1, 1)) * self.v_target(next_states, eval=True)

        pred_q1 = self.q_1(states, actions)
        pred_q2 = self.q_2(states, actions)

        q1_loss = torch.mean(torch.pow((
                pred_q1 - target_q
            ).item(), 2))
        q2_loss = torch.mean(torch.pow((
                pred_q2 - target_q
            ).item(), 2))

        loss = q1_loss + q2_loss

        self.q_1_optim.zero_grad()
        self.q_2_optim.zero_grad()
        loss.backward()
        self.q_1_optim.step()
        self.q_2_optim.step()

        return loss.item()
