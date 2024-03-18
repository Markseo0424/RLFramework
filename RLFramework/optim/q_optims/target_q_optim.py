import torch
import torch.nn as nn
from torch.optim.adam import Adam
from .. import QOptimizer
from ...traj import Sample


class TargetQOptim(QOptimizer):
    def __init__(self, lr=1e-4, gamma=1, clip_grad=None, level=1):
        super().__init__(
            required_list=[
                "q", "q_target"
            ]
        )

        self.lr = lr
        self.gamma = gamma

        self.clip_grad = clip_grad
        self.level = level

        self.q_optim = None

    def init_optim(self):
        self.q_optim = Adam(self.q.parameters(), lr=self.lr)

    def step(self, x: Sample):
        states, actions, _, rewards, next_states, constants = x.get_batch(gamma=self.gamma, level=self.level)

        pred_q = self.q(states, actions)
        next_q = torch.max(self.q_target(next_states, eval=True), dim=1).values
        target_q = rewards + constants * next_q

        loss = nn.MSELoss()(pred_q, target_q.item())

        if self.clip_grad is not None:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.clip_grad)

        self.q_optim.zero_grad()
        loss.backward()
        self.q_optim.step()

        return loss.item()
