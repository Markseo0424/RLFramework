import torch
import torch.nn as nn
from torch.optim.adam import Adam
from .. import ValueOptimizer
from ...traj import Sample


class SoftValueOptim(ValueOptimizer):
    def __init__(self, lr=1e-4, alpha=0.2, gamma=0.99):
        super().__init__(
            required_list=[
                "q_1", "q_2", "v"
            ]
        )

        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma

        self.v_optim = None

    def init_optim(self):
        self.v_optim = Adam(self.v.parameters(), lr=self.lr)

    def step(self, x: Sample):
        states, actions, logprobs, rewards, next_states, constants = x.get_batch(gamma=self.gamma, level=1)

        pred_q = torch.minimum(
            self.q_1(states, actions, eval=True),
            self.q_2(states, actions, eval=True)
        )

        loss = 0.5 * torch.mean(torch.pow((
            self.v(states) -
            (pred_q - self.alpha * logprobs)
        ).item(), 2))

        self.v_optim.zero_grad()
        loss.backward()
        self.v_optim.step()

        return loss.item()
