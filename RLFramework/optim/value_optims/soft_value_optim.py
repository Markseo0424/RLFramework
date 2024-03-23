import torch
import torch.nn as nn
from torch.optim.adam import Adam
from .. import ValueOptimizer
from ...traj import Sample, SampleIterator


class SoftValueOptim(ValueOptimizer):
    def __init__(self, lr=1e-4, epoch=1, batch_size=None, alpha=0.2, gamma=0.99, random_sample=False):
        super().__init__(
            required_list=[
                "q_1", "q_2", "v"
            ]
        )

        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size

        self.alpha = alpha
        self.gamma = gamma

        self.random_sample = random_sample

        self.v_optim = None

    def init_optim(self):
        self.v_optim = Adam(self.v.parameters(), lr=self.lr)

    def step(self, x: Sample):
        states, actions, logprobs, _, _, _ = x.get_batch(gamma=self.gamma, level=1)

        for _states, _actions, _logprobs in SampleIterator(
                states, actions, logprobs,
                epoch=self.epoch, batch_size=self.batch_size, random=self.random_sample):
            pred_q = torch.minimum(
                self.q_1(_states, _actions, eval=True),
                self.q_2(_states, _actions, eval=True)
            )

            loss = 0.5 * torch.mean(torch.pow((
                self.v(_states) -
                (pred_q - self.alpha * _logprobs)
            ).item(), 2))

            self.v_optim.zero_grad()
            loss.backward()
            self.v_optim.step()

        return loss.item()
