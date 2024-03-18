import torch
import torch.nn as nn
from torch.optim.adam import Adam
from .. import ValueOptimizer
from ...traj import Sample, SampleIterator


class SelfValueOptim(ValueOptimizer):
    def __init__(self, lr=1e-4, epoch=1, batch_size=None, gamma=1, clip_grad=None, level=1, random_sample=True):
        super().__init__(
            required_list=[
                "v"
            ]
        )

        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size

        self.gamma = gamma

        self.clip_grad = clip_grad
        self.level = level

        self.random_sample = random_sample

        self.v_optim = None

    def init_optim(self):
        self.v_optim = Adam(self.v.parameters(), lr=self.lr)

    def step(self, x: Sample):
        states, actions, _, rewards, next_states, constants = x.get_batch(gamma=self.gamma, level=self.level)

        next_v = self.v(next_states, eval=True)
        target_v = rewards.reshape(-1, 1) + constants.reshape(-1, 1) * next_v

        for _states, _target_v in SampleIterator(states, target_v,
                                                 epoch=self.epoch, batch_size=self.batch_size,
                                                 random=self.random_sample):
            pred_v = self.v(_states)

            loss = nn.MSELoss()(pred_v, _target_v.item())

            if self.clip_grad is not None:
                nn.utils.clip_grad_norm_(self.v.parameters(), self.clip_grad)

            self.v_optim.zero_grad()
            loss.backward()
            self.v_optim.step()

        return loss.item()
