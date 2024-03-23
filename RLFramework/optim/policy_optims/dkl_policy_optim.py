import torch
import torch.nn as nn
from torch.optim.adam import Adam
from .. import PolicyOptimizer
from ...traj import Sample, SampleIterator


class DKLPolicyOptim(PolicyOptimizer):
    def __init__(self, lr=1e-4, epoch=1, batch_size=None, alpha=0.2, gamma=0.99, random_sample=False):
        super().__init__(
            required_list=[
                "q_1", "q_2", "pi"
            ]
        )

        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size

        self.alpha = alpha
        self.gamma = gamma

        self.random_sample = random_sample

        self.pi_optim = None

    def init_optim(self):
        self.pi_optim = Adam(self.pi.parameters(), lr=self.lr)

    def step(self, x: Sample):
        states, _, _, _, _, _ = x.get_batch(gamma=self.gamma, level=1)

        for _states in SampleIterator(states,
                                      epoch=self.epoch, batch_size=self.batch_size, random=self.random_sample):

            policy = self.pi(_states)
            pred_actions, pred_logprobs = self.pi.sample_action(policy)

            pred_q = torch.minimum(
                self.q_1(_states, pred_actions),
                self.q_2(_states, pred_actions)
            )

            loss = torch.mean(
                self.alpha * pred_logprobs - pred_q
            )

            self.pi_optim.zero_grad()
            loss.backward()
            self.pi_optim.step()

        return loss.item()
