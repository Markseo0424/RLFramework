import torch
import torch.nn as nn
from torch.optim.adam import Adam
from .. import PolicyOptimizer
from ...traj import Sample, SampleIterator


class ClippedSurrogatePolicyOptim(PolicyOptimizer):
    def __init__(self, lr=1e-4, epoch=10, batch_size=64, gamma=0.99, epsilon=0.2, lamda=0.5, entropy_weight=0.01, use_target_v=False):
        super().__init__(
            required_list=[
                "v_target" if use_target_v else "v", "pi"
            ]
        )

        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size

        self.gamma = gamma
        self.epsilon = epsilon
        self.lamda = lamda

        self.entropy_weight = entropy_weight

        self.use_target_v = use_target_v

        self.pi_optim = None

    def init_optim(self):
        self.pi_optim = Adam(self.pi.parameters(), lr=self.lr)

    def step(self, x: Sample):
        states, actions, _, _, _, _ = x.get_batch(gamma=self.gamma, level=1)
        advantages = x.get_GAE(self.gamma, self.lamda, self.v_target if self.use_target_v else self.v)

        old_policy = self.pi(states, eval=True)
        old_logprobs, _ = self.pi.get_logprob(old_policy, actions)

        for _states, _actions, _advantages, _old_logprobs \
                in SampleIterator(states, actions, advantages, old_logprobs,
                                  epoch=self.epoch, batch_size=self.batch_size, random=True):

            policy = self.pi(_states)
            logprobs, entropy = self.pi.get_logprob(policy, _actions)

            r = torch.exp(torch.clamp(logprobs - _old_logprobs, min=-10, max=10))

            clip_loss = torch.minimum(r * _advantages, torch.clamp(r, min=1-self.epsilon, max=1+self.epsilon) * _advantages)
            entropy_loss = torch.sum(entropy, dim=1, keepdim=True)

            loss = - torch.mean(clip_loss + self.entropy_weight * entropy_loss)

            self.pi_optim.zero_grad()
            loss.backward()
            self.pi_optim.step()

        return loss.item()
