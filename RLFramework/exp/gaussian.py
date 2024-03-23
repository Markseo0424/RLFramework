import numpy as np
import torch
from .exploration import Exploration
from ..space import Continuous


class Gaussian(Exploration):
    def __init__(self, action_space: Continuous = None):
        self.scale = None
        self.bias = None

        super().__init__(action_space=action_space)

    def init_space(self, action_space: Continuous):
        super().init_space(action_space)
        self.scale = (self.action_space.upper - self.action_space.lower) / 2
        self.bias = (self.action_space.upper + self.action_space.lower) / 2

    def get_policy_shape(self, action_space: Continuous):
        return (2, *action_space.shape)

    def explore(self, x, numpy=False):
        assert x.shape[-len(self.policy_shape):] == self.policy_shape, f"x must have shape of {self.policy_shape}."
        assert type(x) == torch.Tensor, "input must be torch.Tensor."
        assert self.action_space is not None, "must init action space first!"

        if len(x.shape) == len(self.policy_shape):
            unbatched = True
            x = x.reshape((1, *self.policy_shape))
        else:
            unbatched = False

        mean, logstd = x[:, 0, :], torch.clamp(x[:, 1, :], min=-10, max=10)

        dist = torch.distributions.normal.Normal(mean, torch.exp(logstd))
        action = dist.rsample()

        logprobs = dist.log_prob(action)
        action = torch.tanh(action)

        scale = torch.FloatTensor(self.scale).to(x)
        bias = torch.FloatTensor(self.bias).to(x)

        logprobs = logprobs - torch.log(scale * (1 - action.pow(2)) + 1e-6)
        logprobs = torch.sum(logprobs, dim=1, keepdim=True)
        action = action * scale + bias

        if unbatched:
            action = action.reshape(action.shape[1:])
            logprobs = logprobs.reshape(logprobs.shape[1:])

        if numpy:
            action = action.detach().cpu().numpy()
            logprobs = logprobs.detach().cpu().numpy()

        # if unbatched:
        #     print(f"action : {action}, logprobs : {logprobs}")

        return action, logprobs

    def greedy(self, x, numpy=False):
        assert x.shape[-len(self.policy_shape):] == self.policy_shape, f"x must have shape of {self.policy_shape}."

        if len(x.shape) == len(self.policy_shape):
            unbatched = True
            x = x.reshape((1, *self.policy_shape))
        else:
            unbatched = False

        mean, std = x[:, 0, :], x[:, 1, :]

        action = torch.tanh(mean)

        scale = torch.FloatTensor(self.scale).to(x)
        bias = torch.FloatTensor(self.bias).to(x)

        action = action * scale + bias

        if unbatched:
            action = action.reshape(action.shape[1:])

        if numpy:
            action = action.detach().cpu().numpy()

        return action, 0

    def get_logprob(self, x, action, numpy=False):
        assert x.shape[-len(self.policy_shape):] == self.policy_shape, f"x must have shape of {self.policy_shape}."
        assert type(x) == torch.Tensor, "input must be torch.Tensor."
        assert self.action_space is not None, "must init action space first!"

        if type(action) == np.ndarray:
            action = torch.tensor(action).to(x)

        if len(x.shape) == len(self.policy_shape):
            unbatched = True
            x = x.reshape((1, *self.policy_shape))
        else:
            unbatched = False

        assert x.shape[0] == action.shape[0], "batch size doesn't match!"

        mean, logstd = x[:, 0, :], torch.clamp(x[:, 1, :], min=-10, max=10)

        scale = torch.FloatTensor(self.scale).to(x)
        bias = torch.FloatTensor(self.bias).to(x)

        action = torch.arctanh(torch.clamp((action - bias) / scale, -1 + 1e-7, 1 - 1e-7))

        dist = torch.distributions.normal.Normal(mean, torch.exp(logstd))

        logprobs = dist.log_prob(action)
        entropy = dist.entropy()

        action = torch.tanh(action)

        logprobs = logprobs - torch.log(scale * (1 - action.pow(2)) + 1e-7)
        logprobs = torch.sum(logprobs, dim=1, keepdim=True)

        if unbatched:
            logprobs = logprobs.reshape(logprobs.shape[1:])

        if numpy:
            logprobs = logprobs.detach().cpu().numpy()

        return logprobs, entropy

